# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch import Tensor, nn

from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerDecoder)
from mmdet.models.layers.transformer.dino_layers import (CdnQueryGenerator,
                                                         DinoTransformerDecoder
                                                         )
from mmdet.models.layers.transformer.utils import (MLP, coordinate_to_encoding,
                                                   inverse_sigmoid)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType


@MODELS.register_module()
class PointDINOTransformerDecoder(DinoTransformerDecoder):
    """DINO decoder preserving two-dimensional point references at every layer."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        # Reuse unchanged decoder layers without allocating the box projection.
        DeformableDetrTransformerDecoder._init_layers(self)
        self.ref_point_head = MLP(self.embed_dims, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Refine normalized points and retain look-forward-twice references."""
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :,
                                                          None] * torch.cat([
                                                              valid_ratios,
                                                              valid_ratios
                                                          ], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :,
                                                          None] * valid_ratios[:,
                                                                               None]
            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            query = layer(query,
                          query_pos=query_pos,
                          value=value,
                          key_padding_mask=key_padding_mask,
                          self_attn_mask=self_attn_mask,
                          spatial_shapes=spatial_shapes,
                          level_start_index=level_start_index,
                          valid_ratios=valid_ratios,
                          reference_points=reference_points_input,
                          **kwargs)
            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=0.001)
                elif reference_points.shape[-1] == 2:
                    new_reference_points = tmp[..., :2] + inverse_sigmoid(
                        reference_points, eps=0.001)
                else:
                    raise ValueError(
                        'reference_points must have last dimension 2 or 4, '
                        f'but got {reference_points.shape[-1]}')
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
        if self.return_intermediate:
            return (torch.stack(intermediate),
                    torch.stack(intermediate_reference_points))
        return (query, reference_points)


@MODELS.register_module()
class PointDINOCdnQueryGenerator(CdnQueryGenerator):
    """Contrastive denoising queries from two-dimensional GT points."""

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 point_noise_scale: float = 0.05,
                 group_cfg: OptConfigType = None) -> None:
        super().__init__(num_classes=num_classes,
                         embed_dims=embed_dims,
                         num_matching_queries=num_matching_queries,
                         label_noise_scale=label_noise_scale,
                         box_noise_scale=1.0,
                         group_cfg=group_cfg)
        if point_noise_scale <= 0:
            raise ValueError('point_noise_scale must be greater than 0.')
        self.point_noise_scale = float(point_noise_scale)

    def __call__(self, batch_data_samples: SampleList) -> tuple:
        """Generate contrastive denoising queries from GT points."""
        gt_labels_list = []
        gt_points_list = []
        for sample in batch_data_samples:
            img_h, img_w = sample.img_shape
            gt_points = sample.gt_instances.points
            factor = gt_points.new_tensor([img_w, img_h]).unsqueeze(0)
            gt_points_normalized = gt_points / factor
            gt_points_list.append(gt_points_normalized)
            gt_labels_list.append(sample.gt_instances.labels)
        gt_labels = torch.cat(gt_labels_list, dim=0)
        gt_points = torch.cat(gt_points_list, dim=0)
        num_target_list = [len(points) for points in gt_points_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)
        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)
        dn_point_query = self.generate_dn_point_query(gt_points, num_groups)
        batch_idx = torch.cat([
            torch.full_like(labels.long(), i)
            for i, labels in enumerate(gt_labels_list)
        ])
        dn_label_query, dn_point_query = self.collate_dn_queries(
            dn_label_query, dn_point_query, batch_idx, len(batch_data_samples),
            num_groups)
        attn_mask = self.generate_dn_mask(max_num_target,
                                          num_groups,
                                          device=dn_label_query.device)
        dn_meta = dict(num_denoising_queries=int(max_num_target * 2 *
                                                 num_groups),
                       num_denoising_groups=num_groups)
        return (dn_label_query, dn_point_query, attn_mask, dn_meta)

    def generate_dn_point_query(self, gt_points: Tensor,
                                num_groups: int) -> Tensor:
        """Generate positive and negative noisy point queries.

        Positive DN points:
            distance from GT is in [0, point_noise_scale).

        Negative DN points:
            distance from GT is in
            [point_noise_scale, 2 * point_noise_scale).

        Coordinates are normalized to [0, 1].
        """
        device = gt_points.device
        gt_points_expand = gt_points.repeat(2 * num_groups, 1)
        num_gt = len(gt_points)
        positive_idx = torch.arange(num_gt, dtype=torch.long, device=device)
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += 2 * num_gt * torch.arange(
            num_groups, dtype=torch.long, device=device)[:, None]
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + num_gt
        num_noisy_targets = len(gt_points_expand)
        angle = torch.rand(
            num_noisy_targets, 1, device=device,
            dtype=gt_points.dtype) * (2.0 * torch.pi)
        radius = torch.rand(num_noisy_targets,
                            1,
                            device=device,
                            dtype=gt_points.dtype)
        radius[negative_idx] += 1.0
        direction = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)
        offset = direction * radius * self.point_noise_scale
        noisy_points_expand = gt_points_expand + offset
        noisy_points_expand = noisy_points_expand.clamp(min=0.0, max=1.0)
        dn_point_query = inverse_sigmoid(noisy_points_expand, eps=0.001)
        return dn_point_query

    def collate_dn_queries(self, input_label_query: Tensor,
                           input_point_query: Tensor, batch_idx: Tensor,
                           batch_size: int, num_groups: int) -> Tuple[Tensor]:
        """Collate point DN queries into batched tensors."""
        device = input_label_query.device
        num_target_list = [
            torch.sum(batch_idx == idx) for idx in range(batch_size)
        ]
        max_num_target = max(num_target_list)
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        map_query_index = torch.cat([
            torch.arange(num_target, device=device)
            for num_target in num_target_list
        ])
        map_query_index = torch.cat([
            map_query_index + max_num_target * i for i in range(2 * num_groups)
        ]).long()
        batch_idx_expand = batch_idx.repeat(2 * num_groups, 1).view(-1)
        mapper = (batch_idx_expand, map_query_index)
        batched_label_query = torch.zeros(batch_size,
                                          num_denoising_queries,
                                          self.embed_dims,
                                          device=device)
        batched_point_query = torch.zeros(batch_size,
                                          num_denoising_queries,
                                          2,
                                          device=device)
        batched_label_query[mapper] = input_label_query
        batched_point_query[mapper] = input_point_query
        return (batched_label_query, batched_point_query)
