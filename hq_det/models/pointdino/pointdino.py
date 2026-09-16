# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from mmdet.models.detectors.deformable_detr import DeformableDETR
from mmdet.models.detectors.dino import DINO
from mmdet.models.layers import (DeformableDetrTransformerEncoder,
                                 SinePositionalEncoding)
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList

from .pointdino_fidt_head import PointDINOFIDTAuxHead
from .pointdino_layers import (PointDINOCdnQueryGenerator,
                               PointDINOTransformerDecoder)


@MODELS.register_module()
class PointDINO(DINO):
    """Point-supervised DINO with isolated queries, heads and optional FIDT."""

    def __init__(self,
                 *args,
                 dn_cfg=None,
                 use_dn=True,
                 point_fidt_head=None,
                 **kwargs):
        # Skip DINO's box DN construction so initialization draws match the
        # source PointDINO and no unused box module is registered.
        DeformableDETR.__init__(self, *args, **kwargs)
        self.use_dn = use_dn
        assert self.as_two_stage, 'as_two_stage must be True for PointDINO'
        assert self.with_box_refine, 'with_box_refine must be True for PointDINO'
        dn_cfg = {} if dn_cfg is None else dict(dn_cfg)
        for key in ('num_classes', 'embed_dims', 'num_matching_queries'):
            if key in dn_cfg:
                raise ValueError(
                    f'{key} is configured by the PointDINO detector.')
        dn_type = dn_cfg.pop('type', 'PointDINOCdnQueryGenerator')
        if dn_type not in ('PointDINOCdnQueryGenerator',
                           PointDINOCdnQueryGenerator):
            raise ValueError('dn_cfg.type must be PointDINOCdnQueryGenerator.')
        self.dn_query_generator = PointDINOCdnQueryGenerator(
            num_classes=self.bbox_head.num_classes,
            embed_dims=self.embed_dims,
            num_matching_queries=self.num_queries,
            **dn_cfg)
        self.point_fidt_head = None
        if point_fidt_head is not None:
            fidt_cfg = dict(point_fidt_head)
            enabled = fidt_cfg.pop('enabled', True)
            fidt_type = fidt_cfg.pop('type', 'PointDINOFIDTAuxHead')
            if fidt_type not in ('PointDINOFIDTAuxHead', PointDINOFIDTAuxHead):
                raise ValueError(
                    'point_fidt_head.type must be PointDINOFIDTAuxHead.')
            if enabled and fidt_cfg.get('fidt_loss_weight', 0.0) != 0:
                self.point_fidt_head = PointDINOFIDTAuxHead(**fidt_cfg)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Keep all Stage 3 losses and optionally add one spatial FIDT loss."""
        if self.point_fidt_head is None:
            return super().loss(batch_inputs, batch_data_samples)
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(**head_inputs_dict,
                                     batch_data_samples=batch_data_samples)
        # Share the original neck tensor, retaining gradients into the neck.
        shared_feature = max(img_feats,
                             key=lambda feat: feat.shape[-2] * feat.shape[-1])
        losses.update(
            self.point_fidt_head.loss(shared_feature, batch_data_samples,
                                      batch_inputs.shape[-2:]))
        return losses

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = PointDINOTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            'embed_dims should be exactly 2 times of num_feats. '
            f'Found {self.embed_dims} and {num_feats}.')
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def gen_encoder_output_proposals(
            self, memory: Tensor, memory_mask: Tensor,
            spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate 2D point proposals from encoder feature locations.

        Stage 2 Point-DINO:
        the two-stage encoder proposals contain only normalized
        point coordinates (x, y), without width and height.
        """
        bs = memory.size(0)
        proposals = []
        _cur = 0
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW
            if memory_mask is not None:
                mask_flatten_ = memory_mask[:, _cur:_cur + H * W].view(
                    bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0],
                                    1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0],
                                    1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            else:
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0,
                               H - 1,
                               H,
                               dtype=torch.float32,
                               device=memory.device),
                torch.linspace(0,
                               W - 1,
                               W,
                               dtype=torch.float32,
                               device=memory.device),
                indexing='ij')
            grid = torch.cat([grid_x.unsqueeze(-1),
                              grid_y.unsqueeze(-1)],
                             dim=-1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            proposal = grid.view(bs, -1, 2)
            proposals.append(proposal)
            _cur += H * W
        output_proposals = torch.cat(proposals, dim=1)
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)).sum(
                -1, keepdim=True) == output_proposals.shape[-1]
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))
        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(
                memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)
        return (output_memory, output_proposals)

    def pre_decoder(self,
                    memory: Tensor,
                    memory_mask: Tensor,
                    spatial_shapes: Tensor,
                    batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Select two-dimensional encoder proposals and prepend point DN queries."""
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory)[..., :2] + output_proposals
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0],
                                  k=self.num_queries,
                                  dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 2))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()
        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training and self.use_dn:
            dn_label_query, dn_point_query, dn_mask, dn_meta = self.dn_query_generator(
                batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_point_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = (None, None)
        reference_points = reference_points.sigmoid()
        decoder_inputs_dict = dict(query=query,
                                   memory=memory,
                                   reference_points=reference_points,
                                   dn_mask=dn_mask)
        head_inputs_dict = dict(enc_outputs_class=topk_score,
                                enc_outputs_coord=topk_coords,
                                dn_meta=dn_meta) if self.training else dict()
        return (decoder_inputs_dict, head_inputs_dict)

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        """Decode points and connect the label embedding for empty DN batches."""
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs)
        if self.training and self.use_dn and (query.shape[1]
                                              == self.num_queries):
            # Query length is dimension 1; len(query) is the batch size.
            # The connected zero prevents unused DN parameters on empty GTs.
            inter_states[0] += self.dn_query_generator.label_embedding.weight[
                0, 0] * 0.0
        decoder_outputs_dict = dict(hidden_states=inter_states,
                                    references=list(references))
        return decoder_outputs_dict
