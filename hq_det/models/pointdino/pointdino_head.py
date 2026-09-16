# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple

import torch
from mmcv.cnn import Linear
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmdet.models.dense_heads.dino_head import DINOHead
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean


@MODELS.register_module()
class PointDINOHead(DINOHead):
    """DINO head with two-coordinate regression and point supervision."""

    def __init__(self, *args, point_euclidean_weight: float = 0.0, **kwargs):
        """Configure the pixel-distance term, normalized by eight pixels."""
        self.point_euclidean_weight = point_euclidean_weight
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize classification and 2D point regression branches."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 2))
        reg_branch = nn.Sequential(*reg_branch)
        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                point_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Convert Point-DINO outputs into point predictions."""
        assert len(cls_score) == len(point_pred)
        assert self.loss_cls.use_sigmoid
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        cls_score = cls_score.sigmoid()
        scores, indexes = cls_score.reshape(-1).topk(max_per_img)
        det_labels = indexes % self.num_classes
        point_indexes = indexes // self.num_classes
        det_points = point_pred[point_indexes].clone()
        img_h, img_w = img_meta['img_shape']
        det_points[:, 0] *= img_w
        det_points[:, 1] *= img_h
        if rescale:
            scale_factor = img_meta.get('scale_factor', (1.0, 1.0))
            if not isinstance(scale_factor, torch.Tensor):
                scale_factor = det_points.new_tensor(scale_factor)
            det_points /= scale_factor
        results = InstanceData()
        results.points = det_points
        results.scores = scores
        results.labels = det_labels
        return results

    def _get_targets_single(self, cls_score: Tensor, point_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute classification and point targets for one image."""
        num_points = point_pred.size(0)
        pred_instances = InstanceData(scores=cls_score, points=point_pred)
        assign_result = self.assigner.assign(pred_instances=pred_instances,
                                             gt_instances=gt_instances,
                                             img_meta=img_meta)
        gt_points = gt_instances.points
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(assign_result.gt_inds > 0,
                                 as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0,
                                 as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_points = gt_points[pos_assigned_gt_inds.long(), :]
        labels = gt_points.new_full((num_points, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_points.new_ones(num_points)
        point_targets = torch.zeros_like(point_pred, dtype=gt_points.dtype)
        point_weights = torch.zeros_like(point_pred, dtype=gt_points.dtype)
        point_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']
        factor = gt_points.new_tensor([img_w, img_h]).unsqueeze(0)
        pos_gt_points_normalized = pos_gt_points / factor
        point_targets[pos_inds] = pos_gt_points_normalized
        return (labels, label_weights, point_targets, point_weights, pos_inds,
                neg_inds)

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Split point outputs, supporting training with denoising disabled."""
        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            all_layers_denoising_cls_scores = all_layers_cls_scores[:, :, :
                                                                    num_denoising_queries, :]
            all_layers_denoising_bbox_preds = all_layers_bbox_preds[:, :, :
                                                                    num_denoising_queries, :]
            all_layers_matching_cls_scores = all_layers_cls_scores[:, :,
                                                                   num_denoising_queries:, :]
            all_layers_matching_bbox_preds = all_layers_bbox_preds[:, :,
                                                                   num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds)

    def _loss_point_euclidean(self, point_preds, point_targets, point_weights,
                              batch_img_metas, avg_factor):
        """Pixel-space Euclidean distance loss."""
        num_imgs = len(batch_img_metas)
        if point_preds.dim() == 3:
            _, num_queries, _ = point_preds.shape
        elif point_preds.dim() == 2:
            assert point_preds.size(-1) == 2
            assert point_preds.size(0) % num_imgs == 0
            num_queries = point_preds.size(0) // num_imgs
            point_preds = point_preds.reshape(num_imgs, num_queries, 2)
        else:
            raise ValueError(
                f'Unexpected point_preds shape: {tuple(point_preds.shape)}')
        point_targets = point_targets.reshape(num_imgs, num_queries, 2)
        point_weights = point_weights.reshape(num_imgs, num_queries, 2)
        point_scales = []
        for img_meta in batch_img_metas:
            img_h, img_w = img_meta['img_shape'][:2]
            point_scales.append(point_preds.new_tensor([img_w, img_h]))
        point_scales = torch.stack(point_scales, dim=0).unsqueeze(1)
        delta_pixel = (point_preds - point_targets) * point_scales
        eps = 1e-06
        distance = torch.sqrt((delta_pixel**2).sum(dim=-1) + eps)
        positive_weights = point_weights[..., 0]
        loss = (distance * positive_weights).sum() / max(
            float(avg_factor), 1.0)
        loss = loss / 8.0
        loss = loss * self.point_euclidean_weight
        return loss

    def loss_by_feat_single(
            self, cls_scores: Tensor, point_preds: Tensor,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict]) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute classification loss and point L1 loss for one decoder layer."""
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        point_preds_list = [point_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, point_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, point_targets_list,
         point_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        point_targets = torch.cat(point_targets_list, 0)
        point_weights = torch.cat(point_weights_list, 0)
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=cls_avg_factor)
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        point_preds = point_preds.reshape(-1, 2)
        loss_point = self.loss_bbox(point_preds,
                                    point_targets,
                                    point_weights,
                                    avg_factor=num_total_pos)
        loss_point_euclidean = self._loss_point_euclidean(
            point_preds=point_preds,
            point_targets=point_targets,
            point_weights=point_weights,
            batch_img_metas=batch_img_metas,
            avg_factor=num_total_pos)
        return (loss_cls, loss_point, loss_point_euclidean)

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_point_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss for Point-DINO."""
        assert batch_gt_instances_ignore is None
        (all_layers_matching_cls_scores, all_layers_matching_point_preds,
         all_layers_denoising_cls_scores,
         all_layers_denoising_point_preds) = self.split_outputs(
             all_layers_cls_scores, all_layers_point_preds, dn_meta)
        losses_cls, losses_point, losses_point_euclidean = multi_apply(
            self.loss_by_feat_single,
            all_layers_matching_cls_scores,
            all_layers_matching_point_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)
        loss_dict = dict()
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_point'] = losses_point[-1]
        loss_dict['loss_point_euclidean'] = losses_point_euclidean[-1]
        for num_dec_layer, (loss_cls_i, loss_point_i,
                            loss_point_euclidean_i) in enumerate(
                                zip(losses_cls[:-1], losses_point[:-1],
                                    losses_point_euclidean[:-1])):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_point'] = loss_point_i
            loss_dict[
                f'd{num_dec_layer}.loss_point_euclidean'] = loss_point_euclidean_i
        if enc_cls_scores is not None:
            enc_point_preds = enc_bbox_preds[..., :2]
            enc_loss_cls, enc_loss_point, enc_loss_point_euclidean = self.loss_by_feat_single(
                enc_cls_scores,
                enc_point_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_point'] = enc_loss_point
            loss_dict['enc_loss_point_euclidean'] = enc_loss_point_euclidean
        if all_layers_denoising_cls_scores is not None:
            assert all_layers_denoising_point_preds is not None
            assert dn_meta is not None
            dn_losses_cls, dn_losses_point, dn_losses_point_euclidean = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_point_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_point'] = dn_losses_point[-1]
            loss_dict['dn_loss_point_euclidean'] = dn_losses_point_euclidean[
                -1]
            for num_dec_layer, (loss_cls_i, loss_point_i,
                                loss_point_euclidean_i) in enumerate(
                                    zip(dn_losses_cls[:-1],
                                        dn_losses_point[:-1],
                                        dn_losses_point_euclidean[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_point'] = loss_point_i
                loss_dict[
                    f'd{num_dec_layer}.dn_loss_point_euclidean'] = loss_point_euclidean_i
        return loss_dict

    def _loss_dn_single(
            self, dn_cls_scores: Tensor, dn_point_preds: Tensor,
            batch_gt_instances: InstanceList, batch_img_metas: List[dict],
            dn_meta: Dict[str, int]) -> Tuple[Tensor, Tensor, Tensor]:
        """Point denoising loss for one decoder layer."""
        targets = self.get_dn_targets(batch_gt_instances, batch_img_metas,
                                      dn_meta)
        (labels_list, label_weights_list, point_targets_list,
         point_weights_list, num_total_pos, num_total_neg) = targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        point_targets = torch.cat(point_targets_list, 0)
        point_weights = torch.cat(point_weights_list, 0)
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        if len(cls_scores) > 0:
            loss_cls = self.loss_cls(cls_scores,
                                     labels,
                                     label_weights,
                                     avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(1,
                                   dtype=cls_scores.dtype,
                                   device=cls_scores.device)
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        loss_point_euclidean = self._loss_point_euclidean(
            point_preds=dn_point_preds,
            point_targets=point_targets,
            point_weights=point_weights,
            batch_img_metas=batch_img_metas,
            avg_factor=num_total_pos)
        dn_point_preds_flat = dn_point_preds.reshape(-1, 2)
        loss_point = self.loss_bbox(dn_point_preds_flat,
                                    point_targets,
                                    point_weights,
                                    avg_factor=num_total_pos)
        return (loss_cls, loss_point, loss_point_euclidean)

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
        """Get Point-DN targets for one image."""
        gt_points = gt_instances.points
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_points.device
        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            pos_assigned_gt_inds = t.unsqueeze(0).repeat(num_groups,
                                                         1).flatten()
            pos_inds = torch.arange(num_groups,
                                    dtype=torch.long,
                                    device=device)
            pos_inds = (pos_inds.unsqueeze(1) * num_queries_each_group +
                        t).flatten()
        else:
            pos_inds = gt_points.new_tensor([], dtype=torch.long)
            pos_assigned_gt_inds = gt_points.new_tensor([], dtype=torch.long)
        neg_inds = pos_inds + num_queries_each_group // 2
        labels = gt_points.new_full((num_denoising_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_points.new_ones(num_denoising_queries)
        point_targets = torch.zeros(num_denoising_queries,
                                    2,
                                    device=device,
                                    dtype=gt_points.dtype)
        point_weights = torch.zeros(num_denoising_queries,
                                    2,
                                    device=device,
                                    dtype=gt_points.dtype)
        point_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']
        factor = gt_points.new_tensor([img_w, img_h]).unsqueeze(0)
        gt_points_normalized = gt_points / factor
        if len(gt_labels) > 0:
            point_targets[pos_inds] = gt_points_normalized[
                pos_assigned_gt_inds]
        return (labels, label_weights, point_targets, point_weights, pos_inds,
                neg_inds)
