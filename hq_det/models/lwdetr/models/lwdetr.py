# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
LW-DETR模型和损失函数类
"""
import copy
import math
from typing import Callable
import torch
import torch.nn.functional as F
from torch import nn

from ..util import box_ops
from ..util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer


class LWDETR(nn.Module):
    def __init__(self,
                 backbone,
                 transformer,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 group_detr=1,
                 two_stage=False,
                 lite_refpoint_refine=False,
                 bbox_reparam=False):
        """ 初始化模型
        
        参数:
            backbone: 主干网络模块，用于提取图像特征
            transformer: Transformer架构模块，用于特征转换
            num_classes: 目标类别数量
            num_queries: 查询数量，即模型可以检测的最大目标数量
            aux_loss: 是否使用辅助损失（每个解码器层的损失）
            group_detr: 用于加速DETR训练的组数，默认为1
            two_stage: 是否使用两阶段检测
            lite_refpoint_refine: 是否使用轻量级参考点优化
            bbox_reparam: 是否使用边界框重参数化
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        # 分类头和边界框预测头
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # 查询嵌入
        query_dim = 4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        # 迭代更新相关设置
        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        # 初始化分类头的偏置，用于focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # 初始化边界框预测头
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # 两阶段检测相关设置
        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)])
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)])

        self._export = False

    def export(self):
        """ 导出模型设置 """
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward(self, samples: NestedTensor, targets=None):
        """ 前向传播
        
        参数:
            samples: NestedTensor类型，包含:
                - samples.tensor: 批次图像，形状为 [batch_size x 3 x H x W]
                - samples.mask: 填充像素的二进制掩码，形状为 [batch_size x H x W]
            targets: 目标信息列表，每个元素是一个字典，包含标签和边界框信息
            
        返回:
            包含以下元素的字典:
                - pred_logits: 所有查询的分类logits，形状为 [batch_size x num_queries x num_classes]
                - pred_boxes: 所有查询的归一化边界框坐标，格式为(center_x, center_y, width, height)
                - aux_outputs: 可选的辅助输出，仅在启用辅助损失时返回
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            
        # 提取特征
        features, poss = self.backbone(samples)

        # 处理特征
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        # 选择查询嵌入
        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            # 推理时只使用一个组
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.query_feat.weight[:self.num_queries]

        # Transformer处理
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight)

        # 预测边界框
        if self.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(hs)
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat(
                [outputs_coord_cxcy, outputs_coord_wh], dim=-1
            )
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

        # 预测类别
        outputs_class = self.class_embed(hs)

        # 组织输出
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        # 处理两阶段检测的输出
        if self.two_stage:
            hs_enc_list = hs_enc.split(self.num_queries, dim=1)
            cls_enc = []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc.append(cls_enc_gidx)
            cls_enc = torch.cat(cls_enc, dim=1)
            out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
        return out

    def forward_export(self, tensors):
        """ 导出模式的前向传播 """
        srcs, _, poss = self.backbone(tensors)
        # 推理时只使用一个组
        refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
        query_feat_weight = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, None, poss, refpoint_embed_weight, query_feat_weight)

        if self.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(hs)
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat(
                [outputs_coord_cxcy, outputs_coord_wh], dim=-1
            )
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
        outputs_class = self.class_embed(hs)
        return outputs_coord, outputs_class

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        """ 设置辅助损失
        
        这是一个变通方法，用于使torchscript正常工作，因为torchscript不支持非均匀值的字典。
        """
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        """ 更新drop path率 """
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        for i in range(vit_encoder_num_layers):
            if hasattr(self.backbone[0].encoder.blocks[i].drop_path, 'drop_prob'):
                self.backbone[0].encoder.blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        """ 更新dropout率 """
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


class SetCriterion(nn.Module):
    """ 损失函数计算类
    
    计算LW-DETR的损失，包括两个步骤：
    1) 计算预测框和真实框之间的匈牙利匹配
    2) 对每对匹配的预测框和真实框计算损失（分类损失和边界框损失）
    """
    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 focal_alpha,
                 losses,
                 group_detr=1,
                 sum_group_losses=False,
                 use_varifocal_loss=False,
                 use_position_supervised_loss=False,
                 ia_bce_loss=False,):
        """ 初始化损失函数
        
        参数:
            num_classes: 目标类别数量（不包括背景类）
            matcher: 用于计算预测框和真实框之间匹配的模块
            weight_dict: 损失权重字典
            focal_alpha: Focal Loss中的alpha参数
            losses: 要使用的损失函数列表
            group_detr: 用于加速DETR训练的组数
            sum_group_losses: 是否对组损失求和
            use_varifocal_loss: 是否使用Varifocal Loss
            use_position_supervised_loss: 是否使用位置监督损失
            ia_bce_loss: 是否使用IoU感知的BCE损失
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.sum_group_losses = sum_group_losses
        self.use_varifocal_loss = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss = ia_bce_loss

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """ 计算分类损失（二元focal loss）
        
        参数:
            outputs: 模型输出
            targets: 目标信息
            indices: 匹配索引
            num_boxes: 框的数量
            log: 是否记录日志
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            # IoU感知的BCE损失
            alpha = self.focal_alpha
            gamma = 2 
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets = torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            prob = src_logits.sigmoid()
            
            # 初始化正样本和负样本权重
            pos_weights = torch.zeros_like(src_logits)
            neg_weights = prob ** gamma

            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)

            t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()

            pos_weights[pos_ind] = t
            neg_weights[pos_ind] = 1 - t
            loss_ce = - pos_weights * prob.log() - neg_weights * (1 - prob).log()
            loss_ce = loss_ce.sum() / num_boxes

        elif self.use_position_supervised_loss:
            # 位置监督损失
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets = torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            pos_ious_func = pos_ious

            cls_iou_func_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_func_targets[pos_ind] = pos_ious_func
            norm_cls_iou_func_targets = cls_iou_func_targets \
                / (cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(1, True) + 1e-8)
            loss_ce = position_supervised_loss(src_logits, norm_cls_iou_func_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        elif self.use_varifocal_loss:
            # Varifocal损失
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets = torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()

            cls_iou_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_targets[pos_ind] = pos_ious
            loss_ce = sigmoid_varifocal_loss(src_logits, cls_iou_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        else:
            # 标准focal loss
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:,:,:-1]
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # 计算分类错误率
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ 计算基数损失
        
        计算预测的非空框数量与真实框数量之间的绝对误差。
        这不是真正的损失，仅用于日志记录，不传播梯度。
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # 计算预测的非背景类数量
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """ 计算边界框损失
        
        计算L1回归损失和GIoU损失。
        目标框格式为(center_x, center_y, w, h)，已归一化到图像尺寸。
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        """ 获取源序列的排列索引 """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """ 获取目标序列的排列索引 """
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """ 获取指定类型的损失 """
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ 计算损失
        
        参数:
            outputs: 模型输出的字典
            targets: 目标信息列表，每个元素是一个字典
        """
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 获取最后一层输出与目标的匹配
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        # 计算目标框数量的平均值，用于归一化
        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 处理辅助损失
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # 仅对最后一层启用日志记录
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 处理编码器输出
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # 仅对最后一层启用日志记录
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """ Focal Loss
    
    RetinaNet中使用的损失函数：https://arxiv.org/abs/1708.02002
    
    参数:
        inputs: 任意形状的浮点张量，每个样本的预测值
        targets: 与inputs相同形状的浮点张量，存储每个元素的二元分类标签
        alpha: 平衡正负样本的权重因子，范围(0,1)
        gamma: 调制因子的指数，用于平衡难易样本
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def sigmoid_varifocal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """ Varifocal Loss
    
    参数:
        inputs: 预测值
        targets: 目标值
        num_boxes: 框的数量
        alpha: focal loss的alpha参数
        gamma: focal loss的gamma参数
    """
    prob = inputs.sigmoid()
    focal_weight = targets * (targets > 0.0).float() + \
            (1 - alpha) * (prob - targets).abs().pow(gamma) * \
            (targets <= 0.0).float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * focal_weight

    return loss.mean(1).sum() / num_boxes


def position_supervised_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """ 位置监督损失
    
    参数:
        inputs: 预测值
        targets: 目标值
        num_boxes: 框的数量
        alpha: focal loss的alpha参数
        gamma: focal loss的gamma参数
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * (torch.abs(targets - prob) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcess(nn.Module):
    """ 后处理模块
    
    将模型输出转换为COCO API期望的格式
    """
    def __init__(self, num_select=300) -> None:
        """
        参数:
            num_select: 选择的预测框数量
        """
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ 执行后处理
        
        参数:
            outputs: 模型的原始输出
            target_sizes: 形状为[batch_size x 2]的张量，包含批次中每个图像的尺寸
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # 获取top-k预测
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # 将相对坐标[0,1]转换为绝对坐标[0,height]
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ 简单的多层感知机（也称为FFN）"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: 层数
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """ 前向传播 """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    """ 构建模型
    
    参数:
        args: 配置参数
    """
    # num_classes的命名可能有些误导性
    # 它实际上对应于max_obj_id + 1，其中max_obj_id是数据集中类别的最大ID
    # 例如，COCO的max_obj_id是90，所以我们传递num_classes为91
    # 对于只有一个类别且ID为1的数据集，应该传递num_classes为2（max_obj_id + 1）
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "o365":
        num_classes = 366
    if args.id2names is not None:
        num_classes = len(args.id2names)
    device = torch.device(args.device)

    # 构建主干网络
    backbone = build_backbone(args)

    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args)

    # 构建LW-DETR模型
    model = LWDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,   # 查询数量
        aux_loss=args.aux_loss,         # 辅助损失
        group_detr=args.group_detr,     # 组数量
        two_stage=args.two_stage,       # 两阶段
        lite_refpoint_refine=args.lite_refpoint_refine, # 轻量级参考点细化
        bbox_reparam=args.bbox_reparam, # 边界框重参数化
    )
    
    # 构建匹配器和损失函数
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    # 处理辅助损失
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    try:
        sum_group_losses = args.sum_group_losses
    except:
        sum_group_losses = False
        
    # 构建损失函数
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses, 
                             group_detr=args.group_detr, sum_group_losses=sum_group_losses,
                             use_varifocal_loss = args.use_varifocal_loss,
                             use_position_supervised_loss=args.use_position_supervised_loss,
                             ia_bce_loss=args.ia_bce_loss)
    criterion.to(device)
    
    # 构建后处理器
    postprocessors = {'bbox': PostProcess(num_select=args.num_select)}

    return model, criterion, postprocessors