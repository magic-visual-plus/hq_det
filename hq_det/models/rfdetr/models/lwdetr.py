# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
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
LW-DETR model and criterion classes
"""
import copy
import math
from typing import Callable
import torch
import torch.nn.functional as F
from torch import nn

from hq_det.models.rfdetr.util import box_ops
from hq_det.models.rfdetr.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)

from hq_det.models.rfdetr.models.backbone import build_backbone
from hq_det.models.rfdetr.models.matcher import build_matcher
from hq_det.models.rfdetr.models.transformer import build_transformer

class LWDETR(nn.Module):
    """ This is the Group DETR v3 module that performs object detection """
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
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            group_detr: Number of groups to speed detr training. Default is 1.
            lite_refpoint_refine: TODO
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        query_dim=4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        # iter update
        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # two_stage
        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)])
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)])

        self._export = False

    def reinitialize_detection_head(self, num_classes):
        # Create new classification head
        del self.class_embed
        self.add_module("class_embed", nn.Linear(self.transformer.d_model, num_classes))
        
        # Initialize with focal loss bias adjustment
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        if self.two_stage:
            del self.transformer.enc_out_class_embed
            self.transformer.add_module("enc_out_class_embed", nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(self.group_detr)]))


    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            # only use one group in inference
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight)

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

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc.append(cls_enc_gidx)
            cls_enc = torch.cat(cls_enc, dim=1)
            out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
        return out

    def forward_export(self, tensors):
        srcs, _, poss = self.backbone(tensors)
        # only use one group in inference
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
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        """ """
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        for i in range(vit_encoder_num_layers):
            if hasattr(self.backbone[0].encoder, 'blocks'): # Not aimv2
                if hasattr(self.backbone[0].encoder.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.blocks[i].drop_path.drop_prob = dp_rates[i]
            else: # aimv2
                if hasattr(self.backbone[0].encoder.trunk.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.trunk.blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


class SetCriterion(nn.Module):
    """ 这个类计算条件DETR的损失函数。
    计算过程分为两个步骤：
        1) 计算真实框和模型输出之间的匈牙利匹配
        2) 监督每对匹配的真实值/预测值（监督类别和边界框）
    """
    def __init__(self,
                 num_classes,  # 类别数量，不包括特殊的无目标类别
                 matcher,      # 能够计算目标和提议之间匹配的模块
                 weight_dict,  # 包含损失名称作为键、相对权重作为值的字典
                 focal_alpha,  # Focal Loss中的alpha参数
                 losses,       # 要应用的所有损失列表
                 group_detr=1, # 加速DETR训练的组数，默认为1
                 sum_group_losses=False,      # 是否对组损失求和
                 use_varifocal_loss=False,    # 是否使用Varifocal损失
                 use_position_supervised_loss=False,  # 是否使用位置监督损失
                 ia_bce_loss=False,):         # 是否使用IA-BCE损失
        """ 创建损失准则。
        参数:
            num_classes: 目标类别数量，省略特殊的无目标类别
            matcher: 能够计算目标和提议之间匹配的模块
            weight_dict: 包含损失名称作为键、相对权重作为值的字典
            losses: 要应用的所有损失列表。参见get_loss了解可用损失列表
            focal_alpha: Focal Loss中的alpha参数
            group_detr: 加速DETR训练的组数，默认为1
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
        """分类损失（二元focal loss）
        targets字典必须包含键"labels"，包含维度为[nb_target_boxes]的张量
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # 获取预测的logits

        # 获取源预测的排列索引
        idx = self._get_src_permutation_idx(indices)
        # 连接所有匹配的目标类别
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            # IA-BCE损失：基于IoU的自适应二元交叉熵损失
            alpha = self.focal_alpha
            gamma = 2 
            src_boxes = outputs['pred_boxes'][idx]  # 获取预测框
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # 获取目标框

            # 计算预测框和目标框之间的IoU
            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()  # 正样本IoU
            prob = src_logits.sigmoid()  # 预测概率
            
            # 初始化正样本权重和负样本权重
            pos_weights = torch.zeros_like(src_logits)
            neg_weights =  prob ** gamma

            # 构建正样本索引
            pos_ind=[id for id in idx]
            pos_ind.append(target_classes_o)

            # 计算自适应权重
            t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()

            pos_weights[pos_ind] = t.to(pos_weights.dtype)
            neg_weights[pos_ind] = 1 - t.to(neg_weights.dtype)
            
            # 使用融合的logsigmoid重新表述标准损失，提高数值稳定性
            loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
            loss_ce = loss_ce.sum() / num_boxes

        elif self.use_position_supervised_loss:
            # 位置监督损失：基于IoU的位置感知分类损失
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            # 计算IoU作为位置监督信号
            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            pos_ious_func = pos_ious  # IoU函数

            # 创建类别IoU目标张量
            cls_iou_func_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind=[id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_func_targets[pos_ind] = pos_ious_func
            
            # 归一化IoU目标
            norm_cls_iou_func_targets = cls_iou_func_targets \
                / (cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(1, True) + 1e-8)
            loss_ce = position_supervised_loss(src_logits, norm_cls_iou_func_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        elif self.use_varifocal_loss:
            # Varifocal损失：可变焦距损失
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            # 计算IoU作为质量指标
            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()

            # 创建类别IoU目标张量
            cls_iou_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind=[id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_targets[pos_ind] = pos_ious
            loss_ce = sigmoid_varifocal_loss(src_logits, cls_iou_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        else:
            # 标准Focal Loss
            # 创建目标类别张量，默认为无目标类别
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o  # 设置匹配位置的类别

            # 创建one-hot编码的目标类别
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:,:,:-1]  # 移除无目标类别
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        
        losses = {'loss_ce': loss_ce}

        if log:
            # 计算分类错误率（仅用于日志记录）
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """计算基数误差，即预测非空框数量的绝对误差
        这不是真正的损失，仅用于日志记录目的。不传播梯度
        
        Args:
            outputs: 模型输出字典，包含预测logits
            targets: 目标列表，每个元素包含标签信息
            indices: 匹配索引
            num_boxes: 目标框数量
            
        Returns:
            losses: 包含基数误差的损失字典
        """
        # 获取预测logits
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        
        # 计算每个样本的真实目标框数量
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        
        # 统计预测为非"无目标"类别的数量（排除最后一个类别，即背景类）
        # argmax(-1)获取每个预测的最大概率类别索引
        # 比较是否不等于最后一个类别（背景类）
        # sum(1)在batch维度上求和，得到每个样本的预测目标数量
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        
        # 计算预测数量与真实数量的L1损失
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """计算边界框相关的损失：L1回归损失和GIoU损失
        
        Args:
            outputs: 模型输出字典，必须包含'pred_boxes'
            targets: 目标列表，每个字典必须包含"boxes"键，格式为[nb_target_boxes, 4]
            indices: 匹配索引
            num_boxes: 目标框数量，用于归一化
            
        Returns:
            losses: 包含边界框损失的字典
        """
        # 确保输出中包含预测框
        assert 'pred_boxes' in outputs
        
        # 获取源预测框的排列索引
        idx = self._get_src_permutation_idx(indices)
        
        # 提取匹配的预测框和目标框
        src_boxes = outputs['pred_boxes'][idx]  # 预测的边界框
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # 真实边界框

        # 计算L1回归损失（预测框与目标框之间的绝对差值）
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes  # 归一化L1损失

        # 计算GIoU损失
        # 将中心点格式(cx,cy,w,h)转换为角点格式(x1,y1,x2,y2)
        src_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
        
        # 计算广义IoU，取对角线元素（对应匹配的框对）
        giou_scores = box_ops.generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)
        loss_giou = 1 - torch.diag(giou_scores)  # GIoU损失 = 1 - GIoU分数
        
        losses['loss_giou'] = loss_giou.sum() / num_boxes  # 归一化GIoU损失
        return losses

    def _get_src_permutation_idx(self, indices):
        """获取源预测框的排列索引
        
        Args:
            indices: 匹配索引列表，每个元素为(src_idx, tgt_idx)元组
            
        Returns:
            batch_idx: 批次索引
            src_idx: 源预测框索引
        """
        # 为每个匹配对创建批次索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # 连接所有源索引
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """获取目标框的排列索引
        
        Args:
            indices: 匹配索引列表，每个元素为(src_idx, tgt_idx)元组
            
        Returns:
            batch_idx: 批次索引
            tgt_idx: 目标框索引
        """
        # 为每个匹配对创建批次索引
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        # 连接所有目标索引
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """根据损失类型调用相应的损失计算函数
        
        Args:
            loss: 损失类型字符串
            outputs: 模型输出
            targets: 目标数据
            indices: 匹配索引
            num_boxes: 目标框数量
            **kwargs: 额外参数
            
        Returns:
            计算得到的损失字典
        """
        # 损失函数映射表
        loss_map = {
            'labels': self.loss_labels,      # 分类损失
            'cardinality': self.loss_cardinality,  # 基数损失
            'boxes': self.loss_boxes,        # 边界框损失
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """执行损失计算的主函数
        
        Args:
            outputs: 模型输出字典，包含预测结果
            targets: 目标列表，长度等于batch_size
            
        Returns:
            losses: 包含所有计算损失的字典
        """
        # 训练时使用分组DETR，推理时使用单组
        group_detr = self.group_detr if self.training else 1
        
        # 提取主输出（排除辅助输出）
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 获取最后一层输出与目标的匹配关系
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        # 计算所有节点上目标框的平均数量，用于归一化
        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr  # 如果不求和分组损失，则乘以组数
            
        # 转换为张量并同步到所有进程
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)  # 分布式训练时同步
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()  # 平均并确保最小值为1

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 处理辅助损失：对每个中间层的输出重复此过程
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # 为辅助输出重新计算匹配
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # 仅对最后一层启用日志记录
                        kwargs = {'log': False}
                    # 计算辅助损失并添加层索引后缀
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 处理编码器输出损失
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            # 为编码器输出计算匹配
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # 仅对最后一层启用日志记录
                    kwargs['log'] = False
                # 计算编码器损失并添加_enc后缀
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
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
    prob = inputs.sigmoid()
    focal_weight = targets * (targets > 0.0).float() + \
            (1 - alpha) * (prob - targets).abs().pow(gamma) * \
            (targets <= 0.0).float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * focal_weight

    return loss.mean(1).sum() / num_boxes


def position_supervised_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * (torch.abs(targets - prob) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcess(nn.Module):
    """ 
    后处理模块，将模型的原始输出转换为COCO API期望的格式
    主要功能：
    1. 对预测结果进行sigmoid激活
    2. 选择置信度最高的前num_select个预测框
    3. 将相对坐标转换为绝对坐标
    4. 返回标准化的预测结果字典
    """
    def __init__(self, num_select=300) -> None:
        """
        初始化后处理模块
        
        Args:
            num_select (int): 选择置信度最高的前num_select个预测框，默认300个
        """
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()  # 推理时不需要计算梯度，节省内存
    def forward(self, outputs, target_sizes):
        """ 
        执行后处理计算
        
        Args:
            outputs (dict): 模型的原始输出，包含：
                - pred_logits: 预测的类别logits [batch_size, num_queries, num_classes]
                - pred_boxes: 预测的边界框坐标 [batch_size, num_queries, 4] (cxcywh格式)
            target_sizes (torch.Tensor): 目标图像尺寸 [batch_size, 2]，包含每个batch中图像的原始尺寸
                - 用于评估时：必须是原始图像尺寸（数据增强前）
                - 用于可视化时：应该是数据增强后但padding前的图像尺寸
        
        Returns:
            list[dict]: 每个图像的处理结果列表，每个字典包含：
                - scores: 预测置信度分数
                - labels: 预测的类别标签
                - boxes: 预测的边界框坐标 (xyxy格式)
        """
        # 提取模型输出的预测logits和边界框
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # 对预测logits进行sigmoid激活，得到每个类别的置信度概率
        prob = out_logits.sigmoid()
        
        # 选择置信度最高的前num_select个预测
        # 将prob展平为[batch_size, num_queries * num_classes]，然后选择top-k
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values  # 置信度分数
        
        # 从topk_indexes中解码出对应的查询索引和类别标签
        # topk_boxes: 对应的查询索引 (query index)
        # labels: 对应的类别标签 (class label)
        topk_boxes = topk_indexes // out_logits.shape[2]  # 整除得到查询索引
        labels = topk_indexes % out_logits.shape[2]       # 取余得到类别标签
        
        # 将边界框从中心点格式(cxcywh)转换为左上右下格式(xyxy)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        
        # 根据topk_boxes索引选择对应的边界框
        # unsqueeze(-1).repeat(1,1,4)将索引扩展为[batch_size, num_select, 4]以匹配boxes的维度
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # 将相对坐标[0,1]转换为绝对坐标[0, height/width]
        # 从target_sizes中提取每个图像的高度和宽度
        img_h, img_w = target_sizes.unbind(1)
        # 创建缩放因子张量，用于将相对坐标转换为绝对坐标
        # 格式为[img_w, img_h, img_w, img_h]对应[x1, y1, x2, y2]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # 应用缩放因子，将相对坐标转换为绝对坐标
        boxes = boxes * scale_fct[:, None, :]

        # 将结果组织成COCO格式的字典列表
        # 每个字典包含该图像的所有预测结果
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args.num_classes + 1
    device = torch.device(args.device)


    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=args.pretrained_encoder,
        window_block_indexes=args.window_block_indexes,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=args.use_cls_token,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=args.shape if hasattr(args, 'shape') else (args.resolution, args.resolution) if hasattr(args, 'resolution') else (640, 640),
        rms_norm=args.rms_norm,
        backbone_lora=args.backbone_lora,
        force_no_pretrain=args.force_no_pretrain,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov2_weights=args.pretrain_weights is None,
    )
    if args.encoder_only:
        return backbone[0].encoder, None, None
    if args.backbone_only:
        return backbone, None, None

    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args)

    model = LWDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
    )
    return model

def build_criterion_and_postprocessors(args):
    device = torch.device(args.device)
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    # TODO this is a hack
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
    criterion = SetCriterion(args.num_classes + 1, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses, 
                             group_detr=args.group_detr, sum_group_losses=sum_group_losses,
                             use_varifocal_loss = args.use_varifocal_loss,
                             use_position_supervised_loss=args.use_position_supervised_loss,
                             ia_bce_loss=args.ia_bce_loss)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select)}

    return criterion, postprocessors
