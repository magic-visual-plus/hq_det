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
Modules to compute the matching cost and solve the corresponding LSAP.  
"""  
import numpy as np  
import torch  
from scipy.optimize import linear_sum_assignment  
from torch import nn  

from hq_det.models.rfdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou  

class HungarianMatcher(nn.Module):  
    """This class computes an assignment between the targets and the predictions of the network  
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,  
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,  
    while the others are un-matched (and thus treated as non-objects).  
    """  

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha: float = 0.25, use_pos_only: bool = False,  
                 use_position_modulated_cost: bool = False):  
        """Creates the matcher  
        Params:  
            cost_class: This is the relative weight of the classification error in the matching cost  
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost  
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost  
        """  
        super().__init__()  
        self.cost_class = cost_class  
        self.cost_bbox = cost_bbox  
        self.cost_giou = cost_giou  
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"  
        self.focal_alpha = focal_alpha  

    def _safe_log(self, x, eps=1e-8):  
        """安全的log计算，避免log(0)"""  
        return torch.log(torch.clamp(x, min=eps))  

    def _compute_classification_cost(self, out_prob, tgt_ids, alpha=0.25, gamma=2.0):  
        """计算分类成本，添加数值稳定性保护"""  
        # 限制概率范围，避免极值  
        out_prob = torch.clamp(out_prob, min=1e-8, max=1-1e-8)  
        
        # 计算focal loss成本  
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-self._safe_log(1 - out_prob))  
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-self._safe_log(out_prob))  
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]  
        
        # 数值稳定性检查  
        cost_class = torch.where(torch.isnan(cost_class), torch.zeros_like(cost_class), cost_class)  
        cost_class = torch.where(torch.isinf(cost_class), torch.sign(cost_class) * 100.0, cost_class)  
        cost_class = torch.clamp(cost_class, min=-100.0, max=100.0)  
        
        return cost_class  

    def _compute_giou_cost(self, out_bbox, tgt_bbox):  
        """计算GIoU成本，添加数值稳定性保护"""  
        # 检查bbox的有效性  
        if torch.isnan(out_bbox).any() or torch.isnan(tgt_bbox).any():  
            print("Warning: NaN detected in bbox coordinates")  
            out_bbox = torch.where(torch.isnan(out_bbox), torch.zeros_like(out_bbox), out_bbox)  
            tgt_bbox = torch.where(torch.isnan(tgt_bbox), torch.zeros_like(tgt_bbox), tgt_bbox)  
        
        # 计算GIoU  
        try:  
            giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))  
            cost_giou = -giou  
            
            # 数值稳定性检查  
            cost_giou = torch.where(torch.isnan(cost_giou), torch.zeros_like(cost_giou), cost_giou)  
            cost_giou = torch.where(torch.isinf(cost_giou), torch.sign(cost_giou) * 10.0, cost_giou)  
            cost_giou = torch.clamp(cost_giou, min=-10.0, max=10.0)  
            
        except Exception as e:  
            print(f"Warning: GIoU computation failed: {e}")  
            cost_giou = torch.zeros((out_bbox.shape[0], tgt_bbox.shape[0]), device=out_bbox.device)  
        
        return cost_giou  

    def _compute_bbox_cost(self, out_bbox, tgt_bbox):  
        """计算L1成本，添加数值稳定性保护"""  
        # 检查bbox的有效性  
        if torch.isnan(out_bbox).any() or torch.isnan(tgt_bbox).any():  
            print("Warning: NaN detected in bbox coordinates for L1 cost")  
            out_bbox = torch.where(torch.isnan(out_bbox), torch.zeros_like(out_bbox), out_bbox)  
            tgt_bbox = torch.where(torch.isnan(tgt_bbox), torch.zeros_like(tgt_bbox), tgt_bbox)  
        
        # 计算L1距离  
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  
        
        # 数值稳定性检查  
        cost_bbox = torch.where(torch.isnan(cost_bbox), torch.zeros_like(cost_bbox), cost_bbox)  
        cost_bbox = torch.where(torch.isinf(cost_bbox), torch.ones_like(cost_bbox) * 10.0, cost_bbox)  
        cost_bbox = torch.clamp(cost_bbox, min=0.0, max=10.0)  
        
        return cost_bbox  

    def _fix_cost_matrix(self, cost_matrix):  
        """修复成本矩阵中的无效值"""  
        # 检查是否有无效值  
        has_nan = torch.isnan(cost_matrix).any()  
        has_inf = torch.isinf(cost_matrix).any()  
        
        if has_nan or has_inf:  
            print(f"Warning: Cost matrix contains NaN: {has_nan}, Inf: {has_inf}")  
            print(f"  Matrix shape: {cost_matrix.shape}")  
            print(f"  Matrix range: [{cost_matrix[torch.isfinite(cost_matrix)].min().item():.6f}, {cost_matrix[torch.isfinite(cost_matrix)].max().item():.6f}]")  
            
            # 修复无效值  
            cost_matrix = torch.where(torch.isnan(cost_matrix), torch.zeros_like(cost_matrix), cost_matrix)  
            cost_matrix = torch.where(torch.isinf(cost_matrix), torch.sign(cost_matrix) * 1000.0, cost_matrix)  
            
            # 进一步裁剪极值  
            cost_matrix = torch.clamp(cost_matrix, min=-1000.0, max=1000.0)  
        
        return cost_matrix  

    @torch.no_grad()  
    def forward(self, outputs, targets, group_detr=1):  
        """ Performs the matching  
        Params:  
            outputs: This is a dict that contains at least these entries:  
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits  
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates  
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:  
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth  
                           objects in the target) containing the class labels  
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates  
            group_detr: Number of groups used for matching.  
        Returns:  
            A list of size batch_size, containing tuples of (index_i, index_j) where:  
                - index_i is the indices of the selected predictions (in order)  
                - index_j is the indices of the corresponding selected targets (in order)  
            For each batch element, it holds:  
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)  
        """  
        bs, num_queries = outputs["pred_logits"].shape[:2]  

        # We flatten to compute the cost matrices in a batch  
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]  
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]  

        # Also concat the target labels and boxes  
        tgt_ids = torch.cat([v["labels"] for v in targets])  
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  

        # 添加输入数据的基本检查  
        if torch.isnan(out_prob).any() or torch.isnan(out_bbox).any():  
            print("Warning: NaN detected in model outputs")  
            out_prob = torch.where(torch.isnan(out_prob), torch.zeros_like(out_prob), out_prob)  
            out_bbox = torch.where(torch.isnan(out_bbox), torch.zeros_like(out_bbox), out_bbox)  

        # Compute the giou cost betwen boxes (with stability protection)  
        cost_giou = self._compute_giou_cost(out_bbox, tgt_bbox)  

        # Compute the classification cost (with stability protection)  
        cost_class = self._compute_classification_cost(out_prob, tgt_ids, alpha=0.25, gamma=2.0)  

        # Compute the L1 cost between boxes (with stability protection)  
        cost_bbox = self._compute_bbox_cost(out_bbox, tgt_bbox)  

        # Final cost matrix  
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou  
        
        # 修复最终成本矩阵  
        C = self._fix_cost_matrix(C)  
        
        C = C.view(bs, num_queries, -1).cpu()  

        sizes = [len(v["boxes"]) for v in targets]  
        indices = []  
        g_num_queries = num_queries // group_detr  
        C_list = C.split(g_num_queries, dim=1)  
        
        for g_i in range(group_detr):  
            C_g = C_list[g_i]  
            
            # 替换原来的第105行，添加数值稳定性保护  
            indices_g = []  
            for i, c in enumerate(C_g.split(sizes, -1)):  
                cost_matrix = c[i]  
                
                # 转换为numpy并进行最后的安全检查  
                cost_np = cost_matrix.numpy()  
                
                # 确保矩阵中没有无效值  
                if not np.isfinite(cost_np).all():  
                    print(f"Warning: Non-finite values in cost matrix batch {i}, group {g_i}")  
                    cost_np = np.nan_to_num(cost_np, nan=0.0, posinf=1000.0, neginf=-1000.0)  
                
                # 应用匈牙利算法  
                try:  
                    indices_g.append(linear_sum_assignment(cost_np))  
                except Exception as e:  
                    print(f"Error in linear_sum_assignment: {e}")  
                    # 如果仍然失败，使用简单的贪心匹配作为备选  
                    num_pred, num_gt = cost_np.shape  
                    if num_pred > 0 and num_gt > 0:  
                        # 简单的贪心匹配  
                        pred_indices = []  
                        gt_indices = []  
                        used_gt = set()  
                        for p in range(min(num_pred, num_gt)):  
                            best_gt = -1  
                            best_cost = float('inf')  
                            for g in range(num_gt):  
                                if g not in used_gt and cost_np[p, g] < best_cost:  
                                    best_cost = cost_np[p, g]  
                                    best_gt = g  
                            if best_gt >= 0:  
                                pred_indices.append(p)  
                                gt_indices.append(best_gt)  
                                used_gt.add(best_gt)  
                        indices_g.append((np.array(pred_indices), np.array(gt_indices)))  
                    else:  
                        indices_g.append((np.array([]), np.array([])))  
            
            if g_i == 0:  
                indices = indices_g  
            else:  
                indices = [  
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))  
                    for indice1, indice2 in zip(indices, indices_g)  
                ]  
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]  


def build_matcher(args):  
    return HungarianMatcher(  
        cost_class=args.set_cost_class,  
        cost_bbox=args.set_cost_bbox,  
        cost_giou=args.set_cost_giou,  
        focal_alpha=args.focal_alpha,)