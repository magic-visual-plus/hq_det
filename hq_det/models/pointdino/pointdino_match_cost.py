# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class PointDINOL1Cost(BaseMatchCost):
    """L1 matching cost for 2D point prediction.

    Predicted points are normalized coordinates in [0, 1].
    Ground-truth points are pixel coordinates and will be normalized
    according to image width and height before computing the cost.
    """

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        pred_points = pred_instances.points
        gt_points = gt_instances.points
        img_h, img_w = img_meta['img_shape']
        factor = gt_points.new_tensor([img_w, img_h]).unsqueeze(0)
        gt_points = gt_points / factor
        point_cost = torch.cdist(pred_points, gt_points, p=1)
        return point_cost * self.weight
