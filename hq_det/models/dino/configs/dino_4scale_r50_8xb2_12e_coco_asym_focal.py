from mmengine.config import read_base

with read_base():
    from .dino_4scale_r50_8xb2_12e_coco import *


model['bbox_head']['loss_cls'] = dict(
    type="AsymmetricFocalLoss",
    use_sigmoid=True,
    gamma_pos=2.0,
    gamma_neg=2.0,
    alpha=0.25,
    loss_weight=1.0,
)
