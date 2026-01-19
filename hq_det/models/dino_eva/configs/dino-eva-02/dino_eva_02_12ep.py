import copy
import torch.nn as nn
from functools import partial
import math

from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.backbone import EVA02_ViT, SimpleFeaturePyramid
from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

from ..common.gear_loader_lsj_1024 import dataloader

from hq_det.models.dino_eva.modeling import (
    DINO,
    DINOTransformerEncoder,
    DINOTransformerDecoder,
    DINOTransformer,
    DINOCriterion,
)
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
model = L(DINO)(
    backbone=L(SimpleFeaturePyramid)(
        net=L(EVA02_ViT)(
            img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=16,
        mlp_ratio=4*2/3,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        use_act_checkpoint = False,
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(2.0, 1.0, 0.5),  # (4.0, 2.0, 1.0, 0.5) in ViTDet
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "p3": ShapeSpec(channels=256),
            "p4": ShapeSpec(channels=256),
            "p5": ShapeSpec(channels=256),
            "p6": ShapeSpec(channels=256),
        },
        in_features=["p3", "p4", "p5", "p6"],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(DINOTransformer)(
        encoder=L(DINOTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels=4,
            use_checkpoint=False
        ),
        decoder=L(DINOTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
            use_checkpoint=False,
        ),
        num_feature_levels=4,
        two_stage_num_proposals="${..num_queries}",
    ),
    embed_dim=256,
    num_classes=80,
    num_queries=900,
    aux_loss=True,
    criterion=L(DINOCriterion)(
        num_classes="${..num_classes}",
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
    ),
    dn_number=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    vis_period=0,
    input_format="RGB",
    device="cuda",
)

# set aux loss weight dict
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict
