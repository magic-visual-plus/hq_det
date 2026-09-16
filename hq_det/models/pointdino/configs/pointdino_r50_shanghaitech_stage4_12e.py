"""PointDINO Stage 4 with native image inputs and full-map FIDT supervision."""

_base_ = '../../codetr/configs/_base_/default_runtime.py'

custom_imports = dict(
    imports=['hq_det.models.pointdino'], allow_failed_imports=False)

model = dict(
    type='PointDINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    use_dn=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='PointDINOHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        point_euclidean_weight=0.20,
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        point_noise_scale=0.01,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='PointDINOL1Cost', weight=20.0)
            ])),
    test_cfg=dict(max_per_img=900),
    point_fidt_head=dict(
        enabled=True,
        in_channels=256,
        hidden_channels=64,
        num_groups=8,
        upsample_factor=2,
        gamma=0.02,
        phi=0.75,
        xi=1.0,
        fidt_loss_weight=1.0,
        chunk_size=4096,
        point_chunk_size=256,
        # Preserve the source's detached raw-MSE diagnostic.
        debug=True))

data_root = 'data/shanghaitech_part_b/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='PointDINOLoadAnnotations', with_bbox=False, with_label=False,
        with_point=True),
    dict(type='PointDINORandomFlip', prob=0.5),
    dict(
        type='PointDINOPackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction'))
]

# No resizing: image pixels and GT points retain their original coordinates.
# PointDINOPackDetInputs supplies an identity scale_factor for prediction.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='PointDINOLoadAnnotations', with_bbox=False, with_label=False,
        with_point=True),
    dict(
        type='PointDINOPackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='BaseDetDataset',
        data_root=data_root,
        ann_file='train_point.json',
        data_prefix=dict(img_path='train_data/images/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseDetDataset',
        data_root=data_root,
        ann_file='test_point.json',
        data_prefix=dict(img_path='test_data/images/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='PointDINOMetric', distance_thresholds=[5.0, 10.0],
    score_threshold=0.5)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
        milestones=[11], gamma=0.1)
]
auto_scale_lr = dict(enable=False, base_batch_size=16)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook', interval=1, by_epoch=True, save_last=True,
        max_keep_ckpts=3, save_best='point/f1@10px', rule='greater'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
randomness = dict(seed=0, deterministic=False)

# Supply a converted 2-D Stage-2 PointDINO initialization checkpoint with
# --load-from to reproduce the source experiment's initialization.
load_from = None
resume = False
work_dir = './work_dirs/pointdino_r50_shanghaitech_stage4_12e'
