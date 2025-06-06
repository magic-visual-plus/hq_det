_base_ = '../_base_/default_runtime.py'

"""
本配置文件修改自 CO-DETR 的 ssj_270k_coco-instance.py
主要修改内容:
1. 修改了数据集路径, 通过环境变量设置
2. 修改了图像大小, 通过环境变量设置

所有修改过的模块均使用 HACK 标记, 以方便后续修改
"""
# HACK: -------------------------------
# HACK: 添加 os 模块, 通过环境变量设置数据集路径
import os
# dataset settings
dataset_type = 'CocoDataset'
data_root = os.environ.get('DATA_ROOT', 'data/coco/')   # HACK: 修改原有配置, 通过环境变量设置数据集路径

image_size = eval(os.environ.get('IMAGE_SIZE', '(1024, 1024)'))   # HACK: 修改原有配置, 通过环境变量设置图像大小

if isinstance(image_size, int):
    image_size = (image_size, image_size)
elif isinstance(image_size, tuple):
    assert len(image_size) == 2, f'Invalid image size: {image_size}'
    assert image_size[0] > 0 and image_size[1] > 0, f'Invalid image size: {image_size}'
else:
    raise ValueError(f'Invalid image size: {image_size}')
# -------------------------------------

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

# Standard Scale Jittering (SSJ) resizes and crops an image
# with a resize range of 0.8 to 1.25 of the original image size.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.8, 1.25),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# The model is trained by 270k iterations with batch_size 64,
# which is roughly equivalent to 144 epochs.

max_iters = 270000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer assumes bs=64
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.00004))

# learning rate policy
# lr steps at [0.9, 0.95, 0.975] of the maximum iterations
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=270000,
        by_epoch=False,
        milestones=[243000, 256500, 263250],
        gamma=0.1)
]

default_hooks = dict(checkpoint=dict(by_epoch=False, interval=10000))
log_processor = dict(by_epoch=False)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (32 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
