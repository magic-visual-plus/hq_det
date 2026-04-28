import sys
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
from hq_det import augment
import os
import torch
import torch.optim
from hq_det import torch_utils
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from hq_det.tools import train_dino2 as train_dino


if __name__ == '__main__':
    train_dino.run(
        data_path=sys.argv[1],
        output_path='output',
        num_epoches = 100,
        lr0=1e-4,
        load_checkpoint=sys.argv[2],
        # eval_class_names=[
        #     '划伤', '划痕', '压痕', '吊紧', '异物外漏', '折痕', '抛线', '拼接间隙', '烫伤', '爆针线', '破损', ' 碰伤', '线头', '脏污', '褶皱(贯穿)', '褶皱（轻度）', '褶皱（重度）', '重跳针', '褶皱(贯穿)',
        #     '脏污（彩色）', '脏污（颜料笔）', '褶皱(T型)'
        # ],
        eval_class_names=[],
        batch_size=3,
        image_size=1536,
        gradient_update_interval=1,
        lr_backbone_mult=1.0,
        dino_config_name=None, # swinLarge: dino-5scale_swin-l_8xb2-12e_coco.py, default: dino_4scale_r50_8xb2_12e_coco.py
        # augment_split_size=2560,
        # augment_split_proba=0.5,
        # augment_foreground_path=sys.argv[3],
        # augment_foreground_proba=0.8,
        focal_loss_alpha=0.75,
        devices=list(range(torch.cuda.device_count()))
    )
