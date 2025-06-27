import sys
from hq_det.models.dino import hq_dino
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
from hq_det import augment
import os
import torch
import torch.optim
from hq_det import torch_utils
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from hq_det.tools import train_dino



if __name__ == '__main__':
    train_dino.run(
        data_path=sys.argv[1],
        output_path='output',
        num_epoches = 50,
        lr0=1e-4,
        load_checkpoint=sys.argv[2],
        # eval_class_names=[
        #     '划伤', '划痕', '压痕', '吊紧', '异物外漏', '折痕', '抛线', '拼接间隙', '烫伤', '爆针线', '破损', ' 碰伤', '线头', '脏污', '褶皱(贯穿)', '褶皱（轻度）', '褶皱（重度）', '重跳针', '褶皱(贯穿)',
        #     '脏污（彩色）', '脏污（颜料笔）', '褶皱(T型)'
        # ],
        eval_class_names=[],
        batch_size=1,
        image_size=1536,
        gradient_update_interval=16,
        lr_backbone_mult=1.0,
    )
    pass