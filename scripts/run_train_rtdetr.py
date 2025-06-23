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
from hq_det.tools import train_rtdetr



if __name__ == '__main__':
    train_rtdetr.run(
        data_path=sys.argv[1],
        output_path='output',
        num_epoches = 180,
        lr0=1e-4,
        load_checkpoint=sys.argv[2],
        eval_class_names=[]
    )
    pass