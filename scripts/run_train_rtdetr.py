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
    devices = list(range(int(os.getenv("GPU_NUM", "1"))))
    train_rtdetr.run(
        data_path=sys.argv[1],
        output_path='output',
        num_epoches=int(os.environ.get('NUM_EPOCHES', '180')),
        batch_size=int(os.environ.get('BATCH_SIZE', '6')),
        lr0=1e-4,
        load_checkpoint=sys.argv[2],
        eval_class_names=[],
        devices=devices
    )
    pass
