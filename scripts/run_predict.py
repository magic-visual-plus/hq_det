import sys
from hq_det.models import yolo
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
import os
import torch
from hq_det import torch_utils
from ultralytics.utils import DEFAULT_CFG


if __name__ == '__main__':
    model = yolo.HQYOLO(model=sys.argv[1])
    model.predict(sys.argv[2])
    pass