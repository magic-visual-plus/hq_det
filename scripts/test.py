import sys
from hq_det.models import dino
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
import os
import torch
from hq_det import torch_utils
from ultralytics.utils import DEFAULT_CFG
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    model = dino.HQDINO(model=sys.argv[1])
    model = dino.HQDINO(model=sys.argv[1])