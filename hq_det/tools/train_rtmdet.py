import sys
from hq_det.models import rtmdet
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
import os
import torch
import torch.optim
from . import tools_mmdet


class RtmDetTrainer(HQTrainer):
    def __init__(self, args: HQTrainerArguments):
        super().__init__(args)
        pass

    def build_model(self):
        # Load the YOLO model using the specified path and device
        id2names = self.args.class_id2names
        model = rtmdet.HQRTMDET(class_id2names=id2names, **self.args.model_argument)
        return model
    
    def collate_fn(self, batch):
        return tools_mmdet.collate_fn(batch)
    


