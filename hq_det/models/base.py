from mmengine import MODELS
import torch.nn
from ..common import PredictionResult
import numpy as np
from typing import List
from ..common import HQTrainerArguments


class HQModel(torch.nn.Module):
    def __init__(self, class_id2names=None, **kwargs):
        super(HQModel, self).__init__()
        pass

    def get_class_names(self):
        pass

    def load_model(self, path):
        pass

    def forward(self, batch_data):
        pass
    
    def preprocess(self, batch_data):
        # Preprocess the input data for the YOLO model
        pass

    def postprocess(self, forward_result, batch_data, confidence=0.0):
        pass
        
    
    def predict(self, imgs) -> List[PredictionResult]:
        pass

    def compute_loss(self, batch_data, forward_result):
        pass

    def save(self, path):
        pass

    def get_param_dict(self, args: HQTrainerArguments):
        # Get the parameters of the model
        params_default = []
        params_backbone = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                params_backbone.append(param)
            else:
                params_default.append(param)

        return [
            {
                'params': params_default,
                'lr': args.lr0
            },
            {
                'params': params_backbone,
                'lr': args.lr0 * args.lr_backbone_mult,
            }
        ]
    
    def to(self, device):
        super(HQModel, self).to(device)
        self.device = device

    pass
