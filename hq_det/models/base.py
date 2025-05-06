from mmengine import MODELS
import torch.nn
from ..common import PredictionResult
import numpy as np
from typing import List


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

    pass