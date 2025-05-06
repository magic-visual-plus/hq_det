
import torch.nn
from ...common import PredictionResult
import numpy as np
import os
from .core import YAMLConfig

class HQRTDETR(torch.nn.Module):
    def __init__(self, class_id2names, **kwargs):
        super(HQRTDETR, self).__init__()
        self.id2names = class_id2names
        current_module_path = __file__
        config_path = os.path.join(
            os.path.dirname(current_module_path), 'configs', 'rtdetrv2', 'rtdetrv2_r50vd_m_7x_coco.yml')
        
        cfg = YAMLConfig(config_path)
        cfg.yaml_cfg['num_classes'] = len(class_id2names)
        cfg.yaml_cfg['remap_mscoco_category'] = False
        cfg.yaml_cfg['eval_spatial_size'] = [1024, 1024]
        self.model = cfg.model
        self.criterion = cfg.criterion
        self.postprocessor = cfg.postprocessor
        self.load_model(kwargs['model'])

    def get_class_names(self):
        # Get the class names from the model
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v
            pass
        return names

    def load_model(self, path):
        # Load the YOLO model using the specified path and device
        data = torch.load(path, map_location='cpu')
        state_dict = data['ema']['module']
        new_state_dict={k: v for k, v in state_dict.items() if state_dict[k].shape == self.model.state_dict()[k].shape}
        print(len(new_state_dict), len(state_dict))
        print([k for k in self.model.state_dict().keys() if k not in new_state_dict.keys()])
        self.model.load_state_dict(new_state_dict, strict=False)
        pass

    def extract_target(self, batch_data):
        # Extract the target data from the batch
        return batch_data['targets']
    
    def forward(self, batch_data):
        samples = batch_data['img']
        targets = self.extract_target(batch_data)
        forward_result = self.model(samples, targets)
        
        return forward_result
        pass
    
    def preprocess(self, batch_data):
        # Preprocess the input data for the YOLO model
        pass

    def postprocess(self, forward_result, batch_data, confidence=0.0):
        # Post-process the predictions
        if 'origin_size' not in batch_data:
            origin_size = torch.stack([torch.tensor([batch_data['img'].shape[3], batch_data['img'].shape[2]]) for _ in range(batch_data['img'].shape[0])], 0)
            origin_size = origin_size.to(batch_data['img'].device)
            pass
        else:
            origin_size = batch_data['origin_size']

        preds = self.postprocessor(forward_result, origin_size)

        results = []

        for pred in preds:
            record = PredictionResult()
            if len(pred['labels']) == 0:
                pass
            else:
                # add pred
                pred_bboxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                pred_cls = pred['labels'].cpu().numpy().astype(np.int32)
                record.bboxes = pred_bboxes
                record.cls = pred_cls
                record.scores = pred_scores
                record.image_id = batch_data['image_id']
                pass
            results.append(record)
            pass
        return results
        
    
    def predict(self, batch_data):
        pass

    def compute_loss(self, batch_data, forward_result):
        # Compute the loss using the YOLO model
        # This is a placeholder; actual implementation may vary
        
        if True:
        #if self.training:
            target = self.extract_target(batch_data)
            loss_dict = self.criterion(forward_result, target)

            loss = sum(loss_dict.values())
            info = {k: loss_dict['loss_' + k].item() for k in ['vfl', 'bbox', 'giou']}
        else:
            loss = torch.tensor(0.0, device=forward_result.device)
            info = {
                'vfl': 0.0,
                'bbox': 0.0,
                'giou': 0.0,
            }
        return loss, info

    def save(self, path):
        # Save the YOLO model to the specified path
        
        torch.save(
            {
                'ema': {
                    'module': self.model.state_dict()
                }
            }, path + '.pth')
        pass

    pass