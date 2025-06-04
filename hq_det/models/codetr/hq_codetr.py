from mmengine import MODELS, Config
import torch.nn
from ...common import PredictionResult
from ..base import HQModel
import numpy as np
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
import cv2
from typing import List
from ... import torch_utils
import os


class HQCoDetr(HQModel):
    def __init__(self, class_id2names=None, **kwargs):
        super(HQCoDetr, self).__init__(class_id2names, **kwargs)
        if class_id2names is None:
            data = torch.load(kwargs['model'], map_location='cpu')
            class_names = data['meta']['dataset_meta']['CLASSES']
            self.id2names = {i: name for i, name in enumerate(class_names)}
        else:
            self.id2names = class_id2names
            pass
        self.model = self.build_model(kwargs['model'])
        print(self.model)
        self.load_model(kwargs['model'])
        self.device = 'cpu'

    def get_class_names(self):
        # Get the class names from the model
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v

        return names
    
    def load_model(self, path):
        data = torch.load(path, map_location='cpu')
        new_state_dict = {}
        for k, v in data.items():
            if k in self.model.state_dict() and data[k].shape == self.model.state_dict()[k].shape:
                new_state_dict[k] = v
        print(f"Loaded {len(new_state_dict)}/{len(data)} parameters")
        self.model.load_state_dict(new_state_dict, strict=False)
                

    def build_model(self, path):
        """
        创建模型
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        codetr_config_path = os.path.join(current_dir, 'configs', 'codino', 'co_dino_5scale_swin_l_16xb1_16e_o365tococo.py')
        os.environ['NUM_CLASSES'] = str(len(self.id2names))
        os.environ['PRETRAINED'] = path
        codetr_config = Config.fromfile(codetr_config_path)
        
        return MODELS.build(codetr_config.model)
        

    def forward(self, batch_data):
        print(batch_data.keys())
        batch_data.update(self.model.data_preprocessor(batch_data, self.training))
        inputs = batch_data['inputs']
        batch_data_samples = batch_data['data_samples']
        
        img_feats = self.model.extract_feat(inputs)
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
            self.model.query_head.dn_generator(batch_data_samples)

        outs = self.model.query_head(img_feats, batch_img_metas, dn_label_query, dn_bbox_query, attn_mask)


        return {
            'outs': outs,
            'batch_gt_instances': batch_gt_instances,
            'batch_img_metas': batch_img_metas,
            'dn_meta': dn_meta,
            'img_feats': img_feats,
        }
        
    
    def preprocess(self, batch_data):
        # Preprocess the input data for the YOLO model
        pass

    def postprocess(self, forward_result, batch_data, confidence=0.0):
        # Post-process the predictions
        head_inputs_dict = forward_result['head_inputs_dict']

        preds = self.model.bbox_head.predict(
            head_inputs_dict['hidden_states'],
            head_inputs_dict['references'],
            batch_data_samples = batch_data['data_samples'],
            rescale=False)

        results = []
        for pred in preds:
            record = PredictionResult()
            mask = pred.scores > confidence
            if not mask.any():
                # no pred
                record.bboxes = np.zeros((0, 4), dtype=np.float32)
                record.scores = np.zeros((0,), dtype=np.float32)
                record.cls = np.zeros((0,), dtype=np.int32)
                pass
            else:
                # add pred
                pred_bboxes = pred.bboxes[mask].cpu().numpy()
                pred_scores = pred.scores[mask].cpu().numpy()
                pred_cls = pred.labels[mask].cpu().numpy().astype(np.int32)
                record.bboxes = pred_bboxes
                record.scores = pred_scores
                record.cls = pred_cls
                pass
            results.append(record)
        return results
        

    def imgs_to_batch(self, imgs):
        # Convert a list of images to a batch

        batch_data = {
            "inputs": [],
            "data_samples": [],
        }

        for img in imgs:
            img = torch.permute(torch.from_numpy(img), (2, 0, 1)).contiguous()
            batch_data['inputs'].append(img)
            data_sample = DetDataSample(metainfo={
                'img_shape': (img.shape[1], img.shape[2]),
            })

            gt_instance = InstanceData()
            data_sample.gt_instances = gt_instance

            batch_data['data_samples'].append(data_sample)
            pass

        return batch_data
        pass

    def predict(self, imgs, bgr=False, confidence=0.0, max_size=-1) -> List[PredictionResult]:
        if not bgr:
            # Convert RGB to BGR
            for i in range(len(imgs)):
                imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
                pass
            pass

        img_scales = np.ones((len(imgs),))
        if max_size > 0:
            for i in range(len(imgs)):
                max_hw = max(imgs[i].shape[0], imgs[i].shape[1])
                if max_hw > max_size:
                    rate = max_size / max_hw
                    imgs[i] = cv2.resize(imgs[i], (int(imgs[i].shape[1] * rate), int(imgs[i].shape[0] * rate)))
                    img_scales[i] = rate
                    pass
                pass
            pass
        device = self.device
        with torch.no_grad():
            batch_data = self.imgs_to_batch(imgs)
            batch_data = torch_utils.batch_to_device(batch_data, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                forward_result = self.forward(batch_data)
                preds = self.postprocess(forward_result, batch_data, confidence)
                pass
            pass
        
        for i in range(len(preds)):
            pred = preds[i]
            pred.bboxes = pred.bboxes / img_scales[i]
            # pred.bboxes[:, 0] = np.clip(pred.bboxes[:, 0], 0, imgs[i].shape[1])
            # pred.bboxes[:, 1] = np.clip(pred.bboxes[:, 1], 0, imgs[i].shape[0])
            # pred.bboxes[:, 2] = np.clip(pred.bboxes[:, 2], 0, imgs[i].shape[1])
            # pred.bboxes[:, 3] = np.clip(pred.bboxes[:, 3], 0, imgs[i].shape[0])
            pass

        return preds

    def compute_loss(self, batch_data, forward_result):

        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k, v in losses.items():
                new_k = '{}{}'.format(k, idx)
                if isinstance(v, list) or isinstance(v, tuple):
                    new_losses[new_k] = [i * weight for i in v]
                else:
                    new_losses[new_k] = v * weight
            return new_losses

        # Compute the loss using the YOLO model
        # This is a placeholder; actual implementation may vary
        data_samples = batch_data['data_samples']
        if self.model.training:

            x = forward_result['img_feats'] 
            losses = dict()
            if self.model.with_query_head:
                loss_inputs = forward_result['outs'][:-1] + \
                      (forward_result['batch_gt_instances'], forward_result['batch_img_metas'], forward_result['dn_meta'])
                bbox_losses = self.model.query_head.loss_by_feat(*loss_inputs)
                x = forward_result['outs'][-1]
                losses.update(bbox_losses)
            if self.model.with_rpn:
                proposal_cfg = self.model.train_cfg[self.model.head_idx].get(
                'rpn_proposal', self.model.test_cfg[self.model.head_idx].rpn)

                rpn_data_samples = copy.deepcopy(data_samples)
                # set cat_id of gt_labels to 0 in RPN
                for data_sample in rpn_data_samples:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

                rpn_losses, proposal_list = self.model.rpn_head.loss_and_predict(
                    x, rpn_data_samples, proposal_cfg=proposal_cfg)

                # avoid get same name with roi_head loss
                keys = rpn_losses.keys()
                for key in list(keys):
                    if 'loss' in key and 'rpn' not in key:
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)

                losses.update(rpn_losses)
            else:
                assert data_samples[0].get('proposals', None) is not None
                # use pre-defined proposals in InstanceData for the second stage
                # to extract ROI features.
                proposal_list = [
                    data_sample.proposals for data_sample in data_samples
                ]
            positive_coords = []
            for i in range(len(self.roi_head)):
                roi_losses = self.roi_head[i].loss(x, proposal_list,
                                                data_sample)
                if self.with_pos_coord:
                    positive_coords.append(roi_losses.pop('pos_coords'))
                else:
                    if 'pos_coords' in roi_losses.keys():
                        roi_losses.pop('pos_coords')
                roi_losses = upd_loss(roi_losses, idx=i)
                losses.update(roi_losses)

            for i in range(len(self.model.bbox_head)):
                bbox_losses = self.model.bbox_head[i].loss(x, data_samples)
                if self.with_pos_coord:
                    pos_coords = bbox_losses.pop('pos_coords')
                    positive_coords.append(pos_coords)
                else:
                    if 'pos_coords' in bbox_losses.keys():
                        bbox_losses.pop('pos_coords')
                bbox_losses = upd_loss(bbox_losses, idx=i + len(self.roi_head))
                losses.update(bbox_losses)

            if self.with_pos_coord and len(positive_coords) > 0:
                for i in range(len(positive_coords)):
                    bbox_losses = self.model.query_head.loss_aux(x, positive_coords[i],
                                                        i, data_samples)
                    bbox_losses = upd_loss(bbox_losses, idx=i)
                    losses.update(bbox_losses)

            loss, info = self.model.parse_losses(losses)

            info = {
                'loss': loss.item(),
                'cls': info['loss_cls'].item(),
                'box': info['loss_bbox'].item(),
                'giou': info['loss_iou'].item(),
            }
        else:
            loss = torch.tensor(0.0, device=forward_result['img_feats'].device)
            info = {
                'loss': 0.0,
                'cls': 0.0,
                'box': 0.0,
                'giou': 0.0,
            }
            pass
        return loss, info

    def save(self, path):
        # Save the YOLO model to the specified path
        
        torch.save(
            {
                'state_dict': self.model.state_dict(),
                'meta': {
                    'dataset_meta': {
                        'CLASSES': self.get_class_names(),
                    },
                }
            }, path + '.pth')
        pass

    def to(self, device):
        super(HQCoDetr, self).to(device)
        self.model.to(device)
        self.device = torch.device(device)
    pass