from mmengine import MODELS, Config
import torch.nn
import torch.nn.functional as F
from detrex.utils import inverse_sigmoid
from detectron2.config import LazyConfig, instantiate
from detectron2.modeling import detector_postprocess
from ...common import PredictionResult
from ..base import HQModel
import numpy as np
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
import cv2
from typing import List
from ... import torch_utils
import os


class HQDINO_EVA(HQModel):
    def __init__(self, class_id2names=None, dino_eva_config=None, **kwargs):
        super(HQDINO_EVA, self).__init__(class_id2names, **kwargs)
        if class_id2names is None:
            data = torch.load(kwargs['model'], map_location='cpu')
            class_names = data['meta']['dataset_meta']['CLASSES']
            self.id2names = {i: name for i, name in enumerate(class_names)}
        else:
            self.id2names = class_id2names

        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # dino_eva_config_path = os.path.join(current_dir, 'configs', 'dino-eva-02', 'dino_eva_02_12ep.py')
        # dino_eva_config = Config.fromfile(dino_eva_config_path)
        # dino_eva_config = LazyConfig.load(dino_eva_config_path)
        # print(LazyConfig.to_py(dino_eva_config)) #查看配置结构
        # print(dino_eva_config)
        dino_eva_config.model.num_classes = len(self.id2names)
        # dino_eva_config.model['roi_heads']['box_predictor']['num_classes'] = len(self.id2names)
        # dino_eva_config.model['roi_heads']['mask_head']['num_classes'] = len(self.id2names)
        # dino_eva_config.model['roi_heads']['num_classes'] = len(self.id2names)
        # self.model = MODELS.build(dino_eva_config.model)
        self.model = instantiate(dino_eva_config.model)
        self.load_model(kwargs['model'])
        self.device = torch.device('cpu')

    def get_class_names(self):
        # Get the class names from the model
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v

        return names

    def load_model(self, path):
        # Load the YOLO model using the specified path and device
        data = torch.load(path, map_location='cpu')
        # print("权重文件的所有键：", list(data['model'].keys()))
        state_dict = data['model']
        new_state_dict={k: v for k, v in state_dict.items() if state_dict[k].shape == self.model.state_dict()[k].shape}
        print(len(new_state_dict), len(state_dict))
        print([k for k in self.model.state_dict().keys() if k not in new_state_dict.keys()])
        self.model.load_state_dict(new_state_dict, strict=False)


    def forward(self, batch_data):
        # batch_data.update(self.model.data_preprocessor(batch_data, self.training))
        # inputs = batch_data['inputs']
        # data_samples = batch_data['data_samples']
        # img_feats = self.model.extract_feat(inputs)
        # head_inputs_dict = self.model.forward_transformer(
        #     img_feats, data_samples)
        images, img_size = self.model.preprocess_image(batch_data)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batch_data["data_samples"][img_id].img_shape
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = img_size[img_id][0], img_size[img_id][1]
                img_masks[img_id, :img_h, :img_w] = 0
        
        features = self.model.backbone(images.tensor)  # output feature dict
        
        multi_level_feats = self.model.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.model.position_embedding(multi_level_masks[-1]))
        
        if self.training:
            # gt_instances = [x["instances"].to(self.device) for x in batch_data]
            targets = self.model.prepare_targets(batch_data, batch_size, img_size)
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.model.prepare_for_cdn(
                targets,
                dn_number=self.model.dn_number,
                label_noise_ratio=self.model.label_noise_ratio,
                box_noise_scale=self.model.box_noise_scale,
                num_queries=self.model.num_queries,
                num_classes=self.model.num_classes,
                hidden_dim=self.model.embed_dim,
                label_enc=self.model.label_enc,
            )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
        ) = self.model.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
        )
        # hack implementation for distributed training
        inter_states[0] += self.model.label_enc.weight[0, 0] * 0.0

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.model.class_embed[lvl](inter_states[lvl])
            tmp = self.model.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # denoising postprocessing
        if dn_meta is not None:
            outputs_class, outputs_coord = self.model.dn_post_process(
                outputs_class, outputs_coord, dn_meta
            )

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.model.aux_loss:
            output["aux_outputs"] = self.model._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        interm_coord = enc_reference
        interm_class = self.model.transformer.decoder.class_embed[-1](enc_state)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}
        

        if self.training:
            return output, targets, dn_meta
        else:
            output['images_sizes'] = images.image_sizes
            return output
        pass    
    
    def preprocess(self, batch_data):
        # Preprocess the input data for the YOLO model
        pass

    def postprocess(self, inference_results, batch_data, confidence=0.0):
        # Post-process the predictions
        # head_inputs_dict = forward_result['head_inputs_dict']

        # preds = self.model.bbox_head.predict(
        #     head_inputs_dict['hidden_states'],
        #     head_inputs_dict['references'],
        #     batch_data_samples = batch_data['data_samples'],
        #     rescale=False)

        results = []
        for pred in inference_results:
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
                pred_bboxes = pred.pred_boxes[mask].tensor.cpu().numpy()
                pred_scores = pred.scores[mask].cpu().numpy()
                pred_cls = pred.pred_classes[mask].cpu().numpy().astype(np.int32)
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

    def compute_loss(self, batch_data, forward_result, targets=None, dn_meta=None):
        # Compute the loss using the YOLO model
        # This is a placeholder; actual implementation may vary
        if self.training:
            loss_dict = self.model.criterion(forward_result, targets, dn_meta)
            weight_dict = self.model.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            total_loss = sum(loss for loss in loss_dict.values())

            info = {
                'loss': total_loss.item(),
                'cls': loss_dict['loss_class'].item(),
                'box': loss_dict['loss_bbox'].item(),
                'giou': loss_dict['loss_giou'].item(),
            }
            return total_loss, info
        else:
            loss = torch.tensor(0.0, device=self.model.device)
            info = {
                'loss': 0.0,
                'cls': 0.0,
                'box': 0.0,
                'giou': 0.0,
            }
            box_cls = forward_result["pred_logits"]
            box_pred = forward_result["pred_boxes"]
            image_sizes = forward_result["images_sizes"]
            results = self.model.inference(box_cls, box_pred, image_sizes)
            # processed_results = []
            # for results_per_image, input_per_image, image_size in zip(
            #     results, forward_result, image_sizes
            # ):
            #     height = input_per_image.get("height", image_size[0])
            #     width = input_per_image.get("width", image_size[1])
            #     r = detector_postprocess(results_per_image, height, width)
            #     processed_results.append({"instances": r})
            return loss, info, results


        # head_inputs_dict = forward_result['head_inputs_dict']
        # data_samples = batch_data['data_samples']
        # if self.model.training:

        #     losses = self.model.bbox_head.loss(
        #         **head_inputs_dict, batch_data_samples=data_samples)
            
        #     loss, info = self.model.parse_losses(losses)

        #     info = {
        #         'loss': loss.item(),
        #         'cls': info['loss_cls'].item(),
        #         'box': info['loss_bbox'].item(),
        #         'giou': info['loss_iou'].item(),
        #     }
        # else:
        #     loss = torch.tensor(0.0, device=head_inputs_dict['hidden_states'].device)
        #     info = {
        #         'loss': 0.0,
        #         'cls': 0.0,
        #         'box': 0.0,
        #         'giou': 0.0,
        #     }
        #     pass
        # return loss, info

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
            }, path)
        pass

    def to(self, device):
        super(HQDINO_EVA, self).to(device)
        self.model.to(device)
        self.device = torch.device(device)
    pass