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
from mmdet.models import (
    DINO,
    DetDataPreprocessor,
    SwinTransformer,
    ChannelMapper,
    DINOHead,
    ResNet,
)
from mmdet.models.task_modules import (
    HungarianAssigner,
    FocalLossCost,
    BBoxL1Cost,
    IoUCost,
)
import copy


def get_gdino_config(image_size=224, num_classes=80):
    return dict(
        type=DINO,
        num_queries=900,
        with_box_refine=True,
        as_two_stage=True,
        data_preprocessor=dict(
            type=DetDataPreprocessor,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_mask=False,
        ),
        backbone=dict(
            type=SwinTransformer,
            pretrain_img_size=image_size,
            patch_size=image_size // 56,
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            strides=(image_size // 56, 2, 2, 2),
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(1, 2, 3),
            with_cp=False,
            convert_weights=False),
        neck=dict(
            type=ChannelMapper,
            in_channels=[192, 384, 768],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            bias=True,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        encoder=dict(
            num_layers=6,
            # visual layer config
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        ),
        decoder=dict(
            num_layers=6,
            return_intermediate=True,
            layer_cfg=dict(
                # query self attention layer
                self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                # cross attention layer query to image
                cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
            post_norm_cfg=None),
        positional_encoding=dict(
            num_feats=128, normalize=True, offset=0.0, temperature=20),
        bbox_head=dict(
            type=DINOHead,
            num_classes=num_classes,
            sync_cls_avg_factor=True,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),  # 2.0 in DeformDETR
            loss_bbox=dict(type='L1Loss', loss_weight=5.0)),
        dn_cfg=dict(  # TODO: Move to model.train_cfg ?
            label_noise_scale=0.5,
            box_noise_scale=1.0,  # 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None,
                        num_dn_queries=100)),  # TODO: half num_dn_queries
        # training and testing settings
        train_cfg=dict(
            assigner=dict(
                type=HungarianAssigner,
                match_costs=[
                    dict(type=FocalLossCost, weight=2.0),
                    dict(type=BBoxL1Cost, weight=5.0, box_format='xywh'),
                    dict(type=IoUCost, iou_mode='giou', weight=2.0)
                ])),
        test_cfg=dict(max_per_img=300))

def get_gdino_config2(image_size=224, num_classes=80):
    return dict(
    type=DINO,
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type=ChannelMapper,
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type=DINOHead,
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=FocalLossCost, weight=2.0),
                dict(type=BBoxL1Cost, weight=5.0, box_format='xywh'),
                dict(type=IoUCost, iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

class HQGDINO(HQModel):
    def __init__(self, class_id2names=None, image_size=1120, **kwargs):
        super(HQGDINO, self).__init__(class_id2names, **kwargs)
        if class_id2names is None:
            data = torch.load(kwargs['model'], map_location='cpu')
            class_names = data['meta']['dataset_meta']['CLASSES']
            self.id2names = {i: name for i, name in enumerate(class_names)}
            self.image_size = data.get('image_size', image_size)
        else:
            self.id2names = class_id2names
            self.image_size = image_size
            pass

        if image_size % 224 != 0:
            raise ValueError('image_size must be multiple of 224')
        
        num_classes = max(self.id2names.keys()) + 1
        self.model_config = get_gdino_config2(num_classes=num_classes, image_size=image_size) 
        self.model = MODELS.build(copy.deepcopy(self.model_config))
        self.load_model(kwargs['model'])
        self.device = torch.device('cpu')

    def get_class_names(self):
        # Get the class names from the model
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v

        return names

    def load_model(self, path):
        data = torch.load(path, map_location='cpu')
        new_state_dict={k: v for k, v in data['state_dict'].items() if data['state_dict'][k].shape == self.model.state_dict()[k].shape}
        print(len(new_state_dict), len(data['state_dict']))
        self.model.load_state_dict(new_state_dict, strict=False)


    def forward(self, batch_data):
        batch_data.update(self.model.data_preprocessor(batch_data, self.training))
        inputs = batch_data['inputs']
        data_samples = batch_data['data_samples']
        img_feats = self.model.extract_feat(inputs)
        head_inputs_dict = self.model.forward_transformer(
            img_feats, data_samples)
        
        return {
            'img_feats': img_feats,
            'head_inputs_dict': head_inputs_dict,
        }
        pass
    
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
            img = torch_utils.pad_image_array(img, None, (self.image_size, self.image_size), pad_value=114)
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
        if max_size == -1:
            max_size = self.image_size
            pass

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
            pass

        return preds

    def compute_loss(self, batch_data, forward_result):
        # Compute the loss using the YOLO model
        # This is a placeholder; actual implementation may vary
        head_inputs_dict = forward_result['head_inputs_dict']
        data_samples = batch_data['data_samples']
        if self.model.training:

            losses = self.model.bbox_head.loss(
                **head_inputs_dict, batch_data_samples=data_samples)
            
            loss, info = self.model.parse_losses(losses)

            info = {
                'loss': loss.item(),
                'cls': info['loss_cls'].item(),
                'box': info['loss_bbox'].item(),
                'giou': info['loss_iou'].item(),
            }
        else:
            loss = torch.tensor(0.0, device=head_inputs_dict['hidden_states'].device)
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
                },
                'image_size': self.image_size,
            }, path)
        pass

    def to(self, device):
        super(HQGDINO, self).to(device)
        self.model.to(device)
        self.device = torch.device(device)
    pass