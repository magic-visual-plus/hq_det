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
import copy
from loguru import logger

class HQCoDetr(HQModel):
    """Co-DETR目标检测模型的实现类
    
    该类继承自HQModel,实现了基于Co-DETR的目标检测功能。主要包含模型构建、前向推理、损失计算等功能。
    
    Args:
        class_id2names (dict, optional): 类别ID到类别名称的映射字典
        **kwargs: 其他参数,包括模型路径等
    """
    def __init__(self, class_id2names=None, **kwargs):
        super(HQCoDetr, self).__init__(class_id2names, **kwargs)
        # 如果没有提供类别映射,则从预训练模型中加载
        if class_id2names is None:
            data = torch.load(kwargs['model'], map_location='cpu')
            class_names = data['meta']['dataset_meta']['CLASSES']
            self.id2names = {i: name for i, name in enumerate(class_names)}
        else:
            self.id2names = class_id2names
        # 构建并加载模型
        self.model = self.build_model(kwargs['model'])
        print(self.model)
        self.load_model(kwargs['model'])
        self.device = 'cpu'

    def get_class_names(self):
        """获取所有类别名称列表
        
        Returns:
            list: 类别名称列表
        """
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v
        return names
    
    def load_model(self, path):
        """加载预训练模型权重
        
        Args:
            path (str): 模型权重文件路径
        """
        data = torch.load(path, map_location='cpu')
        new_state_dict = {}
        # 只加载形状匹配的参数
        matched_params = 0
        total_params = len(data)
        for k, v in data.items():
            if k in self.model.state_dict() and data[k].shape == self.model.state_dict()[k].shape:
                new_state_dict[k] = v
                matched_params += 1
        
        match_ratio = matched_params / total_params
        if match_ratio < 0.8:
            logger.warning(f"Parameter match ratio ({match_ratio:.2%}) is below 80% threshold. Only {matched_params}/{total_params} parameters were loaded.")
        
        logger.info(f"Loaded {matched_params}/{total_params} parameters")
        self.model.load_state_dict(new_state_dict, strict=False)


    def build_model(self, path):
        """构建Co-DETR模型
        
        Args:
            path (str): 预训练模型路径
            
        Returns:
            nn.Module: 构建好的模型
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        codetr_config_path = os.path.join(current_dir, 'configs', 'codino', 'co_dino_5scale_r50_8xb2_1x_coco.py')
        # codetr_config_path = os.path.join(current_dir, 'configs', 'codino', 'co_dino_5scale_swin_l_16xb1_16e_o365tococo.py')
        os.environ['NUM_CLASSES'] = str(len(self.id2names))
        os.environ['PRETRAINED'] = path
        codetr_config = Config.fromfile(codetr_config_path)
        return MODELS.build(codetr_config.model)
        
    def forward(self, batch_data):
        """模型前向传播
        
        Args:
            batch_data (dict): 输入数据字典
            
        Returns:
            dict: 包含模型输出的字典
        """
        batch_data.update(self.model.data_preprocessor(batch_data, self.training))
        inputs = batch_data['inputs']
        batch_data_samples = batch_data['data_samples']
        
        # 提取图像特征
        img_feats = self.model.extract_feat(inputs)
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        # 生成查询向量
        dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
            self.model.query_head.dn_generator(batch_data_samples)

        # 通过查询头处理特征
        outs = self.model.query_head(img_feats, batch_img_metas, dn_label_query, dn_bbox_query, attn_mask)

        return {
            'outs': outs,
            'batch_gt_instances': batch_gt_instances,
            'batch_img_metas': batch_img_metas,
            'dn_meta': dn_meta,
            'img_feats': img_feats,
        }
        
    def preprocess(self, batch_data):
        """数据预处理
        
        Args:
            batch_data (dict): 输入数据字典
        """
        pass

    def postprocess(self, forward_result, batch_data, confidence=0.0):
        """后处理预测结果
        
        Args:
            forward_result (dict): 前向传播结果
            batch_data (dict): 输入数据
            confidence (float): 置信度阈值
            
        Returns:
            list: 预测结果列表
        """
        assert self.model.eval_module in ['detr', 'one-stage', 'two-stage']
        batch_data_samples = batch_data['data_samples']
        if self.model.use_lsj:
            for data_samples in batch_data_samples:
                img_metas = data_samples.metainfo
                input_img_h, input_img_w = img_metas['batch_input_shape']
                img_metas['img_shape'] = [input_img_h, input_img_w]

        img_feats = forward_result['img_feats']
        if self.model.with_bbox and self.model.eval_module == 'one-stage':
            results_list = self.model.predict_bbox_head(
                img_feats, batch_data_samples, rescale=False)
        elif self.model.with_roi_head and self.model.eval_module == 'two-stage':
            results_list = self.model.predict_roi_head(
                img_feats, batch_data_samples, rescale=False)
        else:
            results_list = self.model.predict_query_head(
                img_feats, batch_data_samples, rescale=False)

        preds = results_list
        
        results = []
        for pred in preds:
            record = PredictionResult()
            mask = pred.scores > confidence
            if not mask.any():
                # 无预测结果
                record.bboxes = np.zeros((0, 4), dtype=np.float32)
                record.scores = np.zeros((0,), dtype=np.float32)
                record.cls = np.zeros((0,), dtype=np.int32)
            else:
                # 添加预测结果
                pred_bboxes = pred.bboxes[mask].cpu().numpy()
                pred_scores = pred.scores[mask].cpu().numpy()
                pred_cls = pred.labels[mask].cpu().numpy().astype(np.int32)
                record.bboxes = pred_bboxes
                record.scores = pred_scores
                record.cls = pred_cls
            results.append(record)
        return results
        
    def imgs_to_batch(self, imgs):
        """将图像列表转换为批次数据
        
        Args:
            imgs (list): 图像列表
            
        Returns:
            dict: 批次数据字典
        """
        batch_data = {
            "inputs": [],
            "data_samples": [],
        }

        for img in imgs:
            # 调整图像通道顺序并转换为张量
            img = torch.permute(torch.from_numpy(img), (2, 0, 1)).contiguous()
            batch_data['inputs'].append(img)
            data_sample = DetDataSample(metainfo={
                'img_shape': (img.shape[1], img.shape[2]),
            })

            gt_instance = InstanceData()
            data_sample.gt_instances = gt_instance

            batch_data['data_samples'].append(data_sample)

        return batch_data

    def predict(self, imgs, bgr=False, confidence=0.0, max_size=-1) -> List[PredictionResult]:
        """预测图像中的目标
        
        Args:
            imgs (list): 输入图像列表
            bgr (bool): 是否为BGR格式
            confidence (float): 置信度阈值
            max_size (int): 最大图像尺寸
            
        Returns:
            list: 预测结果列表
        """
        # 转换为BGR格式
        if not bgr:
            for i in range(len(imgs)):
                imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)

        # 调整图像大小
        img_scales = np.ones((len(imgs),))
        if max_size > 0:
            for i in range(len(imgs)):
                max_hw = max(imgs[i].shape[0], imgs[i].shape[1])
                if max_hw > max_size:
                    rate = max_size / max_hw
                    imgs[i] = cv2.resize(imgs[i], (int(imgs[i].shape[1] * rate), int(imgs[i].shape[0] * rate)))
                    img_scales[i] = rate

        # 模型推理
        device = self.device
        with torch.no_grad():
            batch_data = self.imgs_to_batch(imgs)
            batch_data = torch_utils.batch_to_device(batch_data, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                forward_result = self.forward(batch_data)
                preds = self.postprocess(forward_result, batch_data, confidence)
        
        # 还原预测框尺寸
        for i in range(len(preds)):
            pred = preds[i]
            pred.bboxes = pred.bboxes / img_scales[i]

        return preds

    def compute_loss(self, batch_data, forward_result):
        """计算损失
        
        Args:
            batch_data (dict): 输入数据
            forward_result (dict): 前向传播结果
            
        Returns:
            tuple: 损失值和信息字典
        """
        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k, v in losses.items():
                new_k = '{}{}'.format(k, idx)
                if isinstance(v, list) or isinstance(v, tuple):
                    new_losses[new_k] = [i * weight for i in v]
                else:
                    new_losses[new_k] = v * weight
            return new_losses

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
            for i in range(len(self.model.roi_head)):
                roi_losses = self.model.roi_head[i].loss(x, proposal_list,
                                                data_samples)
                if self.model.with_pos_coord:
                    positive_coords.append(roi_losses.pop('pos_coords'))
                else:
                    if 'pos_coords' in roi_losses.keys():
                        roi_losses.pop('pos_coords')
                roi_losses = upd_loss(roi_losses, idx=i)
                losses.update(roi_losses)

            for i in range(len(self.model.bbox_head)):
                bbox_losses = self.model.bbox_head[i].loss(x, data_samples)
                if self.model.with_pos_coord:
                    pos_coords = bbox_losses.pop('pos_coords')
                    positive_coords.append(pos_coords)
                else:
                    if 'pos_coords' in bbox_losses.keys():
                        bbox_losses.pop('pos_coords')
                bbox_losses = upd_loss(bbox_losses, idx=i + len(self.model.roi_head))
                losses.update(bbox_losses)

            if self.model.with_pos_coord and len(positive_coords) > 0:
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
            loss = torch.tensor(0.0, device=forward_result['img_feats'][0].device)
            info = {
                'loss': 0.0,
                'cls': 0.0,
                'box': 0.0,
                'giou': 0.0,
            }

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