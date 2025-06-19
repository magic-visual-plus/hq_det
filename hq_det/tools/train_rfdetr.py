import sys
from loguru import logger
from hq_det.models import rfdetr
from hq_det.trainer import HQTrainer, HQTrainerArguments, add_stats
from hq_det.models.rfdetr import datasets as rfdetr_datasets
import os
import torch
import math
from torch import distributed

from tqdm import tqdm
import torch.optim
import torchvision
from hq_det import torch_utils
from hq_det.dataset import CocoDetection as HQCocoDetection
from hq_det.models.rfdetr.util.drop_scheduler import drop_scheduler
import hq_det.models.rfdetr.util.misc as utils
from hq_det.models.rfdetr.util.utils import ModelEma
from hq_det.models.rfdetr.util.misc import NestedTensor
from hq_det.models.rfdetr.util.misc import nested_tensor_from_tensor_list

class MyTrainer(HQTrainer):
    def __init__(self, args: HQTrainerArguments):
        super().__init__(args)

    def build_model(self):
        id2names = self.args.class_id2names
        self.args.model_argument.update({
            'dataset_dir': self.args.data_path,
            'num_classes': len(id2names)
        })
        model = rfdetr.HQRFDETR(class_id2names=id2names, **self.args.model_argument)
        return model
    
    def _collate_fn(self, batch):
        from hq_det.models.rfdetr.util.misc import collate_fn
        batch = collate_fn(batch)

        return batch
    
    def collate_fn(self, batch):
        def safe_to_tensor(data, dtype=None):  
            if isinstance(data, torch.Tensor):  
                result = data.clone().detach()  
                if dtype is not None:  
                    result = result.to(dtype=dtype)
            else:  
                result = torch.as_tensor(data, dtype=dtype)  
            
            return result
        max_h, max_w = 0, 0
        for b in batch:
            h, w = b["img"].shape[:2]
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            
        max_h = round(max_h / 56) * 56
        max_w = round(max_w / 56) * 56
        for b in batch:
            if not isinstance(b['img'], torch.Tensor):
                b['img'] = torch.tensor(b['img'], dtype=torch.float32).permute(2, 0, 1)
            b['img'], b['bboxes_cxcywh_norm'] = torch_utils.pad_image(b['img'], b['bboxes_cxcywh_norm'], (max_h, max_w))

        targets = [{
            'boxes': safe_to_tensor(b['bboxes_cxcywh_norm'], dtype=torch.float32),
            'labels': safe_to_tensor(b['cls'], dtype=torch.int64),
            'iscrowd': safe_to_tensor(b['iscrowd'], dtype=torch.int64),
            'area': safe_to_tensor(b['area'], dtype=torch.float32),
            # 'image_id': b['image_id'],
            'orig_size': safe_to_tensor(b['original_shape'], dtype=torch.int64),
            'size': safe_to_tensor([max_h, max_w], dtype=torch.int64),
        } for b in batch]

        return {
            'targets': tuple(targets),
            'image_id': [b['image_id'] for b in batch],
            'bboxes_xyxy': torch.cat([b['bboxes_xyxy'] for b in batch], 0),
            'cls': torch.cat([b['cls'] for b in batch], 0),
            'batch_idx': torch.cat([b['batch_idx']+i for i, b in enumerate(batch)], 0),
            'img': torch.stack([b['img'] for b in batch], 0)
        }

    def build_dataset(self, train_transforms=None, val_transforms=None):
        # Load the dataset using the specified path and device
        path_train = os.path.join(self.args.data_path, "train")
        path_val = os.path.join(self.args.data_path, "valid")
        image_path_train = path_train
        image_path_val = path_val
        annotation_file_train = os.path.join(path_train, "_annotations.coco.json")
        annotation_file_val = os.path.join(path_val, "_annotations.coco.json")

        dataset_train = HQCocoDetection(
            image_path_train, annotation_file_train, transforms=train_transforms
        )
        dataset_val = HQCocoDetection(
            image_path_val, annotation_file_val, transforms=val_transforms
        )
        return dataset_train, dataset_val
    
    def _build_dataset(self, train_transforms=None, val_transforms=None):
        from hq_det.models.rfdetr.datasets.coco import CocoDetection, make_coco_transforms
        path_train = os.path.join(self.args.data_path, "train")
        path_val = os.path.join(self.args.data_path, "valid")
        image_path_train = path_train
        image_path_val = path_val
        annotation_file_train = os.path.join(path_train, "_annotations.coco.json")
        annotation_file_val = os.path.join(path_val, "_annotations.coco.json")

        dataset_val = CocoDetection(
            image_path_val, annotation_file_val, transforms=make_coco_transforms(
                'val', self.args.image_size, False
            )
        )
        dataset_train = CocoDetection(
            image_path_train, annotation_file_train, transforms=make_coco_transforms(
                'train', self.args.image_size, True
            )
        )
        return dataset_train, dataset_val



    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, total_iters=self.args.num_epoches,
            end_factor=self.args.lr_min / self.args.lr0
        )

    def before_training_start(self, dataloader_train, dataloader_val, model, optimizer, scheduler, scaler, device):
        schedules = {}
        args = model.args

        if args.use_ema:
            self.ema_m = ModelEma(model, decay=args.ema_decay, tau=args.ema_tau)
        else:
            self.ema_m = None

        self.effective_batch_size = self.args.batch_size * self.args.gradient_update_interval
        total_batch_size = self.effective_batch_size * utils.get_world_size()
        # 计算每轮训练的步数
        num_training_steps_per_epoch = (len(dataloader_train.dataset) + total_batch_size - 1) // total_batch_size
        if args.dropout > 0:
            schedules['do'] = drop_scheduler(
                args.dropout, args.epochs, num_training_steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule)
            print("Min DO = %.7f, Max DO = %.7f" % (min(schedules['do']), max(schedules['do'])))
        if args.drop_path > 0:
            schedules['dp'] = drop_scheduler(
                args.drop_path, args.epochs, num_training_steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule)
            print("Min DP = %.7f, Max DP = %.7f" % (min(schedules['dp']), max(schedules['dp'])))
        self.schedules = schedules  # 保存dropout和drop path调度器
        self.num_training_steps_per_epoch = num_training_steps_per_epoch  # 保存每轮训练的步数
    
    def _train_one_epoch(self, 
            model: torch.nn.Module,
            bar_train: tqdm,
            optimizer: torch.optim.Optimizer, 
            scheduler,
            scaler, 
            device: str,
        ):
        train_losses = []   # 训练损失
        train_info = {}   # 训练信息

        args = model.args   # 获取模型参数
        criterion = model.criterion # 损失函数
        model = model.model
        model.to(device)
        model.train()   # 设置模型为训练模式
        criterion.train()   # 设置criterion为训练模式
        optimizer.zero_grad()   # 清空梯度
        schedules = self.schedules  # 保存dropout和drop path调度器
        ema_m = self.ema_m  # 是否使用EMA模型
        max_norm = args.clip_max_norm   # 梯度裁剪的最大范数
        vit_encoder_num_layers = args.vit_encoder_num_layers    # 设置vit编码器层数
        batch_size = self.args.batch_size * self.args.gradient_update_interval
        sub_batch_size = batch_size // self.args.gradient_update_interval

        if "dp" in schedules:
                if args.distributed:
                    model.module.update_drop_path(
                        schedules["dp"][it], vit_encoder_num_layers
                    )
                else:
                    model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        for i_batch, batch_data in enumerate(bar_train):
            if i_batch == len(bar_train) - 1:
                break
            it = i_batch
            batch_data = torch_utils.batch_to_device(batch_data, device)
            samples, targets = batch_data['img'], batch_data['targets']
                
            for i in range(self.args.gradient_update_interval):
                start_idx = i * sub_batch_size
                final_idx = start_idx + sub_batch_size
                new_samples_tensors = samples.tensors[start_idx:final_idx]
                new_samples = NestedTensor(new_samples_tensors, samples.mask[start_idx:final_idx])
                new_samples = new_samples.to(device)    
                new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.enable_amp):
                    outputs = model(new_samples, new_targets)
                    loss_dict = criterion(outputs, new_targets)
                    weight_dict = criterion.weight_dict
                    losses = sum(
                        (1 / self.args.gradient_update_interval) * loss_dict[k] * weight_dict[k]
                        for k in loss_dict.keys()
                        if k in weight_dict
                    )
                scaler.scale(losses).backward()    
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {
                f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
            }
            loss_dict_reduced_scaled = {
                k:  v * weight_dict[k]
                for k, v in loss_dict_reduced.items()
                if k in weight_dict
            }
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print(loss_dict_reduced)
                raise ValueError("Loss is {}, stopping training".format(loss_value))

            if max_norm > 0:
                scaler.unscale_(optimizer)  # 取消缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # 裁剪梯度
            scaler.update()  # 更新梯度缩放器
            optimizer.zero_grad()

            if ema_m is not None:
                if it >= 0:
                    ema_m.update(model)
            # 更新进度条和统计信息
            info = {
                'loss': loss_value,
                'class_error': loss_dict['class_error'].item(),
                'cls': loss_dict['loss_ce'].item(),
                'box': loss_dict['loss_bbox'].item(),
                'giou': loss_dict['loss_giou'].item(),
            }
            bar_train.set_postfix(**info)
            train_losses.append(loss_value)
            train_info = add_stats(train_info, info)
            
                
        return train_losses, train_info


        