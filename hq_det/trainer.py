import os
import time

import torch.distributed
from . import torch_utils
import torch.utils.data
import loguru
import numpy as np
from . import augment
from . import evaluate
from .common import PredictionResult
from tqdm import tqdm
import torch
from .common import HQTrainerArguments
from torch import distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from .models.base import HQModel
from .dataset import CocoDetection
from typing import List, Tuple

from .print_utils import (
    print_model_summary, 
    print_dataset_summary, 
    print_augmentation_steps, 
    print_training_arguments
)

def add_stats(info1, info2):
    for k, v in info2.items():
        if k not in info1:
            info1[k] = v
        else:
            info1[k] += v
            pass
        pass
    return info1


def divide_stats(info, n):
    for k, v in info.items():
        info[k] = v / n
        pass
    return info


def format_stats(info):
    parts = []
    for k, v in info.items():
        if isinstance(v, float):
            parts.append(f'{k}: {v:.2f}')
        elif isinstance(v, int):
            parts.append(f'{k}: {v}')
        else:
            parts.append(f'{k}: {v}')
            pass
        pass
    return ','.join(parts)


def extract_ground_truth(batch_data):
    batch_size = len(batch_data['image_id'])
    gt_boxes = batch_data['bboxes_xyxy']
    gt_cls = batch_data['cls']
    image_ids = batch_data['image_id']

    gt_records = []
    for i in range(batch_size):
        image_id = image_ids[i]
        mask = batch_data['batch_idx'] == i
        if mask.sum() == 0:
            gt_bboxes = np.zeros((0, 4))
            gt_cls = np.zeros((0,))
        else:
            # add gt
            gt_bboxes = batch_data['bboxes_xyxy'][mask].cpu().numpy()
            gt_cls = batch_data['cls'][mask].cpu().numpy()

        record =  PredictionResult(
            image_id=image_id,
            bboxes=gt_bboxes,
            scores=np.ones(gt_bboxes.shape[0]),
            cls=gt_cls,
        )
        gt_records.append(record)

    return gt_records


class DefaultAugmentation:
    """default data augmentation class"""
    
    @staticmethod
    def get_train_transforms(image_size, p=0.3):
        """get train data augmentation"""
        transforms = []
        transforms.append(augment.ToNumpy())
        
        # 数据增强
        transforms.append(augment.RandomCrop(p=p))
        transforms.append(augment.RandomResize(p=p, max_size=image_size))
        # prevent image is too big for speed
        transforms.append(augment.Resize(max_size=image_size*2))

        transforms.append(augment.RandomHorizontalFlip())
        transforms.append(augment.RandomVerticalFlip())
        transforms.append(augment.RandomGrayScale())
        transforms.append(augment.RandomShuffleChannel())
        transforms.append(augment.RandomRotate90(p=p))
        transforms.append(augment.RandomAspectRatio(p=p))
        transforms.append(augment.RandomRotate(p=p, degrees=30))
        transforms.append(augment.RandomAffine(p=p))
        transforms.append(augment.RandomPerspective(p=p))
        transforms.append(augment.RandomNoise(p=p))
        transforms.append(augment.RandomBrightness(p=p, alpha=0.1))
        transforms.append(augment.RandomPixelValueShift(p=p))
        transforms.append(augment.RandomShift(p=p, max_shift=0.02))
        
        # basic processing
        transforms.append(augment.Resize(max_size=image_size))
        transforms.append(augment.FilterSmallBox())
        transforms.append(augment.Format())
        
        return augment.Compose(transforms)
    
    @staticmethod
    def get_val_transforms(image_size):
        """get validation data augmentation"""
        transforms = []
        transforms.append(augment.ToNumpy())
        transforms.append(augment.Resize(max_size=image_size))
        transforms.append(augment.FilterSmallBox())
        transforms.append(augment.Format())
        
        return augment.Compose(transforms)


class HQTrainer:
    def __init__(self, args: HQTrainerArguments):
        self.args = args
        self.logger = loguru.logger
        self.results_file = os.path.join(args.output_path, 'results.csv')
        self.training_state = {
            'current_epoch': 0,     # current epoch
            'best_metric': 0.0,     # best validation metric
            'train_info': {},       # train metrics
            'val_info': {},         # validation metrics
        }
        self.HQ_DEBUG =  int(os.environ.get('HQ_DEBUG', '1'))

    def setup_training_environment(self):
        # print training arguments
        if self.HQ_DEBUG:
            print_training_arguments(self.args)
        
        # setup datasets and transforms
        self.dataset_train, self.dataset_val = self._setup_datasets_and_transforms()
        
        # setup class configuration
        self.eval_class_ids = self._setup_class_configuration()
        
        # setup model
        self.model: HQModel = self._setup_model()
        
        # setup distributed training
        self.model, self.device, self.sampler_train, self.sampler_val = self._setup_distributed_training()
        
        # setup dataloaders
        self.dataloader_train, self.dataloader_val = self._setup_dataloaders()
        
        # setup optimization components
        self.optimizer, self.scheduler, self.scaler = self._setup_optimization_components()
        
        # setup output directories
        self._setup_output_directories()

    def build_dataset(self, train_transforms, val_transforms) -> Tuple[CocoDetection, CocoDetection]:
        raise NotImplementedError

    def build_model(self) -> HQModel:
        raise NotImplementedError
    
    def collate_fn(self, batch):
        raise NotImplementedError

    def build_transforms(self, aug=True):
        if aug:
            return DefaultAugmentation.get_train_transforms(self.args.image_size)
        else:
            return DefaultAugmentation.get_val_transforms(self.args.image_size)

    def build_optimizer(self, model: HQModel) -> torch.optim.Optimizer:
        if isinstance(model, DDP):
            model = model.module
        param_dict = model.get_param_dict(self.args)
        return torch.optim.AdamW(param_dict, lr=self.args.lr0)

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, total_iters=self.args.num_epoches,
            end_factor=self.args.lr_min / self.args.lr0
        )
    
    def is_master(self) -> bool:
        return len(self.args.devices) == 1 or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)
    
    def compute_loss(self, model: HQModel, batch_data, forward_result):
        if isinstance(model, DDP):
            model = model.module
        return model.compute_loss(batch_data, forward_result)
    
    def postprocess(self, model: HQModel, batch_data, forward_result) -> List[PredictionResult]:
        if isinstance(model, DDP):
            model = model.module
        return model.postprocess(forward_result, batch_data)
    
    def save_model(self, model: HQModel, path: str):
        if isinstance(model, DDP):
            model = model.module
        model.save(path)

    def optimizer_step(
        self, 
        optimizer: torch.optim.Optimizer, 
        scaler: torch.cuda.amp.GradScaler, 
        model: HQModel
    ):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    def train_step(
        self, 
        model: HQModel, 
        batch_data, 
        optimizer: torch.optim.Optimizer, 
        scaler: torch.cuda.amp.GradScaler, 
        device: str
    ) -> Tuple[torch.Tensor, dict]:
        batch_data = torch_utils.batch_to_device(batch_data, device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.enable_amp):
            forward_result = model(batch_data)
            loss, info = self.compute_loss(model, batch_data, forward_result)
            
            # gradient synchronization for distributed training
            if len(self.args.devices) > 1:
                for k, v in info.items():
                    if isinstance(v, torch.Tensor):
                        info[k] = distributed.reduce(v, op=torch.distributed.ReduceOp.SUM)
        
        # backward
        scaler.scale(loss / self.args.gradient_update_interval).backward()
        
        return loss, info

    def valid_step(self, model: HQModel, batch_data, device: str) -> Tuple[torch.Tensor, dict, List[PredictionResult]]:
        batch_data = torch_utils.batch_to_device(batch_data, device)
        
        with torch.no_grad():
            forward_result = model(batch_data)
            loss, info = self.compute_loss(model, batch_data, forward_result)
            preds = self.postprocess(model, batch_data, forward_result)
            
            # set image id for prediction results
            for pred, image_id in zip(preds, batch_data['image_id']):
                pred.image_id = image_id
            
            return loss, info, preds

    def train_epoch(self, epoch: int) -> Tuple[List[float], dict]:
        self.model.train()
        train_losses = []
        train_info = {}
        
        # create progress bar
        bar_train = self._create_progress_bar(
            self.dataloader_train, 
            f"Train Epoch[{epoch}/{self.args.num_epoches + self.args.warmup_epochs - 1}]"
        )
        
        for i_batch, batch_data in enumerate(bar_train):
            # train step
            loss, info = self.train_step(
                self.model, batch_data, self.optimizer, self.scaler, self.device
            )
            
            # update progress bar and statistics
            bar_train.set_postfix(**info)
            train_losses.append(loss.item())
            train_info = add_stats(train_info, info)
            
            # gradient accumulation
            if i_batch % self.args.gradient_update_interval == 0:
                self.optimizer_step(self.optimizer, self.scaler, self.model)
        
        # process last batch
        if i_batch % self.args.gradient_update_interval != 0:
            self.optimizer_step(self.optimizer, self.scaler, self.model)

        return train_losses, train_info

    def valid_epoch(self, epoch: int) -> Tuple[List[float], dict, dict]:
        self.model.eval()
        val_losses = []
        val_info = dict()
        stat = dict()
        all_preds = []
        all_gts = []

        if self.is_master():
            bar_val = self._create_progress_bar(
                self.dataloader_val, 
                f"Valid Epoch[{epoch}/{self.args.num_epoches + self.args.warmup_epochs - 1}]"
            )
            
            for i_batch, batch_data in enumerate(bar_val):
                loss, info, preds = self.valid_step(self.model, batch_data, self.device)
                
                val_info = add_stats(val_info, info)
                val_losses.append(loss.item())
                all_preds.extend(preds)
                all_gts.extend(extract_ground_truth(batch_data))
                
                bar_val.set_postfix(**info)
                pass
            bar_val.close()
            val_info = divide_stats(val_info, len(self.dataloader_val))
            stat = self._process_validation_results(all_preds, all_gts, self.eval_class_ids)
            pass

        # distributed.barrier()
        return val_losses, val_info, stat

    def _setup_datasets_and_transforms(self) -> Tuple[CocoDetection, CocoDetection]:
        trainsforms_train = self.build_transforms(aug=True)
        trainsforms_val = self.build_transforms(aug=False)
        
        dataset_train, dataset_val = self.build_dataset(
            trainsforms_train, trainsforms_val)

        # print dataset information
        if self.HQ_DEBUG:
            print_dataset_summary(dataset_train, dataset_val)
            # print augmentation steps
            print_augmentation_steps(trainsforms_train, trainsforms_val)

        return dataset_train, dataset_val

    def _setup_class_configuration(self) -> List[int]:
        if self.args.class_id2names is None:
            self.args.class_id2names = self.dataset_train.class_id2names

        if self.args.eval_class_names is None:
            eval_class_ids = list(self.dataset_train.class_id2names.keys())
        else:
            class_names2id = {v: k for k, v in self.args.class_id2names.items()}
            eval_class_ids = [
                class_names2id[class_name] for class_name in self.args.eval_class_names if class_name in class_names2id]

        if len(eval_class_ids) == 0:
            # cannot find any class, use all class ids to evaluate
            eval_class_ids = list(self.dataset_train.class_id2names.keys())
        
        return eval_class_ids

    def _setup_model(self) -> HQModel:
        """build model and log number of parameters"""
        model = self.build_model()
        if self.HQ_DEBUG:
            # print_model_summary(model)
            pass
        return model

    def _setup_distributed_training(self) -> Tuple[HQModel, str, torch.utils.data.Sampler, torch.utils.data.Sampler]:
        if len(self.args.devices) > 1:
            # distributed training
            self.logger.info("Setting up distributed training environment...")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            torch.distributed.init_process_group("nccl")
            rank = distributed.get_rank()
            # create model and move it to GPU with id rank
            device_id = rank % torch.cuda.device_count()
            device = f'cuda:{device_id}'
            self.model.to(device)
            model = DDP(self.model, device_ids=[device_id], find_unused_parameters=self.args.find_unused_parameters)
            
            self.logger.info(f"Distributed training initialized - Rank: {rank}, Device: {device}, World Size: {torch.distributed.get_world_size()}")
            
            sampler_train = torch.utils.data.distributed.DistributedSampler(
                self.dataset_train,
                num_replicas=torch.distributed.get_world_size(),
                rank=rank,
                shuffle=True,
            )
            sampler_val = None
        else:
            # single GPU training
            self.logger.info("Setting up single GPU training environment...")
            sampler_train = torch.utils.data.RandomSampler(self.dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(self.dataset_val)
            device = self.args.devices[0]
            device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            model = self.model
            
            self.logger.info(f"Single GPU training initialized - Device: {device}")
        
        return model, device, sampler_train, sampler_val

    def _setup_dataloaders(self) -> Tuple[
        torch.utils.data.DataLoader, 
        torch.utils.data.DataLoader
    ]:
        """setup dataloaders"""
        dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=self.args.batch_size, num_workers=self.args.num_data_workers, 
            collate_fn=self.collate_fn, sampler=self.sampler_train)
        dataloader_val = torch.utils.data.DataLoader(
            self.dataset_val, batch_size=self.args.batch_size, num_workers=self.args.num_data_workers,
            collate_fn=self.collate_fn, sampler=self.sampler_val)
        return dataloader_train, dataloader_val

    def _setup_optimization_components(self) -> Tuple[
        torch.optim.Optimizer, 
        torch.optim.lr_scheduler._LRScheduler, 
        torch.cuda.amp.GradScaler
    ]:
        """setup optimizer, scheduler and gradient scaler"""
        optimizer = self.build_optimizer(self.model)
        scheduler = self.build_scheduler(optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.enable_amp)
        return optimizer, scheduler, scaler

    def _setup_output_directories(self) -> None:
        """setup output directories"""
        os.makedirs(self.args.checkpoint_path, exist_ok=True)

    def _update_training_state(self, epoch: int, train_info: dict, val_info: dict,
            metric: float = None) -> None:
        """update training state"""
        self.training_state['current_epoch'] = epoch
        self.training_state['train_info'] = train_info
        self.training_state['val_info'] = val_info
        if metric is not None and metric > self.training_state['best_metric']:
            self.training_state['best_metric'] = metric

    def _should_save_best_model(self, metric: float) -> bool:
        """check if best model should be saved"""
        return metric > self.training_state['best_metric']

    def _save_best_model(self, model: HQModel, metric: float) -> None:
        """save best model"""
        if self._should_save_best_model(metric) and self.is_master():
            best_model_path = os.path.join(self.args.checkpoint_path, 'best_model.pth')
            self.save_model(model, best_model_path)
            self.logger.info(f'New best model saved with metric: {metric:.4f}')

    def _create_progress_bar(self, dataloader, desc, disable=False) -> tqdm:
        """create progress bar"""
        if self.is_master() and not disable:
            return tqdm(dataloader, desc=desc)
        else:
            bar = tqdm(dataloader, desc=desc)
            bar.disable = True
            return bar

    def _compute_epoch_metrics(self, train_losses, val_losses, val_info) -> dict:
        """compute epoch metrics"""
        metrics = {
            'train_loss': np.mean(train_losses) if train_losses else 0.0,
            'val_loss': np.mean(val_losses) if val_losses else 0.0,
        }
        
        # add validation metrics
        for key, value in val_info.items():
            if isinstance(value, (int, float)):
                metrics[f'val_{key}'] = value
        
        return metrics

    def _process_validation_results(self, all_preds, all_gts, eval_class_ids) -> dict:
        """process validation results"""
        if not all_preds or not all_gts:
            return {}
        
        # 评估检测结果
        stat = evaluate.eval_detection_result_by_class_id(
            all_gts, all_preds, eval_class_ids
        )
        
        return stat

    def _log_learning_rates(self, optimizer: torch.optim.Optimizer) -> dict:
        """log learning rates"""
        lr_info = {}
        for i, param_group in enumerate(optimizer.param_groups):
            lr_info[f'lr_group_{i}'] = param_group['lr']
        return lr_info

    def _create_epoch_summary(self, i_epoch: int, train_losses: List[float],
                              val_losses: List[float], val_info: dict,
                              train_time: float, val_time: float,
                              epoch_time: float, stat: dict) -> dict:
        """create epoch summary"""
        lr_info = self._log_learning_rates(self.optimizer)
        metrics = self._compute_epoch_metrics(train_losses, val_losses, val_info)
        
        summary = {
            'epoch': i_epoch,
            'train_loss': metrics['train_loss'],
            'val_loss': metrics['val_loss'],
            'train_time': train_time,
            'val_time': val_time,
            'epoch_time': epoch_time,
            'stat': stat,
            'lr_info': lr_info,
            'val_info': val_info,
        }
        
        return summary

    def _log_epoch_summary(self, summary: dict) -> None:
        """log epoch summary"""
        if not self.is_master():
            return
        
        epoch = summary['epoch']
        train_loss = summary['train_loss']
        val_loss = summary['val_loss']
        val_info = summary['val_info']
        lr_info = summary['lr_info']
        train_time = summary['train_time']
        val_time = summary['val_time']
        epoch_time = summary['epoch_time']
        
        # log learning rate
        lr_str = ', '.join([f'{k}: {v:.6f}' for k, v in lr_info.items()])
        
        # log loss and metrics
        self.logger.info(
            f'Epoch {epoch}, {lr_str}, train loss: {train_loss:.4f}, valid loss: {val_loss:.4f}, '
            f'{format_stats(val_info)}'
        )
        
        # log time information
        train_h, train_m, train_s = train_time
        val_h, val_m, val_s = val_time
        epoch_h, epoch_m, epoch_s = epoch_time
        
        self.logger.info(
            f'Elapsed Time: Train {train_h:02d}:{train_m:02d}:{train_s:02d} | '
            f'Valid {val_h:02d}:{val_m:02d}:{val_s:02d} | '
            f'Epoch {epoch_h:02d}:{epoch_m:02d}:{epoch_s:02d}'
        )

    def _save_checkpoint(self, model: HQModel) -> None:
        """save checkpoint"""
        if self.is_master():
            checkpoint_path = os.path.join(self.args.checkpoint_path, self.args.checkpoint_name)
            self.save_model(model, checkpoint_path)

    def _format_time(self, time_seconds: float) -> Tuple[int, int, int]:
        """format time"""
        hours, remainder = divmod(time_seconds, 3600)
        mins, secs = divmod(remainder, 60)
        return int(hours), int(mins), int(secs)

    def _check_early_stopping(self, val_loss: float, patience: int = 10) -> bool:
        """check if early stopping is needed"""
        if not hasattr(self, '_early_stopping_counter'):
            self._early_stopping_counter = 0
            self._best_val_loss = float('inf')
        
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._early_stopping_counter = 0
        else:
            self._early_stopping_counter += 1
        
        return self._early_stopping_counter >= patience

    def run(self) -> None:
        self.setup_training_environment()

        """main training process"""
        self.logger.info("Start training...")
        self.scheduler.step()
        start_time = time.time()
        
        for i_epoch in range(self.args.num_epoches + self.args.warmup_epochs):
            epoch_start_time = time.time()
            
            # Training process
            train_losses, train_info = self.train_epoch(i_epoch)
            train_time = time.time() - epoch_start_time
            train_time_formatted = self._format_time(train_time)
            
            # Validation process
            val_start_time = time.time()
            val_losses, val_info, stat = self.valid_epoch(i_epoch)
            val_time = time.time() - val_start_time
            val_time_formatted = self._format_time(val_time)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            epoch_time_formatted = self._format_time(epoch_time)
            
            # Create and log epoch summary
            summary = self._create_epoch_summary(
                i_epoch, train_losses, val_losses, val_info,
                train_time_formatted, val_time_formatted, epoch_time_formatted, stat
            )
            
            self._log_epoch_summary(summary)
            
            # Update training state
            self._update_training_state(i_epoch, train_info, val_info, stat.get('mAP', 0.0))

            # Save results and checkpoints
            if self.is_master():
                self.save_epoch_result(i_epoch, stat, self.args.output_path)
                self._save_best_model(self.model, stat.get('mAP', 0.0))
            
            self._save_checkpoint(self.model)

            # Check early stopping
            if self.args.early_stopping and \
                self._check_early_stopping(summary['val_loss'], self.args.early_stopping_patience):
                self.logger.info(f'Early stopping triggered at epoch {i_epoch}')
                break
            
            # Update scheduler
            if i_epoch >= self.args.warmup_epochs:
                self.scheduler.step()
                pass
            pass
        
        # log training summary
        self._log_training_summary(start_time)

    def _log_training_summary(self, start_time: float) -> None:
        """log training summary"""
        total_time = time.time() - start_time
        total_time_formatted = self._format_time(total_time)
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        
        self.logger.info(f'Start Time: {start_time_str}, End Time: {end_time_str}')
        self.logger.info(f'Total Time: {total_time_formatted[0]:02d}:{total_time_formatted[1]:02d}:{total_time_formatted[2]:02d}')

    def save_epoch_result(self, iepoch: int, stat: dict, output_path: str) -> None:
        header = ['mAP', 'precision', 'recall', 'f1_score', 'fnr', 'confidence', 'train/box_loss', 'train/cls_loss', 'train/giou_loss', 'val/box_loss', 'val/cls_loss', 'val/giou_loss']
        
        if iepoch == 0:
            # add header
            with open(self.results_file, 'w') as f:
                f.write(','.join(header) + '\n')

        with open(self.results_file, 'a') as f:
            values = [stat[colname] for colname in header if colname in stat]
            f.write(','.join([str(v) for v in values]) + '\n')
            pass

        # save curve
        plots_path = os.path.join(output_path, 'plots', f'epoch{iepoch}')
        os.makedirs(plots_path, exist_ok=True)
        curve_file = os.path.join(plots_path, 'pr_curve.csv')
        
        precisions = stat['precisions']
        recalls = stat['recalls']

        with open(curve_file, 'w') as f:
            f.write('px,all\n')

            for i in range(len(precisions)):
                f.write(f'{recalls[i]},{precisions[i]}\n')
