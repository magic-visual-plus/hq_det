"""PointDINO training through HQTrainer's shared training lifecycle."""

import csv
from copy import deepcopy
from pathlib import Path
import os
import random
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from mmengine.dataset import pseudo_collate
from mmengine.optim import build_optim_wrapper
from mmengine.runner import set_random_seed
from mmdet.registry import METRICS

from hq_det.common import HQTrainerArguments
from hq_det.trainer import HQTrainer, add_stats, divide_stats
from hq_det.pointdino_dataset import PointDINODataset, PointDINODatasetProvider
from hq_det.models.pointdino.pointdino_hq import PointDINOModel


class PointDINOTrainerArguments(HQTrainerArguments):
    """HQ training arguments extended with the point model's configuration."""

    image_size: Optional[Union[int, Tuple[int, int]]] = None
    cfg: Any
    eval_batch_size: int = 1
    resume: bool = False
    warmup_epochs: int = 0
    sync_bn: bool = False
    max_grad_norm: float = 0.1


class PointDINOTrainer(HQTrainer):
    """Reuse HQ setup, run, train_step, optimizer_step and logging hooks.

    Point data, complete detector loss and point metrics replace the box-only
    adapters. The source optimizer settings and epoch schedule are retained.
    """

    def __init__(self, args):
        super().__init__(args)
        self.cfg = args.cfg
        self.start_epoch = 0
        self.training_state['best_metric'] = float('-inf')
        self.best_metric_name = self.cfg.default_hooks.checkpoint.save_best
        self._rng_states = None
        seed = self.cfg.get('randomness', {}).get('seed', 0)
        set_random_seed(0 if seed is None else seed,
                        deterministic=self.cfg.get('randomness', {}).get('deterministic', False))

    @classmethod
    def from_config(cls, cfg, *, devices=None, image_size=None,
                    checkpoint_name='ckpt.pth'):
        if devices is None:
            devices = list(range(int(os.environ.get('WORLD_SIZE', '1')))) if torch.cuda.is_available() else []
        if len(devices) > 1 and int(os.environ.get('WORLD_SIZE', '1')) == 1:
            raise ValueError('Multiple GPUs require torchrun --nproc_per_node=<GPU count>.')
        if int(os.environ.get('WORLD_SIZE', '1')) > 1 and len(devices) < 2:
            raise ValueError('With torchrun, use devices=None or list all training GPUs.')
        if int(os.environ.get('WORLD_SIZE', '1')) > 1 and not torch.cuda.is_available():
            raise ValueError('Distributed PointDINO training requires CUDA.')
        optimizer = cfg.optim_wrapper.optimizer
        args = PointDINOTrainerArguments(
            data_path=str(cfg.get('data_root', '')), output_path=str(cfg.work_dir),
            checkpoint_path=str(cfg.work_dir), checkpoint_name=checkpoint_name,
            cfg=cfg, image_size=image_size, devices=devices,
            num_epoches=cfg.train_cfg.max_epochs,
            batch_size=cfg.train_dataloader.batch_size,
            eval_batch_size=cfg.val_dataloader.batch_size,
            num_data_workers=cfg.train_dataloader.num_workers,
            lr0=optimizer.lr,
            lr_backbone_mult=cfg.optim_wrapper.get('paramwise_cfg', {}).get(
                'custom_keys', {}).get('backbone', {}).get('lr_mult', 1.),
            gradient_update_interval=cfg.optim_wrapper.get('accumulative_counts', 1),
            max_grad_norm=cfg.optim_wrapper.get('clip_grad', {}).get('max_norm', 0.1),
            enable_amp=cfg.optim_wrapper.get('type') == 'AmpOptimWrapper',
            find_unused_parameters=cfg.get('model_wrapper_cfg', {}).get(
                'find_unused_parameters', not cfg.model.get('use_dn', True)),
            resume=cfg.get('resume', False))
        return cls(args)

    def build_dataset_provider(self):
        return PointDINODatasetProvider(
            self.cfg.train_dataloader.dataset, self.cfg.val_dataloader.dataset,
            self.cfg.test_dataloader.dataset)

    def build_train_transforms(self, image_size, p=0.3):
        return PointDINODataset._compose_pipeline(self.cfg.train_dataloader.dataset.pipeline)

    def build_valid_transforms(self, image_size):
        return PointDINODataset._compose_pipeline(self.cfg.val_dataloader.dataset.pipeline)

    def build_model(self):
        self.cfg.model.bbox_head.num_classes = len(self.args.class_id2names)
        checkpoint = self.cfg.get('load_from')
        if self.args.resume and not checkpoint:
            checkpoint = str(Path(self.args.checkpoint_path) / self.args.checkpoint_name)
        return PointDINOModel(self.cfg, self.args.class_id2names,
                              image_size=self.args.image_size, checkpoint=checkpoint)

    @staticmethod
    def collate_fn(batch):
        return pseudo_collate(batch)

    def is_master(self):
        return not dist.is_initialized() or dist.get_rank() == 0

    def _setup_distributed_training(self):
        if self.args.devices:
            return super()._setup_distributed_training()
        self.model.to('cpu')
        return (self.model, 'cpu', RandomSampler(self.dataset_train),
                SequentialSampler(self.dataset_val))

    def _setup_dataloaders(self):
        train, _ = super()._setup_dataloaders()
        val = DataLoader(self.dataset_val, batch_size=self.args.eval_batch_size,
                         num_workers=self.cfg.val_dataloader.num_workers,
                         collate_fn=self.collate_fn, sampler=self.sampler_val)
        return train, val

    @staticmethod
    def _unwrap(model):
        return model.module if isinstance(model, DistributedDataParallel) else model

    def build_optimizer(self, model):
        # Reuse the source paramwise constructor against the detector so old
        # MMEngine checkpoints retain the same parameter groups and order.
        config = deepcopy(self.cfg.optim_wrapper)
        config.type = 'OptimWrapper'
        config.pop('loss_scale', None)
        scaling = self.cfg.get('auto_scale_lr', {})
        if scaling.get('enable', False):
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            config.optimizer.lr *= self.args.batch_size * world_size / scaling['base_batch_size']
        return build_optim_wrapper(self._unwrap(model).model, config).optimizer

    def build_scheduler(self, optimizer):
        schedules = self.cfg.param_scheduler
        if (len(schedules) != 1 or schedules[0].type != 'MultiStepLR'
                or not schedules[0].get('by_epoch', True)
                or schedules[0].get('begin', 0) != 0):
            raise ValueError('PointDINO HQ training supports the source epoch MultiStepLR schedule.')
        schedule = schedules[0]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=schedule.milestones, gamma=schedule.get('gamma', 0.1))

    def _before_training(self):
        checkpoint = self._unwrap(self.model).loaded_checkpoint
        if self.args.resume:
            if not checkpoint or 'optimizer' not in checkpoint:
                raise ValueError('resume=True requires a training checkpoint with optimizer state; '
                                 'use resume=False for the Stage2 initialization weights.')
            self._validate_checkpoint_categories(checkpoint, self.dataset_train)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = int(checkpoint.get('meta', {}).get('epoch', 0))
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                # Source MMEngine checkpoints store the completed epoch and
                # already-stepped optimizer LR. Continue that epoch schedule.
                self.scheduler.last_epoch = self.start_epoch
                self.scheduler._step_count = self.start_epoch + 1
                self.scheduler._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            scaler_state = checkpoint.get('scaler') or checkpoint['optimizer'].get('loss_scaler')
            if scaler_state:
                self.scaler.load_state_dict(scaler_state)
            self.training_state.update(checkpoint.get('training_state', {}))
            states = checkpoint.get('rng_states')
            if states:
                state = states[dist.get_rank() if dist.is_initialized() else 0]
                random.setstate(state['python'])
                np.random.set_state(state['numpy'])
                torch.set_rng_state(state['torch'])
                if torch.cuda.is_available() and state.get('cuda') is not None:
                    torch.cuda.set_rng_state_all(state['cuda'])
        self._unwrap(self.model).loaded_checkpoint = None
        if self.is_master():
            self.cfg.dump(str(Path(self.args.output_path) / 'pointdino_config.py'))

    def _epoch_range(self):
        return range(self.start_epoch, self.args.num_epoches)

    @staticmethod
    def _validate_checkpoint_categories(checkpoint, dataset):
        metadata = checkpoint.get('meta', {}).get('dataset_meta', {})
        if 'category_ids' in metadata:
            for key in ('category_ids', 'classes'):
                if tuple(metadata[key]) != tuple(dataset.metainfo[key]):
                    raise ValueError('Checkpoint and dataset category ID-to-name mappings differ.')

    def _log_learning_rates(self, optimizer):
        # Paramwise construction can create hundreds of groups with just two
        # distinct rates. Keep HQ's epoch summary readable.
        rates = list(dict.fromkeys(group['lr'] for group in optimizer.param_groups))
        return {f'lr_{index}': rate for index, rate in enumerate(rates)}

    def train_epoch(self, epoch):
        self.model.train()
        if hasattr(self.sampler_train, 'set_epoch'):
            self.sampler_train.set_epoch(epoch)
        losses, info = [], {}
        self.optimizer.zero_grad()
        count = len(self.dataloader_train)
        if count == 0:
            raise ValueError('The PointDINO training dataset has no batches.')
        interval = self.args.gradient_update_interval
        bar = self._create_progress_bar(self.dataloader_train,
                                        f'Train Epoch[{epoch + 1}/{self.args.num_epoches}]')
        for index, batch in enumerate(bar):
            loss, step_info = self.train_step(self.model, batch, self.optimizer,
                                              self.scaler, self.device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite PointDINO loss at epoch {epoch + 1}.')
            losses.append(loss.item())
            add_stats(info, step_info)
            bar.set_postfix(loss=losses[-1])
            if (index + 1) % interval == 0 or index + 1 == count:
                remainder = (index + 1) % interval
                if index + 1 == count and remainder:
                    # HQ train_step divides by the full accumulation interval.
                    # Normalize an incomplete final window by its actual size.
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(interval / remainder)
                self.optimizer_step(self.optimizer, self.scaler, self.model)
        bar.close()
        return losses, divide_stats(info, count)

    @torch.no_grad()
    def _evaluate_loader(self, model, dataloader, evaluator_config):
        model.eval()
        evaluator = METRICS.build(evaluator_config)
        for batch in self._create_progress_bar(dataloader, 'PointDINO evaluation'):
            predictions = model(batch)
            evaluator.process(batch, predictions)
        # Evaluation is rank-zero only; BaseMetric.evaluate would start a
        # distributed gather requiring ranks that are waiting below.
        results = evaluator.compute_metrics(evaluator.results)
        prefix = evaluator.prefix or evaluator.default_prefix
        return {f'{prefix}/{key}' if prefix else key: value
                for key, value in results.items()}

    def valid_epoch(self, epoch):
        self.model.eval()
        stat = {}
        if (epoch + 1) % self.cfg.train_cfg.get('val_interval', 1) == 0:
            if self.is_master():
                stat = self._evaluate_loader(self._unwrap(self.model), self.dataloader_val,
                                              self.cfg.val_evaluator)
            if dist.is_initialized():
                payload = [stat]
                dist.broadcast_object_list(payload, src=0)
                stat = payload[0]
        return [], stat, stat

    def _finish_epoch(self, epoch, train_info, val_info, stat):
        metric = stat.get(self.best_metric_name)
        improved = metric is not None and metric > self.training_state['best_metric']
        self._update_training_state(epoch, train_info, val_info, metric)
        # Save the LR that the next epoch will use, including resume state.
        self.scheduler.step()
        state = dict(python=random.getstate(), numpy=np.random.get_state(),
                     torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        self._rng_states = [state]
        if dist.is_initialized():
            self._rng_states = [None] * dist.get_world_size()
            dist.all_gather_object(self._rng_states, state)
        if self.is_master():
            self.save_epoch_result(epoch, stat, self.args.output_path)
            self._save_checkpoint(self.model)
            hooks = self.cfg.default_hooks.checkpoint
            if (epoch + 1) % hooks.get('interval', 1) == 0 or epoch + 1 == self.args.num_epoches:
                self.save_model(self.model, str(Path(self.args.checkpoint_path) / f'epoch_{epoch + 1}.pth'))
                keep = hooks.get('max_keep_ckpts', -1)
                if keep > 0:
                    checkpoints = sorted(Path(self.args.checkpoint_path).glob('epoch_*.pth'),
                                         key=lambda path: int(path.stem.split('_')[-1]))
                    for old in checkpoints[:-keep]:
                        old.unlink()
            if improved:
                self.save_model(self.model, str(Path(self.args.checkpoint_path) / 'best_model.pth'))
                self.logger.info(f'New best {self.best_metric_name}: {metric:.6f}')

    def _step_epoch_scheduler(self, epoch):
        # Already stepped before saving resumable checkpoints in _finish_epoch.
        pass

    def save_model(self, model, path):
        model = self._unwrap(model)
        checkpoint = model.checkpoint_dict()
        checkpoint.update(optimizer=self.optimizer.state_dict(),
                          scheduler=self.scheduler.state_dict(), scaler=self.scaler.state_dict(),
                          training_state=deepcopy(self.training_state), rng_states=self._rng_states,
                          hq_pointdino=True)
        checkpoint.setdefault('meta', {})['epoch'] = self.training_state['current_epoch'] + 1
        checkpoint['meta'].setdefault('dataset_meta', {}).update(self.dataset_train.metainfo)
        torch.save(checkpoint, path)

    def save_epoch_result(self, iepoch, stat, output_path):
        row = dict(epoch=iepoch + 1,
                   **{f'train/{key}': value for key, value in self.training_state['train_info'].items()},
                   **stat)
        path = Path(self.results_file)
        if path.exists() and iepoch > 0:
            with path.open(encoding='utf-8', newline='') as stream:
                fieldnames = next(csv.reader(stream))
        else:
            fieldnames = list(row)
            evaluator = METRICS.build(self.cfg.val_evaluator)
            prefix = evaluator.prefix or evaluator.default_prefix
            for key in evaluator.compute_metrics([]):
                key = f'{prefix}/{key}' if prefix else key
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open('a' if path.exists() and iepoch > 0 else 'w',
                       encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    def test(self):
        dataset = self.build_dataset_provider().build_test_dataset()
        self.args.class_id2names = dataset.class_id2names
        model = self.build_model()
        self._validate_checkpoint_categories(model.loaded_checkpoint, dataset)
        device = (f'cuda:{self.args.devices[0]}'
                  if self.args.devices and torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.loaded_checkpoint = None
        loader = DataLoader(dataset, batch_size=self.cfg.test_dataloader.batch_size,
                            num_workers=self.cfg.test_dataloader.num_workers,
                            collate_fn=self.collate_fn, shuffle=False)
        results = self._evaluate_loader(model, loader, self.cfg.test_evaluator)
        Path(self.args.output_path).mkdir(parents=True, exist_ok=True)
        import json
        (Path(self.args.output_path) / 'metrics.json').write_text(
            json.dumps(results, indent=2), encoding='utf-8')
        self.logger.info(results)
        return results


def run(data_path, output_path='output/pointdino', *, devices=None,
        checkpoint_name='ckpt.pth', enable_amp=False, **kwargs):
    from .pointdino import build_run_config

    cfg = build_run_config(data_path, output_path, **kwargs)
    if enable_amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
    trainer = PointDINOTrainer.from_config(
        cfg, devices=devices, image_size=kwargs.get('image_size'),
        checkpoint_name=checkpoint_name)
    trainer.run()
    return trainer


def test(data_path, load_checkpoint, output_path='output/pointdino_test', *,
         devices=None, **kwargs):
    from .pointdino import build_run_config

    cfg = build_run_config(data_path, output_path, load_checkpoint=load_checkpoint,
                           evaluation=True, **kwargs)
    trainer = PointDINOTrainer.from_config(cfg, devices=devices,
                                          image_size=kwargs.get('image_size'))
    return trainer.test()
