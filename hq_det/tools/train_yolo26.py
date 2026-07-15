import json
import math
import os
import time
import warnings
from copy import copy

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from hq_det import torch_utils
from hq_det.models import yolo26
from hq_det.models.yolo import YOLO26_SCALES, build_yolo26_model_name
from hq_det.print_utils import print_augmentation_steps, print_dataset_summary
from hq_det.trainer import HQTrainer, add_stats, de_parallel
from hq_det.training import OptimizerContext
from hq_det.training.recipes.ultralytics_v847 import (
    ULTRALYTICS_V847_VERSION,
    UltralyticsV847Recipe,
)
from hq_det.yolo_dataset import HQYOLODataset

from .train_yolo import YoloTrainer, YoloTrainerArguments


class Yolo26Trainer(YoloTrainer):
    """HQ-DET trainer composed with the framework-owned 8.4.7 recipe."""

    def __init__(self, args: YoloTrainerArguments):
        self.recipe = UltralyticsV847Recipe()
        self.recipe.validate_arguments(args)
        rounded_size = math.ceil(int(args.image_size) / 32) * 32
        args.image_size = rounded_size
        super().__init__(args)
        self._global_batch_size = int(args.batch_size)
        self._optimizer_build_result = None

    def _build_yolo_hyp(self):
        return self.recipe.build_config(self.args)

    def train_step(self, model, batch_data, optimizer, scaler, device):
        batch_data = torch_utils.batch_to_device(batch_data, device)

        with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.args.enable_amp,
        ):
            forward_result = model(batch_data)
            loss, info = self.compute_loss(model, batch_data, forward_result)

        backward_loss = loss
        if torch.distributed.is_initialized():
            backward_loss = loss * torch.distributed.get_world_size()

        scaler.scale(
            backward_loss / self.args.gradient_update_interval
        ).backward()

        return loss, info

    def setup_training_environment(self):
        rank = int(os.environ.get("RANK", "-1"))
        self.recipe.runtime.initialize_seed(
            seed=int(self.yolo_hyp.seed),
            deterministic=bool(self.yolo_hyp.deterministic),
            rank=rank,
        )
        HQTrainer.setup_training_environment(self)

    def _setup_datasets_and_transforms(self):
        provider = self.build_dataset_provider()
        source_train = provider.build_train_dataset(None)
        source_val = provider.build_valid_dataset(None)

        (
            self.original_class_id2names,
            self.yolo_class_id_map,
            self.yolo_class_id2names,
        ) = self._make_contiguous_mapping(source_train.class_id2names)
        if not self.yolo_class_id2names:
            raise ValueError("The training dataset does not define any categories.")

        name_to_yolo_id = {
            name: class_id for class_id, name in self.yolo_class_id2names.items()
        }
        val_class_id_map = {}
        for original_id, name in source_val.class_id2names.items():
            if name not in name_to_yolo_id:
                raise ValueError(
                    f"Validation class '{name}' is missing from training classes."
                )
            val_class_id_map[int(original_id)] = name_to_yolo_id[name]

        if bool(getattr(self.yolo_hyp, "single_cls", False)):
            raise ValueError(
                "single_cls is not supported with HQ-DET's per-class dataset mapping."
            )

        world_size = max(len(self.args.devices), 1)
        if self._global_batch_size % world_size:
            raise ValueError(
                "YOLO26 batch_size is the global batch and must be divisible by "
                f"the number of devices ({world_size})."
            )
        rank_batch = self._global_batch_size // world_size
        self.args.class_id2names = dict(self.yolo_class_id2names)
        train_dataset = HQYOLODataset(
            source_train,
            image_size=self.args.image_size,
            hyp=self.yolo_hyp,
            class_id_map=self.yolo_class_id_map,
            class_id2names=self.yolo_class_id2names,
            augment=True,
            batch_size=rank_batch,
            rect=bool(getattr(self.yolo_hyp, "rect", False)),
            stride=32,
            pad=0.0,
            classes=getattr(self.yolo_hyp, "classes", None),
            fraction=float(getattr(self.yolo_hyp, "fraction", 1.0)),
            augmentation_strategy=self.recipe.augmentation,
            metadata_mode=True,
        )
        val_dataset = HQYOLODataset(
            source_val,
            image_size=self.args.image_size,
            hyp=self.yolo_hyp,
            class_id_map=val_class_id_map,
            class_id2names=self.yolo_class_id2names,
            augment=False,
            batch_size=rank_batch * 2,
            rect=True,
            stride=32,
            pad=0.5,
            classes=getattr(self.yolo_hyp, "classes", None),
            augmentation_strategy=self.recipe.augmentation,
            metadata_mode=True,
        )

        self.logger.info(f"YOLO26 category id mapping: {self.yolo_class_id_map}")
        if self.HQ_DEBUG:
            print_dataset_summary(train_dataset, val_dataset)
            print_augmentation_steps(
                train_dataset.transforms, val_dataset.transforms
            )
        return train_dataset, val_dataset

    def build_model(self):
        return yolo26.HQYOLO26(
            class_id2names=self.args.class_id2names,
            epochs=self.args.num_epoches,
            image_size=self.args.image_size,
            args=self.yolo_hyp,
            **self.args.model_argument,
        )

    def _setup_distributed_training(self):
        inner_model = getattr(self.model, "model", self.model)
        self.recipe.runtime.prepare_model(
            inner_model, freeze=getattr(self.yolo_hyp, "freeze", None)
        )
        return HQTrainer._setup_distributed_training(self)

    def _setup_dataloaders(self):
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else -1
        )
        if self._global_batch_size % world_size:
            raise ValueError(
                "Global YOLO26 batch size must be divisible by distributed world size."
            )
        rank_batch = self._global_batch_size // world_size
        train_sampler = self.sampler_train if world_size > 1 else None
        train_loader = self.recipe.dataloader.build(
            self.dataset_train,
            batch_size=rank_batch,
            workers=self.args.num_data_workers,
            collate_fn=self.collate_fn,
            shuffle=not self.dataset_train.rect,
            sampler=train_sampler,
            rank=rank,
        )
        val_loader = self.recipe.dataloader.build(
            self.dataset_val,
            batch_size=rank_batch * 2,
            workers=self.args.num_data_workers * 2,
            collate_fn=self.collate_fn,
            shuffle=False,
            sampler=None,
            rank=rank,
        )
        self.sampler_train = train_loader.sampler
        self.sampler_val = val_loader.sampler
        return train_loader, val_loader

    def build_optimizer(self, model):
        wrapped_model = de_parallel(model)
        inner_model = getattr(wrapped_model, "model", wrapped_model)
        context = OptimizerContext(
            num_classes=len(self.yolo_class_id2names),
            dataset_size=len(self.dataset_train),
            epochs=int(self.args.num_epoches),
            global_batch_size=self._global_batch_size,
            nominal_batch_size=int(self.yolo_hyp.nbs),
        )
        result = self.recipe.optimizer.build(inner_model, self.yolo_hyp, context)
        self._optimizer_build_result = result
        self._base_accumulate = result.accumulate
        if hasattr(wrapped_model, "_apply_args"):
            wrapped_model._apply_args(self.yolo_hyp)
            wrapped_model.model.args = wrapped_model.hyp
        self.logger.info(
            "YOLO26 recipe {} optimizer={} lr={} momentum={} "
            "global_batch={} accumulate={} weight_decay={}",
            self.recipe.version,
            result.name,
            result.learning_rate,
            result.momentum,
            self._global_batch_size,
            result.accumulate,
            result.scaled_weight_decay,
        )
        return result.optimizer

    def build_scheduler(self, optimizer):
        result = self.recipe.scheduler.build(
            optimizer, self.yolo_hyp, self.args.num_epoches
        )
        self.yolo_lr_lambda = result.lr_lambda
        result.scheduler.last_epoch = -1
        return result.scheduler

    def _resolve_amp(self):
        wrapped_model = de_parallel(self.model)
        inner_model = getattr(wrapped_model, "model", wrapped_model)
        requested = bool(self.args.enable_amp)
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            enabled = (
                self.recipe.precision.resolve_amp(
                    inner_model, self.device, requested
                )
                if rank == 0
                else False
            )
            value = torch.tensor(
                int(enabled), device=self.device, dtype=torch.int32
            )
            torch.distributed.broadcast(value, src=0)
            return bool(value.item())
        return self.recipe.precision.resolve_amp(inner_model, self.device, requested)

    def _setup_optimization_components(self):
        optimizer = self.build_optimizer(self.model)
        scheduler = self.build_scheduler(optimizer)
        amp_enabled = self._resolve_amp()
        self.args.enable_amp = amp_enabled
        self.yolo_hyp.amp = amp_enabled
        scaler = self.recipe.precision.build_scaler(amp_enabled)
        return optimizer, scheduler, scaler

    def _setup_ema(self):
        self.logger.info(
            "YOLO26 8.4.7 EMA enabled - decay: {}, tau: {}",
            self.args.ema_decay,
            self.args.ema_tau,
        )
        return self.recipe.ema.build(
            self.model, decay=self.args.ema_decay, tau=self.args.ema_tau
        )

    def _setup_output_directories(self):
        HQTrainer._setup_output_directories(self)
        if not self.is_master():
            return
        optimizer_groups = [
            {
                "index": index,
                "name": group.get("param_group"),
                "learning_rate": group.get("lr"),
                "weight_decay": group.get("weight_decay", 0.0),
                "use_muon": group.get("use_muon", False),
                "parameter_count": len(group["params"]),
            }
            for index, group in enumerate(self.optimizer.param_groups)
        ]
        result = self._optimizer_build_result
        manifest = {
            "recipe": "ultralytics_v847",
            "version": self.recipe.version,
            "model_source": str(self.args.model_argument.get("model")),
            "class_id_map": self.yolo_class_id_map,
            "class_names": self.yolo_class_id2names,
            "global_batch_size": self._global_batch_size,
            "optimizer": {
                "name": result.name,
                "learning_rate": result.learning_rate,
                "momentum": result.momentum,
                "iterations": result.iterations,
                "accumulate": result.accumulate,
                "scaled_weight_decay": result.scaled_weight_decay,
                "groups": optimizer_groups,
            },
            "config": vars(self.yolo_hyp),
        }
        manifest_path = os.path.join(
            self.args.output_path, "yolo26_recipe_manifest.json"
        )
        os.makedirs(self.args.output_path, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2, default=str)

    def _apply_warmup(self, iteration, epoch, batches_per_epoch):
        return self.recipe.scheduler.apply_warmup(
            self.optimizer,
            self.yolo_hyp,
            self.yolo_lr_lambda,
            iteration,
            epoch,
            batches_per_epoch,
            self._base_accumulate,
            self._global_batch_size,
        )

    def compute_loss(self, model, batch_data, forward_result):
        return self.recipe.loss.compute(model, batch_data, forward_result)

    def postprocess(self, model, batch_data, forward_result):
        return self.recipe.postprocess.process(model, batch_data, forward_result)

    def update_model_epoch(self, model):
        self.recipe.loss.on_epoch_end(model)

    def train_epoch(self, epoch):
        self.model.train()
        if isinstance(self.model, DDP) and self.sampler_train is not None:
            self.sampler_train.set_epoch(epoch)

        train_losses = []
        train_info = {}
        batches_per_epoch = len(self.dataloader_train)
        progress = self._create_progress_bar(
            self.dataloader_train,
            f"Train Epoch[{epoch}/{self.args.num_epoches - 1}]",
        )
        for batch_index, batch_data in enumerate(progress):
            iteration = batch_index + batches_per_epoch * epoch
            accumulate = self._apply_warmup(
                iteration, epoch, batches_per_epoch
            )
            loss, info = self.train_step(
                self.model, batch_data, self.optimizer, self.scaler, self.device
            )
            progress.set_postfix(**info)
            train_losses.append(loss.item())
            train_info = add_stats(train_info, info)
            if iteration - self._last_opt_step >= accumulate:
                self.optimizer_step(self.optimizer, self.scaler, self.model)
                self._last_opt_step = iteration
        return train_losses, train_info

    def _close_mosaic_if_needed(self, epoch):
        close_mosaic = int(self.yolo_hyp.close_mosaic)
        close_epoch = int(self.args.num_epoches) - close_mosaic
        if (
            not self._mosaic_closed
            and close_mosaic > 0
            and close_epoch >= 0
            and epoch == close_epoch
        ):
            self.logger.info("Closing YOLO26 mosaic/mix augmentations at epoch {}", epoch)
            self.dataset_train.close_mosaic(copy(self.yolo_hyp))
            if hasattr(self.dataloader_train, "reset"):
                self.dataloader_train.reset()
            self._mosaic_closed = True

    def run(self):
        self.setup_training_environment()
        self.logger.info(
            "Start YOLO26 training with local Ultralytics {} recipe...",
            self.recipe.version,
        )
        self.optimizer.zero_grad()
        start_time = time.time()

        for epoch in range(int(self.args.num_epoches)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
            self._close_mosaic_if_needed(epoch)
            epoch_start_time = time.time()

            train_losses, train_info = self.train_epoch(epoch)
            self.update_model_epoch(self.model)
            train_time = time.time() - epoch_start_time

            val_start_time = time.time()
            val_losses, val_info, stat = self.valid_epoch(epoch)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            val_time = time.time() - val_start_time
            epoch_time = time.time() - epoch_start_time

            train_info_avg = {
                key: value / max(len(self.dataloader_train), 1)
                for key, value in train_info.items()
            }
            summary = self._create_epoch_summary(
                epoch,
                train_losses,
                val_losses,
                val_info,
                self._format_time(train_time),
                self._format_time(val_time),
                self._format_time(epoch_time),
                stat,
            )
            self._log_epoch_summary(summary)

            metric = stat.get("mAP", 0.0)
            if self.is_master():
                self.save_epoch_result(
                    epoch,
                    stat,
                    self.args.output_path,
                    train_info=train_info_avg,
                    val_info=val_info,
                    lr_info=summary["lr_info"],
                )
                self._save_best_model(self.get_eval_model(), metric)

            self._update_training_state(epoch, train_info_avg, val_info, metric)
            self._save_checkpoint(self.get_eval_model())
            if self.args.early_stopping and self._check_early_stopping(
                summary["val_loss"], self.args.early_stopping_patience
            ):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self._log_training_summary(start_time)


def _build_arguments(
    data_path,
    output_path,
    scale,
    model_source,
    num_epoches,
    warmup_epochs,
    batch_size,
    image_size,
    lr0,
    lr_min,
    lrf,
    optimizer,
    momentum,
    weight_decay,
    nbs,
    warmup_momentum,
    warmup_bias_lr,
    cos_lr,
    seed,
    deterministic,
    box,
    cls,
    dfl,
    eval_class_names,
    devices,
    num_data_workers,
    gradient_update_interval,
    checkpoint_name,
    checkpoint_interval,
    pretrained,
    p2,
    use_ema,
    ema_decay,
    ema_tau,
    enable_amp,
    close_mosaic,
    hsv_h,
    hsv_s,
    hsv_v,
    degrees,
    translate,
    scale_gain,
    shear,
    perspective,
    flipud,
    fliplr,
    bgr,
    mosaic,
    mixup,
    cutmix,
    copy_paste,
    copy_paste_mode,
    yolo_overrides,
    augment_proba,
    augment_split_size,
    augment_split_proba,
    augment_foreground_path,
    augment_foreground_proba,
    augment_force_resize,
):
    return YoloTrainerArguments(
        data_path=data_path,
        num_epoches=num_epoches,
        warmup_epochs=warmup_epochs,
        num_data_workers=num_data_workers,
        lr0=lr0,
        lr_min=lr_min,
        lrf=lrf,
        optimizer=optimizer,
        momentum=momentum,
        weight_decay=weight_decay,
        nbs=nbs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,
        cos_lr=cos_lr,
        seed=seed,
        deterministic=deterministic,
        box=box,
        cls=cls,
        dfl=dfl,
        batch_size=batch_size,
        devices=devices or [0],
        output_path=output_path,
        checkpoint_path=output_path,
        checkpoint_interval=checkpoint_interval,
        image_size=image_size,
        model_argument={
            "model": model_source,
            "scale": scale,
            "pretrained": pretrained,
            "p2": p2,
        },
        eval_class_names=eval_class_names,
        gradient_update_interval=gradient_update_interval,
        checkpoint_name=checkpoint_name,
        find_unused_parameters=True,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_tau=ema_tau,
        enable_amp=enable_amp,
        max_grad_norm=10.0,
        close_mosaic=close_mosaic,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale_gain=scale_gain,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        bgr=bgr,
        mosaic=mosaic,
        mixup=mixup,
        cutmix=cutmix,
        copy_paste=copy_paste,
        copy_paste_mode=copy_paste_mode,
        yolo_overrides=yolo_overrides or {},
        augment_proba=augment_proba,
        augment_split_size=augment_split_size,
        augment_split_proba=augment_split_proba,
        augment_foreground_path=augment_foreground_path,
        augment_foreground_proba=augment_foreground_proba,
        augment_force_resize=augment_force_resize,
    )


def run(
    data_path,
    output_path="output",
    scale="n",
    load_checkpoint=None,
    num_epoches=100,
    warmup_epochs=3.0,
    batch_size=4,
    image_size=1024,
    lr0=0.01,
    lr_min=1e-4,
    lrf=0.01,
    optimizer="auto",
    momentum=0.937,
    weight_decay=0.0005,
    nbs=64,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=False,
    seed=0,
    deterministic=True,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    eval_class_names=None,
    devices=None,
    num_data_workers=8,
    gradient_update_interval=1,
    checkpoint_name="ckpt.pth",
    checkpoint_interval=1,
    scratch=False,
    p2=False,
    use_ema=True,
    ema_decay=0.9999,
    ema_tau=2000.0,
    enable_amp=True,
    close_mosaic=10,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale_gain=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    bgr=0.0,
    mosaic=1.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    copy_paste_mode="flip",
    yolo_overrides=None,
    augment_proba=0.3,
    augment_split_size=-1,
    augment_split_proba=0.5,
    augment_foreground_path="",
    augment_foreground_proba=0.8,
    augment_force_resize=False,
):
    scale = str(scale).lower()
    if scale not in YOLO26_SCALES:
        raise ValueError(f"Unsupported YOLO26 scale: {scale}. Use {YOLO26_SCALES}.")
    model_source = load_checkpoint or build_yolo26_model_name(
        scale=scale, pretrained=not scratch, p2=p2
    )
    args = _build_arguments(
        data_path,
        output_path,
        scale,
        model_source,
        num_epoches,
        warmup_epochs,
        batch_size,
        image_size,
        lr0,
        lr_min,
        lrf,
        optimizer,
        momentum,
        weight_decay,
        nbs,
        warmup_momentum,
        warmup_bias_lr,
        cos_lr,
        seed,
        deterministic,
        box,
        cls,
        dfl,
        eval_class_names,
        devices,
        num_data_workers,
        gradient_update_interval,
        checkpoint_name,
        checkpoint_interval,
        not scratch,
        p2,
        use_ema,
        ema_decay,
        ema_tau,
        enable_amp,
        close_mosaic,
        hsv_h,
        hsv_s,
        hsv_v,
        degrees,
        translate,
        scale_gain,
        shear,
        perspective,
        flipud,
        fliplr,
        bgr,
        mosaic,
        mixup,
        cutmix,
        copy_paste,
        copy_paste_mode,
        yolo_overrides,
        augment_proba,
        augment_split_size,
        augment_split_proba,
        augment_foreground_path,
        augment_foreground_proba,
        augment_force_resize,
    )
    trainer = Yolo26Trainer(args)
    trainer.run()
    return trainer


def evaluate(
    data_path,
    model,
    output_path="output/yolo26_eval",
    scale="n",
    batch_size=4,
    image_size=1024,
    num_data_workers=8,
    device=0,
    eval_class_names=None,
    p2=False,
    yolo_overrides=None,
):
    scale = str(scale).lower()
    if scale not in YOLO26_SCALES:
        raise ValueError(f"Unsupported YOLO26 scale: {scale}. Use {YOLO26_SCALES}.")
    args = _build_arguments(
        data_path,
        output_path,
        scale,
        model,
        1,
        0.0,
        batch_size,
        image_size,
        0.01,
        1e-4,
        0.01,
        "auto",
        0.937,
        0.0005,
        64,
        0.8,
        0.1,
        False,
        0,
        True,
        7.5,
        0.5,
        1.5,
        eval_class_names,
        [device],
        num_data_workers,
        1,
        "eval.pth",
        -1,
        True,
        p2,
        True,
        0.9999,
        2000.0,
        True,
        0,
        0.015,
        0.7,
        0.4,
        0.0,
        0.1,
        0.5,
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        "flip",
        yolo_overrides,
        0.3,
        -1,
        0.5,
        "",
        0.8,
        False,
    )
    trainer = Yolo26Trainer(args)
    trainer.setup_training_environment()
    _, _, stat = trainer.valid_epoch(0)
    return trainer, stat


MyTrainer = Yolo26Trainer


__all__ = [
    "MyTrainer",
    "ULTRALYTICS_V847_VERSION",
    "Yolo26Trainer",
    "evaluate",
    "run",
]
