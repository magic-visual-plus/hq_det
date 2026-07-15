import math
import os
import time
from copy import copy

import numpy as np
import torch
from pydantic import Field
from torch.nn.parallel import DistributedDataParallel as DDP
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import ModelEMA as UltralyticsModelEMA, init_seeds

from hq_det import torch_utils
from hq_det.common import HQTrainerArguments
from hq_det.models import yolo
from hq_det.print_utils import print_augmentation_steps, print_dataset_summary
from hq_det.trainer import HQTrainer, add_stats, de_parallel
from hq_det.yolo_dataset import HQYOLODataset


class YoloTrainerArguments(HQTrainerArguments):
    """YOLO-only settings kept separate from the shared trainer configuration."""

    warmup_epochs: float = 3.0
    max_grad_norm: float = 10.0
    seed: int = 0
    deterministic: bool = True

    optimizer: str = "auto"
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    nbs: int = 64
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    cos_lr: bool = False

    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5

    close_mosaic: int = 10
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale_gain: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    bgr: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    cutmix: float = 0.0
    copy_paste: float = 0.0
    copy_paste_mode: str = "flip"

    # Forward-compatible escape hatch for YOLO26-specific arguments added upstream.
    yolo_overrides: dict = Field(default_factory=dict)


class YoloTrainer(HQTrainer):
    def __init__(self, args: HQTrainerArguments):
        super().__init__(args)
        self.yolo_hyp = self._build_yolo_hyp()
        self.original_class_id2names = {}
        self.yolo_class_id_map = {}
        self.yolo_class_id2names = {}
        self._base_accumulate = 1
        self._last_opt_step = -1
        self._mosaic_closed = False

    def get_total_epochs(self):
        return int(self.args.num_epoches)

    def _arg(self, name, default):
        return getattr(self.args, name, default)

    def _build_yolo_hyp(self):
        hyp = copy(DEFAULT_CFG)
        values = {
            "epochs": int(self.args.num_epoches),
            "imgsz": int(self.args.image_size),
            "batch": int(self.args.batch_size),
            "amp": bool(self.args.enable_amp),
            "seed": int(self._arg("seed", 0)),
            "deterministic": bool(self._arg("deterministic", True)),
            "optimizer": self._arg("optimizer", "auto"),
            "lr0": float(self.args.lr0),
            "lrf": float(self._arg("lrf", self.args.lr_min / self.args.lr0)),
            "momentum": float(self._arg("momentum", 0.937)),
            "weight_decay": float(self._arg("weight_decay", 0.0005)),
            "nbs": int(self._arg("nbs", 64)),
            "warmup_epochs": float(self.args.warmup_epochs),
            "warmup_momentum": float(self._arg("warmup_momentum", 0.8)),
            "warmup_bias_lr": float(self._arg("warmup_bias_lr", 0.1)),
            "cos_lr": bool(self._arg("cos_lr", False)),
            "box": float(self._arg("box", 7.5)),
            "cls": float(self._arg("cls", 0.5)),
            "dfl": float(self._arg("dfl", 1.5)),
            "close_mosaic": int(self._arg("close_mosaic", 10)),
            "hsv_h": float(self._arg("hsv_h", 0.015)),
            "hsv_s": float(self._arg("hsv_s", 0.7)),
            "hsv_v": float(self._arg("hsv_v", 0.4)),
            "degrees": float(self._arg("degrees", 0.0)),
            "translate": float(self._arg("translate", 0.1)),
            "scale": float(self._arg("scale_gain", 0.5)),
            "shear": float(self._arg("shear", 0.0)),
            "perspective": float(self._arg("perspective", 0.0)),
            "flipud": float(self._arg("flipud", 0.0)),
            "fliplr": float(self._arg("fliplr", 0.5)),
            "bgr": float(self._arg("bgr", 0.0)),
            "mosaic": float(self._arg("mosaic", 1.0)),
            "mixup": float(self._arg("mixup", 0.0)),
            "cutmix": float(self._arg("cutmix", 0.0)),
            "copy_paste": float(self._arg("copy_paste", 0.0)),
            "copy_paste_mode": self._arg("copy_paste_mode", "flip"),
        }
        values.update(self._arg("yolo_overrides", {}) or {})
        for name, value in values.items():
            setattr(hyp, name, value)
        return hyp

    def setup_training_environment(self):
        rank = int(os.environ.get("RANK", "-1"))
        init_seeds(
            int(getattr(self.yolo_hyp, "seed", 0)) + 1 + rank,
            deterministic=bool(getattr(self.yolo_hyp, "deterministic", True)),
        )
        super().setup_training_environment()

    @staticmethod
    def _make_contiguous_mapping(class_id2names):
        original = {int(k): v for k, v in class_id2names.items()}
        ordered_ids = sorted(original)
        names = [original[class_id] for class_id in ordered_ids]
        if len(names) != len(set(names)):
            raise ValueError("YOLO class mapping requires unique class names.")
        id_map = {class_id: index for index, class_id in enumerate(ordered_ids)}
        contiguous_names = {index: original[class_id] for class_id, index in id_map.items()}
        return original, id_map, contiguous_names

    def _setup_datasets_and_transforms(self):
        provider = self.build_dataset_provider()
        source_train = provider.build_train_dataset(None)
        source_val = provider.build_valid_dataset(None)

        (
            self.original_class_id2names,
            self.yolo_class_id_map,
            self.yolo_class_id2names,
        ) = self._make_contiguous_mapping(source_train.class_id2names)

        name_to_yolo_id = {
            name: class_id for class_id, name in self.yolo_class_id2names.items()
        }
        val_class_id_map = {}
        for original_id, name in source_val.class_id2names.items():
            if name not in name_to_yolo_id:
                raise ValueError(f"Validation class '{name}' is missing from training classes.")
            val_class_id_map[int(original_id)] = name_to_yolo_id[name]

        self.args.class_id2names = dict(self.yolo_class_id2names)
        train_dataset = HQYOLODataset(
            source_train,
            image_size=self.args.image_size,
            hyp=self.yolo_hyp,
            class_id_map=self.yolo_class_id_map,
            class_id2names=self.yolo_class_id2names,
            augment=True,
            batch_size=self.args.batch_size,
        )
        val_dataset = HQYOLODataset(
            source_val,
            image_size=self.args.image_size,
            hyp=self.yolo_hyp,
            class_id_map=val_class_id_map,
            class_id2names=self.yolo_class_id2names,
            augment=False,
            batch_size=self.args.batch_size,
        )

        self.logger.info(f"YOLO category id mapping: {self.yolo_class_id_map}")
        if self.HQ_DEBUG:
            print_dataset_summary(train_dataset, val_dataset)
            print_augmentation_steps(train_dataset.transforms, val_dataset.transforms)
        return train_dataset, val_dataset

    def _setup_class_configuration(self):
        if self.args.eval_class_names is None:
            return list(self.yolo_class_id2names)

        name_to_id = {name: class_id for class_id, name in self.yolo_class_id2names.items()}
        eval_ids = [
            name_to_id[name]
            for name in self.args.eval_class_names
            if name in name_to_id
        ]
        return eval_ids or list(self.yolo_class_id2names)

    def build_model(self):
        model_source = (
            self.args.model_argument.get("model")
            or self.args.model_argument.get("model_path")
            or self.args.model_argument.get("weights")
        )
        return yolo.HQYOLO(
            model=model_source,
            class_id2names=self.args.class_id2names,
            epochs=self.args.num_epoches,
            image_size=self.args.image_size,
            args=self.yolo_hyp,
            **{
                key: value
                for key, value in self.args.model_argument.items()
                if key not in {"model", "model_path", "weights"}
            },
        )

    def collate_fn(self, batch):
        max_h = math.ceil(max(item["img"].shape[1] for item in batch) / 32) * 32
        max_w = math.ceil(max(item["img"].shape[2] for item in batch) / 32) * 32

        for item in batch:
            item["img"], boxes = torch_utils.pad_image(
                item["img"], item["bboxes_cxcywh_norm"], (max_h, max_w)
            )
            item["bboxes"] = boxes
            item["bboxes_cxcywh_norm"] = boxes

        new_batch = {}
        batch = [dict(sorted(item.items())) for item in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(item.values()) for item in batch]))

        for index, key in enumerate(keys):
            value = values[index]
            if key in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            if key in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"} or key.startswith("bboxes_"):
                value = torch.cat(value, 0)
            new_batch[key] = value

        batch_indices = list(new_batch["batch_idx"])
        for index in range(len(batch_indices)):
            batch_indices[index] += index
        new_batch["batch_idx"] = torch.cat(batch_indices, 0)
        new_batch["bboxes"] = new_batch["bboxes_cxcywh_norm"]

        classes = new_batch["cls"].long().reshape(-1)
        if classes.numel():
            min_id = int(classes.min().item())
            max_id = int(classes.max().item())
            num_classes = len(self.yolo_class_id2names)
            if min_id < 0 or max_id >= num_classes:
                raise ValueError(
                    f"YOLO labels must be in [0, {num_classes - 1}], got [{min_id}, {max_id}]"
                )
        new_batch["cls"] = classes
        return new_batch

    def build_optimizer(self, model):
        inner_model = de_parallel(model)
        inner_model = getattr(inner_model, "model", inner_model)

        world_size = max(len(self.args.devices), 1)
        global_batch = max(int(self.args.batch_size) * world_size, 1)
        nominal_batch = max(int(getattr(self.yolo_hyp, "nbs", 64)), 1)
        requested_accumulate = max(int(self.args.gradient_update_interval), 1)
        self._base_accumulate = max(
            round(nominal_batch / global_batch), requested_accumulate, 1
        )
        scaled_decay = (
            float(getattr(self.yolo_hyp, "weight_decay", 0.0005))
            * global_batch
            * self._base_accumulate
            / nominal_batch
        )
        iterations = (
            math.ceil(len(self.dataset_train) / max(global_batch, nominal_batch))
            * int(self.args.num_epoches)
        )

        builder = object.__new__(BaseTrainer)
        builder.args = self.yolo_hyp
        builder.data = {"nc": len(self.yolo_class_id2names)}
        builder.model = inner_model
        optimizer = BaseTrainer.build_optimizer(
            builder,
            model=inner_model,
            name=getattr(self.yolo_hyp, "optimizer", "auto"),
            lr=float(getattr(self.yolo_hyp, "lr0", self.args.lr0)),
            momentum=float(getattr(self.yolo_hyp, "momentum", 0.937)),
            decay=scaled_decay,
            iterations=iterations,
        )
        wrapped_model = de_parallel(model)
        if hasattr(wrapped_model, "_apply_args"):
            wrapped_model._apply_args(self.yolo_hyp)
            wrapped_model.model.args = wrapped_model.hyp
        self.logger.info(
            "YOLO optimizer alignment - global batch: {}, accumulate: {}, weight decay: {:.6g}",
            global_batch,
            self._base_accumulate,
            scaled_decay,
        )
        return optimizer

    def build_scheduler(self, optimizer):
        epochs = max(int(self.args.num_epoches), 1)
        final_factor = float(getattr(self.yolo_hyp, "lrf", 0.01))
        if bool(getattr(self.yolo_hyp, "cos_lr", False)):
            self.yolo_lr_lambda = lambda epoch: (
                (1 - math.cos(epoch * math.pi / epochs)) / 2
            ) * (final_factor - 1) + 1
        else:
            self.yolo_lr_lambda = lambda epoch: max(1 - epoch / epochs, 0) * (
                1 - final_factor
            ) + final_factor
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self.yolo_lr_lambda
        )

    def _log_learning_rates(self, optimizer):
        return {
            f"lr/pg{index}": group["lr"]
            for index, group in enumerate(optimizer.param_groups)
        }

    def _setup_ema(self):
        if not self.args.use_ema:
            return None
        self.logger.info(
            f"Ultralytics EMA enabled - decay: {self.args.ema_decay}, tau: {self.args.ema_tau}"
        )
        return UltralyticsModelEMA(
            self.model,
            decay=self.args.ema_decay,
            tau=self.args.ema_tau,
        )

    def _apply_warmup(self, iteration, epoch, batches_per_epoch):
        warmup_epochs = float(getattr(self.yolo_hyp, "warmup_epochs", 0.0))
        warmup_iterations = (
            max(round(warmup_epochs * batches_per_epoch), 100)
            if warmup_epochs > 0
            else -1
        )
        accumulate = self._base_accumulate
        if iteration <= warmup_iterations:
            accumulate = max(
                1,
                int(
                    np.interp(
                        iteration,
                        [0, warmup_iterations],
                        [1, self._base_accumulate],
                    ).round()
                ),
            )
            for group in self.optimizer.param_groups:
                target_lr = group["initial_lr"] * self.yolo_lr_lambda(epoch)
                start_lr = (
                    float(getattr(self.yolo_hyp, "warmup_bias_lr", 0.1))
                    if group.get("param_group") == "bias"
                    else 0.0
                )
                group["lr"] = float(
                    np.interp(
                        iteration,
                        [0, warmup_iterations],
                        [start_lr, target_lr],
                    )
                )
                if "momentum" in group:
                    group["momentum"] = float(
                        np.interp(
                            iteration,
                            [0, warmup_iterations],
                            [
                                float(getattr(self.yolo_hyp, "warmup_momentum", 0.8)),
                                float(getattr(self.yolo_hyp, "momentum", 0.937)),
                            ],
                        )
                    )
        return accumulate

    def train_step(self, model, batch_data, optimizer, scaler, device):
        batch_data = torch_utils.batch_to_device(batch_data, device)
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.autocast(
            device_type=device_type,
            dtype=torch.float16 if device_type == "cuda" else torch.bfloat16,
            enabled=self.args.enable_amp and device_type == "cuda",
        ):
            forward_result = model(batch_data)
            loss, info = self.compute_loss(model, batch_data, forward_result)
        backward_loss = loss
        if isinstance(model, DDP) and torch.distributed.is_initialized():
            backward_loss = loss * torch.distributed.get_world_size()
        scaler.scale(backward_loss).backward()
        return loss, info

    def valid_step(self, model, batch_data, device):
        batch_data = torch_utils.batch_to_device(batch_data, device)
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.no_grad(), torch.autocast(
            device_type=device_type,
            dtype=torch.float16 if device_type == "cuda" else torch.bfloat16,
            enabled=self.args.enable_amp and device_type == "cuda",
        ):
            forward_result = model(batch_data)
            loss, info = self.compute_loss(model, batch_data, forward_result)
            preds = self.postprocess(model, batch_data, forward_result)

        for pred, image_id in zip(preds, batch_data["image_id"]):
            pred.image_id = image_id
        return loss, info, preds

    def train_epoch(self, epoch):
        self.model.train()
        if isinstance(self.model, DDP) and self.sampler_train is not None:
            self.sampler_train.set_epoch(epoch)

        train_losses = []
        train_info = {}
        batches_per_epoch = len(self.dataloader_train)
        bar_train = self._create_progress_bar(
            self.dataloader_train,
            f"Train Epoch[{epoch}/{self.args.num_epoches - 1}]",
        )

        for batch_index, batch_data in enumerate(bar_train):
            iteration = batch_index + batches_per_epoch * epoch
            accumulate = self._apply_warmup(
                iteration, epoch, batches_per_epoch
            )
            loss, info = self.train_step(
                self.model, batch_data, self.optimizer, self.scaler, self.device
            )
            bar_train.set_postfix(**info)
            train_losses.append(loss.item())
            train_info = add_stats(train_info, info)

            final_batch = (
                epoch + 1 == self.args.num_epoches
                and batch_index + 1 == batches_per_epoch
            )
            if iteration - self._last_opt_step >= accumulate or final_batch:
                self.optimizer_step(self.optimizer, self.scaler, self.model)
                self._last_opt_step = iteration

        return train_losses, train_info

    def update_model_epoch(self, model):
        super().update_model_epoch(model)
        if self.ema is not None:
            ema_model = de_parallel(self.ema.ema)
            if hasattr(ema_model, "update_epoch"):
                ema_model.update_epoch()

    def _close_mosaic_if_needed(self, epoch):
        close_mosaic = int(getattr(self.yolo_hyp, "close_mosaic", 0))
        close_epoch = int(self.args.num_epoches) - close_mosaic
        if (
            not self._mosaic_closed
            and close_mosaic > 0
            and close_epoch >= 0
            and epoch == close_epoch
        ):
            self.logger.info("Closing YOLO mosaic/mix augmentations at epoch {}", epoch)
            self.dataset_train.close_mosaic(copy(self.yolo_hyp))
            self._mosaic_closed = True

    def run(self):
        self.setup_training_environment()
        self.logger.info("Start YOLO-aligned training...")
        self.optimizer.zero_grad()
        start_time = time.time()

        for epoch in range(int(self.args.num_epoches)):
            self._close_mosaic_if_needed(epoch)
            epoch_start_time = time.time()

            train_losses, train_info = self.train_epoch(epoch)
            self.update_model_epoch(self.model)
            train_time = time.time() - epoch_start_time

            val_start_time = time.time()
            val_losses, val_info, stat = self.valid_epoch(epoch)
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

            self.scheduler.step()

        self._log_training_summary(start_time)


MyTrainer = YoloTrainer


__all__ = ["YoloTrainer", "YoloTrainerArguments", "MyTrainer"]
