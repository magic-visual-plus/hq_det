"""Ultralytics 8.4.7 epoch scheduler and batch warmup."""

import math

import numpy as np
import torch

from hq_det.training.interfaces import SchedulerBuildResult, SchedulerStrategy


class UltralyticsV847Scheduler(SchedulerStrategy):
    def build(self, optimizer, hyp, epochs: int) -> SchedulerBuildResult:
        epochs = max(int(epochs), 1)
        final_factor = float(hyp.lrf)
        if bool(hyp.cos_lr):
            lr_lambda = lambda epoch: max(
                (1 - math.cos(epoch * math.pi / epochs)) / 2, 0
            ) * (final_factor - 1) + 1
        else:
            lr_lambda = lambda epoch: max(1 - epoch / epochs, 0) * (
                1 - final_factor
            ) + final_factor
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda
        )
        return SchedulerBuildResult(scheduler=scheduler, lr_lambda=lr_lambda)

    def apply_warmup(
        self,
        optimizer,
        hyp,
        lr_lambda,
        iteration: int,
        epoch: int,
        batches_per_epoch: int,
        base_accumulate: int,
        global_batch_size: int,
    ) -> int:
        warmup_iterations = (
            max(round(float(hyp.warmup_epochs) * batches_per_epoch), 100)
            if float(hyp.warmup_epochs) > 0
            else -1
        )
        accumulate = int(base_accumulate)
        if iteration <= warmup_iterations:
            bounds = [0, warmup_iterations]
            accumulate = max(
                1,
                int(
                    np.interp(
                        iteration,
                        bounds,
                        [1, float(hyp.nbs) / max(int(global_batch_size), 1)],
                    ).round()
                ),
            )
            for group in optimizer.param_groups:
                start_lr = (
                    float(hyp.warmup_bias_lr)
                    if group.get("param_group") == "bias"
                    else 0.0
                )
                group["lr"] = float(
                    np.interp(
                        iteration,
                        bounds,
                        [start_lr, group["initial_lr"] * lr_lambda(epoch)],
                    )
                )
                if "momentum" in group:
                    group["momentum"] = float(
                        np.interp(
                            iteration,
                            bounds,
                            [float(hyp.warmup_momentum), float(hyp.momentum)],
                        )
                    )
        return accumulate


__all__ = ["UltralyticsV847Scheduler"]
