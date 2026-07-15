"""Framework-owned copy of the Ultralytics 8.4.7 MuSGD grouping recipe."""

import math
import re

from torch import nn

from hq_det.training.interfaces import OptimizerBuildResult, OptimizerContext, OptimizerStrategy

from .musgd import MuSGD


_HIGH_LR_PATTERN = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg|flow_model")


def _unwrap_model(model):
    while True:
        if hasattr(model, "_orig_mod") and isinstance(model._orig_mod, nn.Module):
            model = model._orig_mod
        elif hasattr(model, "module") and isinstance(model.module, nn.Module):
            model = model.module
        else:
            return model


class UltralyticsV847Optimizer(OptimizerStrategy):
    """Reproduce 8.4.7 auto selection, MuSGD grouping, and decay scaling."""

    def build(self, model, hyp, context: OptimizerContext) -> OptimizerBuildResult:
        global_batch = max(int(context.global_batch_size), 1)
        nominal_batch = max(int(context.nominal_batch_size), 1)
        accumulate = max(round(nominal_batch / global_batch), 1)
        scaled_decay = (
            float(hyp.weight_decay) * global_batch * accumulate / nominal_batch
        )
        iterations = (
            math.ceil(context.dataset_size / max(global_batch, nominal_batch))
            * int(context.epochs)
        )

        requested_name = str(hyp.optimizer)
        if requested_name.lower() == "auto":
            lr_fit = round(0.002 * 5 / (4 + int(context.num_classes)), 6)
            name = "MuSGD"
            lr = 0.01 if iterations > 10000 else lr_fit
            momentum = 0.9
            hyp.warmup_bias_lr = 0.0
        elif requested_name.lower() == "musgd":
            name = "MuSGD"
            lr = float(hyp.lr0)
            momentum = float(hyp.momentum)
        else:
            raise ValueError(
                "The Ultralytics 8.4.7 YOLO26 recipe only supports auto or MuSGD."
            )

        groups = [{}, {}, {}, {}]
        norm_layers = tuple(value for key, value in nn.__dict__.items() if "Norm" in key)
        for module_name, module in _unwrap_model(model).named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                fullname = (
                    f"{module_name}.{parameter_name}" if module_name else parameter_name
                )
                if parameter.ndim >= 2:
                    groups[3][fullname] = parameter
                elif "bias" in fullname:
                    groups[2][fullname] = parameter
                elif isinstance(module, norm_layers) or "logit_scale" in fullname:
                    groups[1][fullname] = parameter
                else:
                    groups[0][fullname] = parameter

        optimizer_args = dict(lr=lr, momentum=momentum, nesterov=True)
        groups[2] = {
            "params": groups[2],
            **optimizer_args,
            "param_group": "bias",
        }
        groups[0] = {
            "params": groups[0],
            **optimizer_args,
            "weight_decay": scaled_decay,
            "param_group": "weight",
        }
        groups[1] = {
            "params": groups[1],
            **optimizer_args,
            "weight_decay": 0.0,
            "param_group": "bn",
        }
        groups[3] = {
            "params": groups[3],
            **optimizer_args,
            "weight_decay": scaled_decay,
            "use_muon": True,
            "param_group": "muon",
        }

        split_groups = []
        for group in groups:
            named_parameters = group.pop("params")
            high_lr = [
                value
                for key, value in named_parameters.items()
                if _HIGH_LR_PATTERN.search(key)
            ]
            regular_lr = [
                value
                for key, value in named_parameters.items()
                if not _HIGH_LR_PATTERN.search(key)
            ]
            split_groups.extend(
                [
                    {"params": high_lr, **group, "lr": lr * 3},
                    {"params": regular_lr, **group},
                ]
            )

        muon, sgd = (0.1, 1.0) if iterations > 10000 else (0.5, 0.5)
        optimizer = MuSGD(params=split_groups, muon=muon, sgd=sgd)
        return OptimizerBuildResult(
            optimizer=optimizer,
            accumulate=accumulate,
            scaled_weight_decay=scaled_decay,
            iterations=iterations,
            name=name,
            learning_rate=lr,
            momentum=momentum,
        )


__all__ = ["UltralyticsV847Optimizer"]
