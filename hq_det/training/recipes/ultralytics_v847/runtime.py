"""Reproducibility and model preparation from the 8.4.7 trainer."""

import os
import random

import numpy as np
import torch

from hq_det.training.interfaces import RuntimeStrategy


class UltralyticsV847Runtime(RuntimeStrategy):
    def initialize_seed(self, seed: int, deterministic: bool, rank: int) -> None:
        seed = int(seed) + 1 + int(rank)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["PYTHONHASHSEED"] = str(seed)
        else:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            os.environ.pop("PYTHONHASHSEED", None)

    def prepare_model(self, model, freeze=None) -> None:
        freeze_list = (
            freeze
            if isinstance(freeze, list)
            else range(freeze)
            if isinstance(freeze, int)
            else []
        )
        freeze_names = [f"model.{index}." for index in freeze_list] + [".dfl"]
        for name, parameter in model.named_parameters():
            if any(frozen_name in name for frozen_name in freeze_names):
                parameter.requires_grad = False
            elif not parameter.requires_grad and parameter.dtype.is_floating_point:
                parameter.requires_grad = True


__all__ = ["UltralyticsV847Runtime"]
