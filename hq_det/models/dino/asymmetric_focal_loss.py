# -----------------------------------------------------------------------------
# Asymmetric Sigmoid Focal Loss (CUDA).
#
# This file is a byte-for-byte port of the combination of
#   mmcv/ops/focal_loss.py                (SigmoidFocalLossFunction)
#   mmdet/models/losses/focal_loss.py     (sigmoid_focal_loss, FocalLoss)
# The ONLY intentional modifications are:
#   1. Identifier renames to the "asymmetric_*" namespace.
#   2. The single `gamma` argument is replaced by two: `gamma_pos` (applied to
#      the positive term) and `gamma_neg` (applied to the negative term).
# Everything else (input validation, reduction semantics, weight handling,
# dispatch logic, `weight_reduce_loss`, tensor contiguity) is preserved exactly.
# -----------------------------------------------------------------------------
import fcntl
import glob
import os
import subprocess
import sys
import time
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from mmengine import MODELS
from mmdet.models.losses.utils import weight_reduce_loss


_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc')
_SOURCE_FILES = [
    'asymmetric_focal_loss.cpp',
    'asymmetric_focal_loss_cuda.cu',
    'asymmetric_sigmoid_focal_loss_cuda_kernel.cuh',
    'setup.py',
]


def _so_up_to_date() -> bool:
    """Return True iff a compiled .so exists for the current interpreter and
    is newer than every source file."""
    # Match python-version-specific suffix (e.g. cpython-312-x86_64-linux-gnu.so)
    so_files = glob.glob(
        os.path.join(_CSRC_DIR, 'asymmetric_focal_loss_cuda*.so'))
    if not so_files:
        return False
    src_paths = [os.path.join(_CSRC_DIR, f) for f in _SOURCE_FILES]
    src_paths = [p for p in src_paths if os.path.exists(p)]
    if not src_paths:
        return bool(so_files)
    src_mtime = max(os.path.getmtime(p) for p in src_paths)
    so_mtime = max(os.path.getmtime(p) for p in so_files)
    return so_mtime >= src_mtime


def _compile_extension() -> None:
    """Invoke `python setup.py build_ext --inplace` in csrc/."""
    print(f'[asymmetric_focal_loss] Compiling CUDA extension in {_CSRC_DIR} ...',
          flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, 'setup.py', 'build_ext', '--inplace'],
        cwd=_CSRC_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(
            'Failed to compile asymmetric_focal_loss CUDA extension. '
            'See build output above.')
    print(f'[asymmetric_focal_loss] Build complete in {time.time() - t0:.1f}s.',
          flush=True)


def _ensure_compiled() -> None:
    """First-time compile guard. Safe under DDP: uses flock so only one
    process actually builds while others wait."""
    if _so_up_to_date():
        return
    os.makedirs(_CSRC_DIR, exist_ok=True)
    lock_path = os.path.join(_CSRC_DIR, '.build.lock')
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        # Double-checked locking: another process may have built it already.
        if _so_up_to_date():
            return
        _compile_extension()


_ensure_compiled()

try:
    if _CSRC_DIR not in sys.path:
        sys.path.insert(0, _CSRC_DIR)
    import asymmetric_focal_loss_cuda as ext_module
    CUDA_AVAILABLE = True
except ImportError as e:
    CUDA_AVAILABLE = False
    _import_error = str(e)


# -----------------------------------------------------------------------------
# Layer 1: CUDA autograd Function
# Mirrors mmcv.ops.focal_loss.SigmoidFocalLossFunction byte-for-byte.
# -----------------------------------------------------------------------------
class AsymmetricSigmoidFocalLossFunction(Function):

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                target: Union[torch.LongTensor, torch.cuda.LongTensor],
                gamma_pos: float = 2.0,
                gamma_neg: float = 2.0,
                alpha: float = 0.25,
                weight: Optional[torch.Tensor] = None,
                reduction: str = 'mean') -> torch.Tensor:

        assert target.dtype == torch.long
        assert input.dim() == 2
        assert target.dim() == 1
        assert input.size(0) == target.size(0)
        if weight is None:
            weight = input.new_empty(0)
        else:
            assert weight.dim() == 1
            assert input.size(1) == weight.size(0)
        ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
        assert reduction in ctx.reduction_dict.keys()

        ctx.gamma_pos = float(gamma_pos)
        ctx.gamma_neg = float(gamma_neg)
        ctx.alpha = float(alpha)
        ctx.reduction = ctx.reduction_dict[reduction]

        output = input.new_zeros(input.size())

        ext_module.asymmetric_sigmoid_focal_loss_forward(
            input, target, weight, output,
            gamma_pos=ctx.gamma_pos, gamma_neg=ctx.gamma_neg, alpha=ctx.alpha)
        if ctx.reduction == ctx.reduction_dict['mean']:
            output = output.sum() / input.size(0)
        elif ctx.reduction == ctx.reduction_dict['sum']:
            output = output.sum()
        ctx.save_for_backward(input, target, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        input, target, weight = ctx.saved_tensors

        grad_input = input.new_zeros(input.size())

        ext_module.asymmetric_sigmoid_focal_loss_backward(
            input, target, weight, grad_input,
            gamma_pos=ctx.gamma_pos, gamma_neg=ctx.gamma_neg, alpha=ctx.alpha)

        grad_input *= grad_output
        if ctx.reduction == ctx.reduction_dict['mean']:
            grad_input /= input.size(0)
        return grad_input, None, None, None, None, None, None


_asymmetric_sigmoid_focal_loss = AsymmetricSigmoidFocalLossFunction.apply


# -----------------------------------------------------------------------------
# Layer 2: mmdet-style top-level wrapper.
# Mirrors mmdet.models.losses.focal_loss.sigmoid_focal_loss byte-for-byte.
# -----------------------------------------------------------------------------
def asymmetric_sigmoid_focal_loss(pred,
                                  target,
                                  weight=None,
                                  gamma_pos=2.0,
                                  gamma_neg=2.0,
                                  alpha=0.25,
                                  reduction='mean',
                                  avg_factor=None):
    # Function.apply does not accept kwargs, so positional only.
    loss = _asymmetric_sigmoid_focal_loss(pred.contiguous(),
                                          target.contiguous(),
                                          gamma_pos, gamma_neg, alpha,
                                          None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # (num_priors,) -> (num_priors, 1)
                weight = weight.view(-1, 1)
            else:
                # (num_priors * num_class,) -> (num_priors, num_class)
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


# -----------------------------------------------------------------------------
# Layer 3: nn.Module wrapper.
# Mirrors mmdet.models.losses.focal_loss.FocalLoss byte-for-byte.
# -----------------------------------------------------------------------------
@MODELS.register_module()
class AsymmetricFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma_pos=2.0,
                 gamma_neg=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        """Asymmetric Focal Loss; same API as mmdet.FocalLoss but with
        two focusing parameters: `gamma_pos` (positive term) and `gamma_neg`
        (negative term). When `gamma_pos == gamma_neg`, numerical output and
        gradients are identical to mmdet.FocalLoss.
        """
        super().__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss is supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                f'asymmetric_focal_loss_cuda is not compiled or cannot be '
                f'loaded. Import error: {_import_error}. Please compile it '
                f'first: cd csrc && python setup.py build_ext --inplace')
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                raise NotImplementedError(
                    'AsymmetricFocalLoss does not support `activated=True`.')
            # mmdet FocalLoss dispatch: when target is 1D and pred is CUDA, go
            # through the CUDA kernel. The one-hot / CPU fallback branches are
            # intentionally omitted because we only provide a CUDA kernel.
            assert pred.dim() != target.dim(), (
                'AsymmetricFocalLoss expects 1D integer target '
                '(class indices). One-hot targets are not supported.')
            assert torch.cuda.is_available() and pred.is_cuda, (
                'AsymmetricFocalLoss only supports CUDA tensors.')
            calculate_loss_func = asymmetric_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma_pos=self.gamma_pos,
                gamma_neg=self.gamma_neg,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
