# Copyright (c) OpenMMLab. All rights reserved.
"""Training-only spatial supervision for Point-DINO (no point decoding)."""
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdet.registry import MODELS


@torch.no_grad()
def distance_to_fidt(distance: Tensor,
                     gamma: float = 0.02,
                     phi: float = 0.75,
                     xi: float = 1.0) -> Tensor:
    """Evaluate 1 / (D ** (gamma * D + phi) + xi) in the log domain.

    This avoids overflowing the power at large pixel distances. At D=0,
    log_power=-inf and the result is exactly 1/xi (1 with the FIDT defaults).
    """
    distance = distance.float()
    log_power = (gamma * distance + phi) * distance.log()
    return torch.exp(
        -torch.logaddexp(log_power, distance.new_tensor(math.log(xi))))


@torch.no_grad()
def generate_fidt_target(
        points: Tensor,
        map_shape: Tuple[int, int],
        padded_shape: Tuple[int, int],
        img_shape: Tuple[int, int],
        gamma: float = 0.02,
        phi: float = 0.75,
        xi: float = 1.0,
        chunk_size: int = 4096,
        point_chunk_size: int = 256,
        local_radius_px: Optional[float] = None,
        return_valid_mask: bool = False) -> Tuple[Tensor, ...]:
    """Build the original FIDT target and a boolean loss mask [H_f, W_f].

    Points are the current augmented, floating-point image coordinates.
    Map cell centers are mapped using the actual batch tensor's padded H/W,
    not a nominal stride or the per-image pad_shape. Only centers within
    img_shape are valid. Empty images have an all-zero target.

    With local_radius_px=None, the loss mask is the original valid mask.
    Otherwise only valid cells within the pixel radius of the nearest GT
    participate; empty images select no cells. Target values are unchanged,
    including outside the local radius. Optionally also return the original
    valid mask as a third tensor, for debug counts without rebuilding grids.

    Both grid and GT points are chunked, bounding each distance temporary
    to chunk_size * point_chunk_size, independently of image/GT count.
    """
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError('points must have shape [N, 2].')
    if chunk_size <= 0 or point_chunk_size <= 0:
        raise ValueError('FIDT chunk sizes must be positive.')
    if local_radius_px is not None and (not math.isfinite(local_radius_px)
                                        or local_radius_px <= 0):
        raise ValueError(
            'local_radius_px must be None or finite and positive.')
    if not (math.isfinite(gamma) and gamma >= 0 and math.isfinite(phi)
            and phi > 0 and math.isfinite(xi) and xi > 0):
        raise ValueError('FIDT requires finite gamma>=0, phi>0 and xi>0.')
    map_h, map_w = map_shape
    pad_h, pad_w = padded_shape
    img_h, img_w = img_shape[:2]
    if min(map_h, map_w, pad_h, pad_w) <= 0:
        raise ValueError('Map and padded input dimensions must be positive.')
    if not (0 <= img_h <= pad_h and 0 <= img_w <= pad_w):
        raise ValueError('img_shape must lie within the batch input tensor.')

    # Explicit float32 also preserves pixel precision under mixed precision.
    points = points.detach().to(dtype=torch.float32)
    x = (torch.arange(map_w, device=points.device, dtype=torch.float32) +
         0.5) * (pad_w / map_w)
    y = (torch.arange(map_h, device=points.device, dtype=torch.float32) +
         0.5) * (pad_h / map_h)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    valid = (grid_x < img_w) & (grid_y < img_h)
    loss_mask = valid if local_radius_px is None else torch.zeros_like(valid)
    target = points.new_zeros((map_h, map_w))
    if points.shape[0] == 0:
        result = (target, loss_mask)
        return result + (valid, ) if return_valid_mask else result

    grid = torch.stack((grid_x[valid], grid_y[valid]), dim=-1)
    values = points.new_empty(grid.shape[0])
    selected = (torch.empty(
        grid.shape[0], device=points.device, dtype=torch.bool)
                if local_radius_px is not None else None)
    with torch.autocast(device_type=points.device.type, enabled=False):
        for start in range(0, grid.shape[0], chunk_size):
            grid_chunk = grid[start:start + chunk_size]
            nearest = points.new_full((grid_chunk.shape[0], ), float('inf'))
            for point_start in range(0, points.shape[0], point_chunk_size):
                distances = torch.cdist(
                    grid_chunk,
                    points[point_start:point_start + point_chunk_size],
                    p=2,
                    # Direct distances avoid cancellation near subpixel GTs.
                    compute_mode='donot_use_mm_for_euclid_dist')
                nearest = torch.minimum(nearest, distances.min(dim=1).values)
            values[start:start + chunk_size] = distance_to_fidt(
                nearest, gamma, phi, xi)
            if selected is not None:
                # Reuse the same pixel-space distances; no second cdist pass.
                selected[start:start + chunk_size] = nearest <= local_radius_px
    target[valid] = values
    if selected is not None:
        loss_mask[valid] = selected
    result = (target, loss_mask)
    return result + (valid, ) if return_valid_mask else result


@MODELS.register_module()
class PointDINOFIDTAuxHead(nn.Module):
    """Shared-feature FIDT head, called only by PointDINO.loss.

    The optional branch has no access to queries, matching, DN or point
    prediction. Its registry entry is independent of all existing heads.
    fidt_loss_weight defaults to zero; the detector skips construction when
    disabled or zero-weight, preserving the Stage 3 parameter set and RNG.
    local_radius_px=None preserves full-map supervision; a positive radius
    selects only local valid cells without changing the target or network.
    """

    def __init__(self,
                 in_channels: int = 256,
                 hidden_channels: int = 64,
                 num_groups: int = 8,
                 upsample_factor: int = 1,
                 gamma: float = 0.02,
                 phi: float = 0.75,
                 xi: float = 1.0,
                 fidt_loss_weight: float = 0.0,
                 chunk_size: int = 4096,
                 point_chunk_size: int = 256,
                 debug: bool = False,
                 local_radius_px: Optional[float] = None) -> None:
        super().__init__()
        if upsample_factor not in (1, 2):
            raise ValueError('FIDT supports no upsample or one x2 upsample.')
        if not math.isfinite(fidt_loss_weight) or fidt_loss_weight < 0:
            raise ValueError(
                'fidt_loss_weight must be finite and nonnegative.')
        if num_groups <= 0 or hidden_channels % num_groups:
            raise ValueError(
                'hidden_channels must be divisible by num_groups.')
        if local_radius_px is not None and (not math.isfinite(local_radius_px)
                                            or local_radius_px <= 0):
            raise ValueError(
                'local_radius_px must be None or finite and positive.')
        self.upsample_factor = upsample_factor
        self.fidt_loss_weight = float(fidt_loss_weight)
        self.debug = debug
        self.local_radius_px = local_radius_px
        self.target_cfg = dict(gamma=gamma,
                               phi=phi,
                               xi=xi,
                               chunk_size=chunk_size,
                               point_chunk_size=point_chunk_size)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, hidden_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels,
                      hidden_channels,
                      3,
                      padding=1,
                      bias=False), nn.GroupNorm(num_groups, hidden_channels),
            nn.ReLU(inplace=True))
        self.output = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, feature: Tensor) -> Tensor:
        """Return sigmoid FIDT prediction [B, 1, H_f, W_f]."""
        x = self.conv1(feature)
        if self.upsample_factor == 2:
            x = F.interpolate(x,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=False)
        return self.output(self.conv2(x)).float().sigmoid()

    @staticmethod
    def masked_mse(prediction: Tensor, target: Tensor,
                   valid: Tensor) -> Tensor:
        """Global mean over selected cells; an empty mask gives connected 0.

        Masking the squared error keeps the prediction graph even when no
        cells are selected, with exactly zero gradients at ignored cells.
        """
        squared_error = (prediction.float() - target.float()).square()
        return squared_error.masked_fill(~valid,
                                         0).sum() / valid.sum().clamp(min=1)

    def loss(self, feature: Tensor, batch_data_samples: Sequence,
             padded_shape: Tuple[int, int]) -> dict:
        """Supervise current augmented GTs once, retaining shared gradients."""
        prediction = self(feature)
        if prediction.shape[0] != len(batch_data_samples):
            raise ValueError('Feature batch and data sample count differ.')
        targets, masks = [], []
        valid_count = prediction.new_zeros(())
        for sample in batch_data_samples:
            meta = sample.metainfo
            # DetDataPreprocessor supplies batch_input_shape and pad_shape.
            # The tensor is authoritative; pad_shape can be smaller per image.
            if ('batch_input_shape' in meta and tuple(
                    meta['batch_input_shape']) != tuple(padded_shape)):
                raise ValueError(
                    'batch_input_shape disagrees with input tensor.')
            target, loss_mask, valid = generate_fidt_target(
                sample.gt_instances.points.to(device=prediction.device),
                prediction.shape[-2:],
                padded_shape,
                meta['img_shape'],
                local_radius_px=self.local_radius_px,
                return_valid_mask=True,
                **self.target_cfg)
            targets.append(target)
            masks.append(loss_mask)
            if self.debug and self.local_radius_px is not None:
                valid_count += valid.sum()
        target = torch.stack(targets).unsqueeze(1)
        loss_mask = torch.stack(masks).unsqueeze(1)
        raw_mse = self.masked_mse(prediction, target, loss_mask)
        losses = dict(loss_fidt=raw_mse * self.fidt_loss_weight)
        if self.debug:
            # MMEngine logs tensor scalars but sums only names with 'loss'.
            # Keep this name free of 'loss' so it cannot double supervision.
            losses['fidt_mse'] = raw_mse.detach()
            if self.local_radius_px is not None:
                local_count = loss_mask.sum().to(dtype=prediction.dtype)
                losses['fidt_local_count'] = local_count.detach()
                losses['fidt_local_ratio'] = (
                    local_count / valid_count.clamp(min=1)).detach()
        return losses
