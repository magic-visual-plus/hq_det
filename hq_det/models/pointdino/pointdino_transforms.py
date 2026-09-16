# Copyright (c) OpenMMLab. All rights reserved.
"""Point annotation support isolated from MMDetection's box transforms."""

import numpy as np

from mmdet.datasets.transforms import (LoadAnnotations, PackDetInputs,
                                       RandomCrop, RandomFlip, Resize)
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PointDINOLoadAnnotations(LoadAnnotations):
    """Load pixel coordinates from ``point`` / ``point_label`` instances.

    Coordinates are copied without a half-pixel or one-based adjustment, as
    in the source PointDINO pipeline. Load before augmentation for training;
    load after resizing for evaluation against original-coordinate predictions.
    """

    def __init__(self, with_point=True, **kwargs):
        kwargs.setdefault('with_bbox', False)
        kwargs.setdefault('with_label', False)
        super().__init__(**kwargs)
        self.with_point = with_point

    def transform(self, results):
        results = super().transform(results)
        if self.with_point:
            instances = results.get('instances', [])
            results['gt_points'] = np.array(
                [instance['point'] for instance in instances],
                dtype=np.float32).reshape((-1, 2))
            results['gt_points_labels'] = np.array(
                [instance['point_label'] for instance in instances],
                dtype=np.int64)
            results['gt_ignore_flags'] = np.array(
                [instance.get('ignore_flag', 0) for instance in instances],
                dtype=bool)
        return results


@TRANSFORMS.register_module()
class PointDINOPackDetInputs(PackDetInputs):
    """Pack points and their labels using the existing ignore-flag handling."""

    mapping_table = {
        **PackDetInputs.mapping_table,
        'gt_points_labels': 'labels',
        'gt_points': 'points',
    }


@TRANSFORMS.register_module()
class PointDINOResize(Resize):
    """Apply the image's width/height scale factors to point annotations."""

    def transform(self, results):
        results = super().transform(results)
        if results.get('gt_points') is not None:
            w_scale, h_scale = results['scale_factor']
            results['gt_points'][:, 0] *= w_scale
            results['gt_points'][:, 1] *= h_scale
        return results


@TRANSFORMS.register_module()
class PointDINORandomFlip(RandomFlip):
    """Flip points with the source pixel-index convention ``width - 1 - x``."""

    def _flip(self, results):
        super()._flip(results)
        if results.get('gt_points') is not None:
            h, w = results['img'].shape[:2]
            direction = results['flip_direction']
            if direction in ('horizontal', 'diagonal'):
                results['gt_points'][:, 0] = (
                    w - 1 - results['gt_points'][:, 0])
            if direction in ('vertical', 'diagonal'):
                results['gt_points'][:, 1] = (
                    h - 1 - results['gt_points'][:, 1])


@TRANSFORMS.register_module()
class PointDINORandomCrop(RandomCrop):
    """Crop points using the same sampled offset as the existing image crop.

    The parent retains responsibility for image, box, mask, and homography
    handling. Only the source point translation and target filtering are added.
    """

    def _rand_offset(self, margin):
        offset = super()._rand_offset(margin)
        self._pointdino_crop_offset = offset
        return offset

    def _crop_data(self, results, crop_size, allow_negative_crop):
        points = results.get('gt_points')
        labels = results.get('gt_points_labels')
        ignore_flags = results.get('gt_ignore_flags')
        results = super()._crop_data(results, crop_size, allow_negative_crop)
        if results is None or points is None:
            return results

        points = points.copy()
        offset_h, offset_w = self._pointdino_crop_offset
        points[:, 0] -= offset_w
        points[:, 1] -= offset_h
        crop_h, crop_w = results['img_shape'][:2]
        valid = ((points[:, 0] >= 0) & (points[:, 0] < crop_w)
                 & (points[:, 1] >= 0) & (points[:, 1] < crop_h))
        if not valid.any() and not allow_negative_crop:
            return None

        results['gt_points'] = points[valid]
        if labels is not None:
            results['gt_points_labels'] = labels[valid]
        if ignore_flags is not None:
            results['gt_ignore_flags'] = ignore_flags[valid]
        return results
