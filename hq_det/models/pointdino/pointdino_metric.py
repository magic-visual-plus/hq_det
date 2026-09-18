# Copyright (c) OpenMMLab. All rights reserved.
"""The source PointDINO thresholded, one-to-one point evaluation."""

from typing import Dict, List, Optional

import numpy as np
from mmengine.evaluator import BaseMetric
from scipy.optimize import linear_sum_assignment

from mmdet.registry import METRICS


@METRICS.register_module()
class PointDINOMetric(BaseMetric):
    """Evaluate point detections at pixel-distance thresholds.

    Predictions and ground truth must use the same image coordinate system.
    Images use their native size by default. With explicit resizing, the test
    pipeline keeps original GT coordinates to match ``rescale=True`` predictions.

    Matching first maximizes the number of pairs within a distance threshold,
    then minimizes their Euclidean distance. This preserves the source metric.
    """

    default_prefix = 'point'

    def __init__(self,
                 distance_thresholds=(5.0, 10.0),
                 score_threshold: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.distance_thresholds = [float(x) for x in distance_thresholds]
        for threshold in self.distance_thresholds:
            if threshold <= 0:
                raise ValueError('distance thresholds must be > 0.')
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError('score_threshold must be in [0, 1].')
        self.score_threshold = float(score_threshold)

    def _match_points(self, pred_points, gt_points, distance_threshold):
        num_pred, num_gt = len(pred_points), len(gt_points)
        if num_pred == 0 or num_gt == 0:
            return 0, 0.0
        distance_matrix = np.linalg.norm(
            pred_points[:, None, :] - gt_points[None, :, :], axis=-1)
        num_pairs = min(num_pred, num_gt)
        invalid_cost = (
            (num_pairs + 1) * max(distance_threshold, 1.0) + 1.0)
        cost_matrix = np.where(distance_matrix <= distance_threshold,
                               distance_matrix, invalid_cost)
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        matched_distances = distance_matrix[pred_indices, gt_indices]
        valid_matches = matched_distances <= distance_threshold
        return (int(valid_matches.sum()),
                float(matched_distances[valid_matches].sum()))

    def process(self, data_batch: dict, data_samples: List) -> None:
        for data_sample in data_samples:
            if isinstance(data_sample, dict):
                pred_instances = data_sample['pred_instances']
                gt_instances = data_sample['gt_instances']
                pred_points = pred_instances['points']
                pred_scores = pred_instances['scores']
                gt_points = gt_instances['points']
            else:
                pred_points = data_sample.pred_instances.points
                pred_scores = data_sample.pred_instances.scores
                gt_points = data_sample.gt_instances.points

            pred_points = pred_points.detach().cpu().numpy()
            pred_scores = pred_scores.detach().cpu().numpy()
            gt_points = gt_points.detach().cpu().numpy()
            pred_points = pred_points[pred_scores >= self.score_threshold]

            image_result = {}
            for threshold in self.distance_thresholds:
                tp, distance_sum = self._match_points(
                    pred_points, gt_points, threshold)
                image_result[str(int(threshold))] = dict(
                    tp=tp, fp=len(pred_points) - tp,
                    fn=len(gt_points) - tp, distance_sum=distance_sum)
            self.results.append(image_result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        metrics = {}
        for threshold in self.distance_thresholds:
            key = str(int(threshold))
            tp = sum(result[key]['tp'] for result in results)
            fp = sum(result[key]['fp'] for result in results)
            fn = sum(result[key]['fn'] for result in results)
            distance_sum = sum(result[key]['distance_sum'] for result in results)
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)
                  if precision + recall > 0 else 0.0)
            mean_error = distance_sum / tp if tp > 0 else 0.0
            metrics[f'precision@{key}px'] = float(precision)
            metrics[f'recall@{key}px'] = float(recall)
            metrics[f'f1@{key}px'] = float(f1)
            metrics[f'mean_localization_error@{key}px'] = float(mean_error)
            metrics[f'tp@{key}px'] = int(tp)
            metrics[f'fp@{key}px'] = int(fp)
            metrics[f'fn@{key}px'] = int(fn)
        return metrics
