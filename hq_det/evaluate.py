from typing import List, Set
from .common import PredictionResult
import itertools
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def prediction_result_list_to_coco(pred_records: List[PredictionResult], class_ids: Set[int] = None) -> List[dict]:
    """
    Convert a list of PredictionResult objects to COCO format.

    Args:
        pred_records (List[PredictionResult]): List of predicted records.

    Returns:
        List[dict]: List of dictionaries in COCO format.
    """
    coco_results = []
    for pred in pred_records:
        for ann in pred.to_coco():
            ann['id'] = len(coco_results)
            if class_ids is not None and ann['category_id'] not in class_ids:
                continue
            ann['category_id'] = 0
            coco_results.append(ann)
            pass
        pass
    return coco_results


def eval_detection_result(gt_records: List[PredictionResult], 
                          pred_records: List[PredictionResult], class_names) -> float:
    """
    Evaluate detection results using COCO evaluation metrics.

    Args:
        gt_records (List[PredictionResult]): List of ground truth records.
        pred_records (List[PredictionResult]): List of predicted records.
        iou_threshold (float): IoU threshold for evaluation.

    Returns:
        float: Mean Average Precision (mAP) score.
    """

    # Initialize COCO ground truth and detection objects
    coco_gt = COCO()
    coco_pred = COCO()

    # Load ground truth and detection results into COCO format
    coco_gt.dataset['annotations'] = prediction_result_list_to_coco(gt_records)
    coco_gt.dataset['categories'] = [{'id': i, 'name': name} for i, name in enumerate(class_names)]
    coco_gt.dataset['images'] = [{'id': rec.image_id} for rec in gt_records]
    coco_gt.createIndex()
    
    
    coco_pred.dataset['annotations'] = prediction_result_list_to_coco(pred_records)
    coco_pred.dataset['categories'] = coco_gt.dataset['categories']
    coco_pred.dataset['images'] = [{'id': rec.image_id} for rec in pred_records]
    coco_pred.createIndex()

    # Initialize COCO evaluation object
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


    # Return mAP score
    return coco_eval.stats[0]  # mAP at IoU=0.5:0.95


def eval_detection_result_by_class_id(
        gt_records: List[PredictionResult], pred_records: List[PredictionResult], class_ids):

    coco_gt = COCO()
    coco_pred = COCO()

    class_ids = set(class_ids)
    coco_gt.dataset['annotations'] = prediction_result_list_to_coco(gt_records, class_ids)
    coco_gt.dataset['categories'] = [{'id': 0, 'name': 'ng'}]
    coco_gt.dataset['images'] = [{'id': rec.image_id} for rec in gt_records]
    coco_gt.createIndex()
    coco_pred.dataset['annotations'] = prediction_result_list_to_coco(pred_records, class_ids)
    coco_pred.dataset['categories'] = [{'id': 0, 'name': 'ng'}]
    coco_pred.dataset['images'] = [{'id': rec.image_id} for rec in pred_records]
    coco_pred.createIndex()

    # Initialize COCO evaluation object
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map = coco_eval.stats[0]  # mAP at IoU=0.5:0.95
    precisions = np.clip(coco_eval.eval['precision'][0, :, 0, 0, 2], 0, 1)
    recalls = coco_eval.params.recThrs
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)

    best_idx = np.argmax(f1s)
    best_f1 = f1s[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    best_confidence = coco_eval.eval['scores'][0, :, 0, 0, 2][best_idx]

    fnr = 1 - best_recall

    return {
        'mAP': map,
        'precision': best_precision,
        'recall': best_recall,
        'f1_score': best_f1,
        'fnr': fnr,
        'confidence': best_confidence,
        'precisions': precisions,
        'recalls': recalls,
    }

    pass