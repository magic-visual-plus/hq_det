from typing import List
from .common import PredictionResult
import itertools


def prediction_result_list_to_coco(pred_records: List[PredictionResult]) -> List[dict]:
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
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

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