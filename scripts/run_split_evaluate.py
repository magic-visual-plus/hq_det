

import sys
from hq_det.models import rtdetr
from hq_det.models.dino import hq_dino
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
from hq_det.common import PredictionResult
from hq_det import augment
from hq_det import split_utils
import os
import torch
from hq_det import torch_utils
from ultralytics.utils import DEFAULT_CFG
import cv2
from tqdm import tqdm
from hq_det import evaluate
import numpy as np


if __name__ == '__main__':
    input_path = sys.argv[2]
    
    eval_class_names=[
            '划伤', '划痕', '压痕', '吊紧', '异物外漏', '折痕', '抛线', '拼接间隙', '烫伤', '爆针线', '破损', '碰伤', '线头', '脏污', '褶皱(贯穿)', '褶皱（轻度）', '褶皱（重度）', '重跳针'
        ]
    
    model = rtdetr.HQRTDETR(model=sys.argv[1])
    model.eval()
    
    model.to("cuda:0")

    class_names = model.get_class_names()
    eval_class_ids = [class_names.index(c) for c in eval_class_names]

    transforms = []
    transforms.append(augment.ToNumpy())
    # transforms.append(augment.Resize(max_size=1024))
    # transforms.append(augment.FilterSmallBox())
    # transforms.append(augment.Format())

    transforms = augment.Compose(transforms)

    dataset_ = CocoDetection(
        input_path,
        os.path.join(input_path, '_annotations.coco.json'),
        transforms=transforms)

    preds = []
    gts = []
    for idx in tqdm(range(len(dataset_))):
        data = dataset_[idx]

        img = data['img']
        bboxes = data['bboxes']
        cls = data['cls']

        gt_record = PredictionResult()
        gt_record.bboxes = bboxes
        gt_record.cls = cls
        gt_record.image_id = data['image_id']
        gt_record.scores = np.ones(len(bboxes), dtype=np.float32)
        gts.append(gt_record)
        
        # results = model.predict([img], bgr=True, confidence=0.0, max_size=1024)
        # result = results[0]
        result = split_utils.predict_split(model, img, 0.2, 1024, 20, 2, bgr=True, gt_boxes=bboxes, gt_cls=cls)
        result.image_id = data['image_id']

        preds.append(result)
        pass

    evaluate.eval_detection_result(gts, preds, model.get_class_names())
    stat = evaluate.eval_detection_result_by_class_id(gts, preds, eval_class_ids)
    print(stat)

    for pred in preds:
        mask = pred.scores > 0.4
        pred.bboxes = pred.bboxes[mask]
        pred.cls = pred.cls[mask]
        pred.scores = pred.scores[mask]
        pass

    stat = evaluate.eval_detection_result_by_class_id(gts, preds, eval_class_ids)
    print(stat)
    pass