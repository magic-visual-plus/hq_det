import os
import json
from tqdm import tqdm
import cv2
import numpy as np


def extract_boxes(input_path: str, output_path: str):
    annotation_file = os.path.join(input_path, '_annotations.coco.json')

    with open(annotation_file, 'r') as f:
        coco_input = json.load(f)
        pass

    os.makedirs(output_path, exist_ok=True)
    image_id2ann_idx = dict()
    for idx, ann in enumerate(coco_input['annotations']):
        image_id = ann['image_id']
        if image_id not in image_id2ann_idx:
            image_id2ann_idx[image_id] = []
            pass
        image_id2ann_idx[image_id].append(idx)
        pass

    for i, image in enumerate(tqdm(coco_input['images'])):
        image_path = os.path.join(input_path, image['file_name'])
        ann_idx = image_id2ann_idx.get(image['id'], [])
        if len(ann_idx) == 0:
            continue
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        for aidx in ann_idx:
            ann = coco_input["annotations"][aidx]
            x, y, w, h = ann['bbox']
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)

            if h < 0:
                h = -h
                pass
            if w < 0:
                w = -w
                pass

            c = ann["category_id"]
            subimg = img[int(y):int(y+h), int(x):int(x+w)]
            subimg_name = f"{os.path.splitext(image['file_name'])[0]}_{aidx}_{c}.jpg"
            subimg_path = os.path.join(output_path, subimg_name)
            cv2.imencode('.jpg', subimg)[1].tofile(subimg_path)
            pass
        pass
    pass