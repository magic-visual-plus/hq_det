import cv2
import numpy as np
import os
import json
from collections import defaultdict
from tqdm import tqdm
import shutil
from .models import base


def split_image(img, boxes, cls, stride=1024, shift=20, max_split=3):
    splits = []
    max_size = stride * max_split - (max_split - 1) * shift
    h, w = img.shape[:2]

    max_hw = max(h, w)
    if max_hw <= stride:
        # if the image is small enough, just return the image
        splits.append((img, boxes, cls, 0, 0))
        return splits
    elif max_hw <= stride * 1.3:
        rate = stride / max_hw
        img = cv2.resize(img, (int(w * rate), int(h * rate)))
        boxes_ = boxes * rate
        splits.append((img, boxes_, cls, 0, 0))
        return splits
    else:
        rate = stride / max_hw
        img_ = cv2.resize(img, (int(w * rate), int(h * rate)))
        boxes_ = boxes * rate
        splits.append((img_, boxes_, cls, 0, 0))

        # split the image
        if max_hw > max_size:
            rate = max_size / max_hw
            img = cv2.resize(img, (int(w * rate), int(h * rate)))
            boxes = boxes * rate
            pass
        
        stride = stride - shift
        for i in range(0, img.shape[0], stride):
            if (i + shift) >= img.shape[0]:
                break
            for j in range(0, img.shape[1], stride):
                if (j + shift) >= img.shape[1]:
                    break

                if i + stride > h:
                    i = h - stride
                if j + stride > w:
                    j = w - stride

                startx = j
                endx = j + stride + shift
                starty = i
                endy = i + stride + shift

                # check if any box is in the current split
                # calculate intersection of box and window
                # box: [x1, y1, x2, y2]
                # window: [startx, starty, endx, endy]

                x_overlap = np.maximum(0, np.minimum(endx, boxes[:, 2]) - np.maximum(startx, boxes[:, 0]))
                y_overlap = np.maximum(0, np.minimum(endy, boxes[:, 3]) - np.maximum(starty, boxes[:, 1]))
                # calculate area of intersection
                intersection_area = x_overlap * y_overlap
                original_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

                mask = (intersection_area / original_area) > 0.5
                # mask = (boxes[:, 0] < endx) & (boxes[:, 1] < endy) & \
                #           (boxes[:, 2] > startx) & (boxes[:, 3] > starty)
                
                if mask.sum() == 0:
                    sub_boxes = np.zeros((0, 4), dtype=np.float32)
                    sub_cls = np.zeros((0,), dtype=np.int64)
                else:
                    sub_boxes = boxes[mask]
                    sub_cls = cls[mask]
                    sub_boxes[:, 0] -= startx
                    sub_boxes[:, 1] -= starty
                    sub_boxes[:, 2] -= startx
                    sub_boxes[:, 3] -= starty
                    sub_boxes[:, 0] = np.clip(sub_boxes[:, 0], 0, stride + shift)
                    sub_boxes[:, 1] = np.clip(sub_boxes[:, 1], 0, stride + shift)
                    sub_boxes[:, 2] = np.clip(sub_boxes[:, 2], 0, stride + shift)
                    sub_boxes[:, 3] = np.clip(sub_boxes[:, 3], 0, stride + shift)
                    pass

                sub_img = img[starty:endy, startx:endx]
                splits.append((sub_img, sub_boxes, sub_cls, startx, starty))
                pass


            pass

        return splits
    pass



def split_coco(input_path, output_path, stride, shift, max_split):
    os.makedirs(output_path, exist_ok=True)

    ann_file = os.path.join(input_path, '_annotations.coco.json')

    with open(ann_file, 'r') as f:
        data = json.load(f)
        images = data['images']
        annotations = data['annotations']
        categories = data['categories']
        pass
    
    data_split = dict()
    data_split['images'] = []
    data_split['annotations'] = []
    data_split['categories'] = categories

    image_anns = defaultdict(list)
    for ann in annotations:
        image_id = ann['image_id']
        
        image_anns[image_id].append(ann)
        pass

    ann_id_num = 0
    for image in tqdm(images):
        image_id = image['id']
        image_name = image['file_name']
        image_width = image['width']
        image_height = image['height']
        
        img = cv2.imread(os.path.join(input_path, image_name))
        boxes = []
        cls = []
        for ann in image_anns[image_id]:
            box = ann['bbox']
            box = np.array(box, dtype=np.float32)
            box[2] += box[0]
            box[3] += box[1]

            boxes.append(box)
            cls.append(ann['category_id'])
            pass
        
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            cls = np.array(cls, dtype=np.int64)
            pass
        splits = split_image(img, boxes, cls, stride, shift, max_split)

        for sub_img, sub_boxes, sub_cls, startx, starty in splits:
            sub_image_id = image_id * (max_split ** 2 + 1) + len(data_split['images'])
            data_split['images'].append({
                'id': sub_image_id,
                'file_name': f'{sub_image_id}.jpg',
                'width': sub_img.shape[1],
                'height': sub_img.shape[0],
            })
            for box, c in zip(sub_boxes, sub_cls):
                box[2] -= box[0]
                box[3] -= box[1]
                # print(box.tolist())
                data_split['annotations'].append({
                    'image_id': sub_image_id,
                    'bbox': box.tolist(),
                    'category_id': int(c),
                    'area': float(box[2] * box[3]),
                    'iscrowd': 0,
                    'id': ann_id_num,
                })
                ann_id_num += 1
                pass
            # if sub_image_id == 937:
            #     print('sub_image_id:', sub_image_id)
            cv2.imwrite(os.path.join(output_path, f'{sub_image_id}.jpg'), sub_img)
            pass
        pass

    with open(os.path.join(output_path, '_annotations.coco.json'), 'w') as f:
        json.dump(data_split, f, indent=4)
        pass
    pass


def predict_split(model: base.HQModel, img, thr, stride, shift, max_split):
    boxes = np.zeros((0, 4), dtype=np.float32)
    cls = np.zeros((0,), dtype=np.int64)
    splits = split_image(img, boxes, cls, stride, shift, max_split)
    results = []
    
    imgs = [s[0] for s in splits]
    results = model.predict(imgs)
    
    # merge results
    total_boxes = []
    for i, (sub_img, sub_boxes, sub_cls, startx, starty) in enumerate(splits):
        if len(results[i].bboxes) == 0:
            continue

        boxes = results[i].bboxes
        cls = results[i].cls
        scores = results[i].scores

        mask = scores > thr
        boxes = boxes[mask]
        cls = cls[mask]
        scores = scores[mask]
        
        
        pass