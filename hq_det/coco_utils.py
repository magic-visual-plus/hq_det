import os
import json
from tqdm import tqdm
import cv2
import numpy as np
import random
import shutil


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


def coco_sample(src_path, dst_path, max_size=-1, only_positive=False, max_num=-1):
    src_annotation_file = os.path.join(src_path, "_annotations.coco.json")
    os.makedirs(dst_path, exist_ok=True)
    with open(src_annotation_file, "r") as f:
        src_coco_data = json.load(f)
        pass

    image2annotations = dict()
    for annotation in src_coco_data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in image2annotations:
            image2annotations[image_id] = []
        image2annotations[image_id].append(annotation)
        pass
    
    src_images = src_coco_data["images"]
    random.shuffle(src_images)

    dst_images = []
    dst_annotations = []
    dst_categories = src_coco_data["categories"]

    total_size = 0
    total_num = 0
    for image in src_images:
        image_id = image["id"]
        annotations = image2annotations.get(image_id, [])

        if only_positive and len(annotations) == 0:
            continue

        file_name = image["file_name"]
        image_size = os.path.getsize(os.path.join(src_path, file_name))

        # recode image id and annotation id
        image["id"] = len(dst_images)
        for annotation in annotations:
            annotation["id"] = len(dst_annotations)
            annotation["image_id"] = image["id"]
            dst_annotations.append(annotation)
            pass
        dst_images.append(image)

        shutil.copy(os.path.join(src_path, file_name), os.path.join(dst_path, file_name))

        total_size += image_size
        total_num += 1

        if max_size > 0 and total_size > max_size:
            break

        if max_num > 0 and total_num >= max_num:
            break
        pass

    dst_coco_data = {
        "images": dst_images,
        "annotations": dst_annotations,
        "categories": dst_categories,
    }

    dst_annotation_file = os.path.join(dst_path, "_annotations.coco.json")
    with open(dst_annotation_file, "w") as f:
        json.dump(dst_coco_data, f, indent=4, ensure_ascii=False)
        pass
    pass


def coco_label_align(input_file1, input_file2, output_file1=None, output_file2=None):
    with open(input_file1, 'r') as f:
        data1 = json.load(f)
        pass

    with open(input_file2, 'r') as f:
        data2 = json.load(f)
        pass

    category_id_map = dict()
    category_names = {cat['name']: cat for cat in data1['categories']}
    for cat in data2['categories']:
        if cat['name'] in category_names:
            category_id_map[cat['id']] = category_names[cat['name']]['id']
            pass
        else:
            new_cat = cat.copy()
            new_cat['id'] = len(data1['categories'])
            category_id_map[cat['id']] = new_cat['id']
            data1['categories'].append(new_cat)
            pass
        pass

    data2['categories'] = data1['categories']
    for ann in data2['annotations']:
        ann['category_id'] = category_id_map[ann['category_id']]
        pass

    if output_file1 is not None:
        with open(output_file1, 'w') as f:
            json.dump(
                data1, f, indent=4, ensure_ascii=False)
            pass
        pass

    if output_file2 is not None:
        with open(output_file2, 'w') as f:
            json.dump(
                data2, f, indent=4, ensure_ascii=False)
            pass