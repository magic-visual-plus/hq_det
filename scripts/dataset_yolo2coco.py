import sys
import os
import yaml
from collections import defaultdict
import cv2
from tqdm import tqdm
import numpy as np
import json


if __name__ == '__main__':
    yolo_path = sys.argv[1]
    yolo_yaml_file = sys.argv[2]
    coco_path = sys.argv[3]

    # Convert COCO dataset to YOLO format
    
    coco_annotation_file = os.path.join(coco_path, "_annotations.coco.json")
    coco_image_path = coco_path
    yolo_image_path = os.path.join(yolo_path, "images")
    yolo_label_path = os.path.join(yolo_path, "labels")

    os.makedirs(coco_image_path, exist_ok=True)

    with open(yolo_yaml_file, 'r') as f:
        yolo_data = yaml.safe_load(f)
        pass

    class_names = yolo_data['names']

    categories = []
    images = []
    annotations = []

    image_filenames = os.listdir(yolo_image_path)

    for i, image_filename in enumerate(tqdm(image_filenames)):
        if all(ext not in image_filename for ext in ['.jpg', '.jpeg', '.png']):
            continue

        image_name = os.path.splitext(image_filename)[0]
        image_file = os.path.join(yolo_image_path, image_filename)
        label_file = os.path.join(yolo_label_path, image_name + ".txt")
        img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)
        w = img.shape[1]
        h = img.shape[0]
        cv2.imencode('.jpg', img)[1].tofile(os.path.join(coco_image_path, image_filename))
        images.append({
            'id': i,
            'file_name': image_filename,
            'width': w,
            'height': h
        })
        if not os.path.exists(label_file):
            pass
        else:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    category_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    annotations.append({
                        'image_id': i,
                        'category_id': category_id,
                        'bbox': [
                            (x_center - width / 2) * w,  # x
                            (y_center - height / 2) * h,  # y
                            width * w,  # width
                            height * h  # height
                        ],
                        'area': width * height * w * h,
                        'iscrowd': 0,
                        'id': len(annotations)
                    })
                    pass
                pass
            pass
        pass

    categories = [{'id': i, 'name': name} for i, name in enumerate(class_names)]
    coco_data = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    with open(coco_annotation_file, 'w', encoding='utf8') as f:
        json.dump(coco_data, f, indent=4)
        pass
    pass