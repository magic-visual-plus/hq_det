import sys
import os
import yaml
from collections import defaultdict
import cv2
from tqdm import tqdm


if __name__ == '__main__':
    coco_path = sys.argv[1]
    yolo_path = sys.argv[2]

    # Convert COCO dataset to YOLO format
    
    annotation_file = os.path.join(coco_path, "_annotations.coco.json")
    image_path = coco_path
    yolo_image_path = os.path.join(yolo_path, "images")
    yolo_label_path = os.path.join(yolo_path, "labels")

    os.makedirs(yolo_image_path, exist_ok=True)
    os.makedirs(yolo_label_path, exist_ok=True)

    with open(annotation_file, 'r') as f:
        coco_data = yaml.safe_load(f)
        pass

    categories = coco_data['categories']
    class_names = ['' for _ in range(len(categories))]
    for i, category in enumerate(categories):
        class_names[category['id']] = category['name']
        pass
    images = coco_data['images']
    images = {image['id']: image for image in images}
    annotations = coco_data['annotations']

    bboxes = defaultdict(list)
    for ann in tqdm(annotations):
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']
        # Convert COCO bbox format (x, y, width, height) to YOLO format (x_center, y_center, width, height)
        x_center = (bbox[0] + bbox[2] / 2) / images[image_id]['width']
        y_center = (bbox[1] + bbox[3] / 2) / images[image_id]['height']
        width = bbox[2] / images[image_id]['width']
        height = bbox[3] / images[image_id]['height']
        # Save the YOLO format annotation
        bboxes[image_id].append((category_id, x_center, y_center, width, height))
        pass

    for _, image in tqdm(images.items()):
        image_id = image['id']
        image_name = image['file_name']
        image_width = image['width']
        image_height = image['height']
        # Save the YOLO format annotation
        yolo_image_file = os.path.join(yolo_image_path, image_name)
        basename = os.path.splitext(image_name)[0]
        yolo_image_file = os.path.join(yolo_image_path, basename + ".jpg")
        yolo_label_file = os.path.join(yolo_label_path, basename + ".txt")
        with open(yolo_label_file, 'w') as f:
            for bbox in bboxes[image_id]:
                category_id, x_center, y_center, width, height = bbox
                f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")
                pass
            pass
        img = cv2.imread(os.path.join(image_path, image_name))
        if img is not None:
            cv2.imwrite(yolo_image_file, img)
            pass
        else:
            print(f"Error reading image: {image_name}")
        pass

    print(class_names)
    pass