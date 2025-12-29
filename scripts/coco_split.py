
import sys
import os
import json
import random
import shutil
from tqdm import tqdm


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    train_ratio = 0.8

    with open(os.path.join(input_path, '_annotations.coco.json'), 'r') as f:
        annotations = json.load(f)
        pass

    annotations_train = annotations.copy()
    annotations_val = annotations.copy()
    annotations_train['images'] = []
    annotations_train['annotations'] = []
    annotations_val['images'] = []
    annotations_val['annotations'] = []
    for image in annotations['images']:
        basename = image['file_name']

        if random.random() < train_ratio:
            split = 'train'
            annotations_train['images'].append(image)
        else:
            split = 'val'
            annotations_val['images'].append(image)
            pass

        dst_file = os.path.join(output_path, split, basename)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        src_file = os.path.join(input_path, basename)
        shutil.copyfile(src_file, dst_file)
        pass

    image_id_set_train = {img['id'] for img in annotations_train['images']}
    image_id_set_val = {img['id'] for img in annotations_val['images']}

    annotations_train['annotations'] = [
        ann for ann in annotations['annotations'] if ann['image_id'] in image_id_set_train
    ]
    annotations_val['annotations'] = [
        ann for ann in annotations['annotations'] if ann['image_id'] in image_id_set_val
    ]
    with open(os.path.join(output_path, 'train', '_annotations.coco.json'), 'w') as f:
        json.dump(annotations_train, f, indent=4, ensure_ascii=False)
        pass
    with open(os.path.join(output_path, 'val', '_annotations.coco.json'), 'w') as f:
        json.dump(annotations_val, f, indent=4, ensure_ascii=False)
        pass
