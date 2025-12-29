import sys
import os
import json


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    label_map = dict()
    with open(input_file, 'r') as f:
        coco_data = json.load(f)
        pass

    coco_data['categories'] = [
        coco_data['categories'][0]
    ]
    coco_data['categories'][0]['name'] = 'object'

    for ann in coco_data['annotations']:
        ann['category_id'] = 0
        pass

    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)
        pass

    pass