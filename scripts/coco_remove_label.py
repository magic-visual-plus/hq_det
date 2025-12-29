import sys
import json


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    label_to_remove = sys.argv[3]

    with open(input_file, 'r') as f:
        data = json.load(f)
        pass

    categories = data['categories']
    label_id_to_remove = [cat['id'] for cat in categories if cat['name'] == label_to_remove]
    if not label_id_to_remove:
        print(f"Label '{label_to_remove}' not found in categories.")
        sys.exit(1)

    label_id_to_remove = label_id_to_remove[0]
    categories = [cat for cat in categories if cat['name'] != label_to_remove]
    cat_map = {cat['id']: i for i, cat in enumerate(categories)}
    for cat in categories:
        cat['id'] = cat_map[cat['id']]
        pass

    data['categories'] = categories

    annotations = data['annotations']
    annotations_ = []
    for ann in annotations:
        if ann['category_id'] != label_id_to_remove:
            ann['category_id'] = cat_map[ann['category_id']]
            annotations_.append(ann)
            pass
        pass
    data['annotations'] = annotations_

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
        pass
    pass