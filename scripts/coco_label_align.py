import sys
import os
import json


if __name__ == '__main__':
    input_file1 = sys.argv[1]
    input_file2 = sys.argv[2]
    output_file = sys.argv[3]

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

    with open(output_file, 'w') as f:
        json.dump(
            data2, f, indent=4, ensure_ascii=False)
        pass
    pass
