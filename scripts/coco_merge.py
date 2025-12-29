import sys
import os
import json
import shutil

if __name__ == '__main__':
    input_paths = sys.argv[1:-1]
    output_path = sys.argv[-1]

    annotations_merged = None
    for input_path in input_paths:
        with open(os.path.join(input_path, '_annotations.coco.json'), 'r') as f:
            annotations = json.load(f)
            pass

        if annotations_merged is None:
            annotations_merged = annotations

            # copy all files
            for image in annotations['images']:
                src_file = os.path.join(input_path, image['file_name'])
                dst_file = os.path.join(output_path, image['file_name'])
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copyfile(src_file, dst_file)
                pass
            pass
        else:
            image_id_map = dict()
            category_id_map = dict()
            category_names = {cat['name']: cat for cat in annotations_merged['categories']}
            for cat in annotations['categories']:
                if cat['name'] in category_names:
                    category_id_map[cat['id']] = category_names[cat['name']]['id']
                    pass
                else:
                    new_cat = cat.copy()
                    new_cat['id'] = len(annotations_merged['categories'])
                    category_id_map[cat['id']] = new_cat['id']
                    annotations_merged['categories'].append(new_cat)
                    pass
                pass

            for image in annotations['images']:
                new_image_id = len(annotations_merged['images'])
                image_id_map[image['id']] = new_image_id
                image['id'] = new_image_id
                annotations_merged['images'].append(image)

                # copy file
                src_file = os.path.join(input_path, image['file_name'])
                dst_file = os.path.join(output_path, image['file_name'])
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copyfile(src_file, dst_file)
                pass

            for ann in annotations['annotations']:
                ann['category_id'] = category_id_map[ann['category_id']]
                ann['image_id'] = image_id_map[ann['image_id']]
                ann['id'] = len(annotations_merged['annotations'])
                annotations_merged['annotations'].append(ann)
                pass
            pass
        pass

    with open(os.path.join(output_path, '_annotations.coco.json'), 'w') as f:
        json.dump(annotations_merged, f, indent=4, ensure_ascii=False)
        pass
    pass