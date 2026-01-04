import sys
import os
import json
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_paths', type=str, nargs='+', required=True,
                        help='List of input dataset paths to merge')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output dataset path')
    parser.add_argument('--same_category', action='store_true', default=False,
                        help='Whether to treat categories with the same name as the same category')
    args = parser.parse_args()

    output_path = args.output_path
    annotations_merged = None
    for input_path in args.input_paths:
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
            max_category_id = max(cat['id'] for cat in annotations_merged['categories'])
            max_image_id = max(img['id'] for img in annotations_merged['images'])

            for cat in annotations['categories']:
                if cat['name'] in category_names:
                    if args.same_category and cat['name'] in ['其他', '其它']:
                        # ignore 'other' category when merging with same_category option
                        pass
                    else:
                        category_id_map[cat['id']] = category_names[cat['name']]['id']
                        pass
                    pass
                else:
                    # add new category
                    if not args.same_category:
                        new_cat = cat.copy()
                        new_cat['id'] = max_category_id + 1
                        max_category_id += 1
                        category_id_map[cat['id']] = new_cat['id']
                        annotations_merged['categories'].append(new_cat)
                        pass
                    pass
                pass

            need_category_ids = set(category_id_map.keys())
            need_image_ids = set(ann['image_id'] for ann in annotations['annotations'] if ann['category_id'] in need_category_ids)

            for image in annotations['images']:
                if image['id'] not in need_image_ids:
                    continue
                    pass
                new_image_id = max_image_id + 1
                max_image_id += 1
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
                if ann['category_id'] not in need_category_ids:
                    continue
                    pass
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