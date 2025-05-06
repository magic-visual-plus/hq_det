import sys
import os
import json
import random
import shutil


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    output_path_train = os.path.join(output_path, "train")
    output_path_val = os.path.join(output_path, "valid")

    os.makedirs(output_path_train, exist_ok=True)
    os.makedirs(output_path_val, exist_ok=True)

    input_ann_file = os.path.join(input_path, "_annotations.coco.json")

    with open(input_ann_file, "r") as f:
        data = json.load(f)
        pass

    
    images = data["images"]
    # split images into train and val
    image_ids = [im['id'] for im in images]
    random.shuffle(image_ids)
    num_train = int(len(image_ids) * 0.8)
    train_ids = image_ids[:num_train]
    val_ids = image_ids[num_train:]

    train_ids = set(train_ids)
    data_train = {
        "images": [],
        "annotations": [],
        "categories": data["categories"],
    }
    data_val = {
        "images": [],
        "annotations": [],
        "categories": data["categories"],
    }
    for image in images:
        if image["id"] in train_ids:
            data_train["images"].append(image)
        else:
            data_val["images"].append(image)
            pass
        pass
    for ann in data["annotations"]:
        if ann["image_id"] in train_ids:
            data_train["annotations"].append(ann)
        else:
            data_val["annotations"].append(ann)
            pass
        pass
    with open(os.path.join(output_path_train, "_annotations.coco.json"), "w") as f:
        json.dump(data_train, f)
        pass
    with open(os.path.join(output_path_val, "_annotations.coco.json"), "w") as f:
        json.dump(data_val, f)
        pass

    # copy images
    for image in images:
        if image["id"] in train_ids:
            src = os.path.join(input_path, image["file_name"])
            dst = os.path.join(output_path_train, image["file_name"])
            shutil.copyfile(src, dst)
        else:
            src = os.path.join(input_path, image["file_name"])
            dst = os.path.join(output_path_val, image["file_name"])
            shutil.copyfile(src, dst)
            pass
        pass
    pass