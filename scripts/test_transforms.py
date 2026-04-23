import sys
import os
from hq_det import augment, dataset
import cv2
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    transforms = [
        augment.ToNumpy(),
        augment.RandomHorizontalFlip(),
        augment.RandomVerticalFlip(),
        augment.RandomGrayScale(),
        augment.RandomShuffleChannel(),
        augment.RandomRotate(),
        augment.RandomRotate90(),
        augment.RandomAffine(),
        augment.RandomPerspective(),
        augment.RandomNoise(),
        augment.RandomBrightness(),
        augment.RandomCrop(),
        augment.RandomResize(),
    ]
    transforms = augment.Compose(transforms)

    dataset_ = dataset.CocoDetection(
        input_path,
        os.path.join(input_path, '_annotations.coco.json'),
        transforms=transforms)
    
    # dataset_ = dataset.SplitDataset(
    #     dataset_,
    #     transforms=None,
    #     training=False,
    # )
    
    num = 0
    total = 0
    for i in tqdm(range(200)):
        data = dataset_[i]

        img = data['img']
        bboxes = data['bboxes']
        img_id = data['image_id']

        width = bboxes[:, 2] - bboxes[:, 0]
        height = bboxes[:, 3] - bboxes[:, 1]

        mask = (width < 4) | (height < 4)
        num += np.sum(mask)
        total += len(bboxes)
        for box in bboxes:
            img = cv2.rectangle(
                img.copy(),
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0),
                3
            )
            pass
        if img_id == 937:
            print(f"img_id: {img_id}, num: {num}, total: {total}, ratio: {num / total}")
            pass
        cv2.imwrite(os.path.join(output_path, f'{img_id}.jpg'), img)
        pass
    print(f"total: {total}, num: {num}, ratio: {num / total}")
    pass