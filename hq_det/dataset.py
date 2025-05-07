import os
from torch.utils.data import Dataset
import torchvision
import cv2
import torch
from pycocotools import mask as coco_mask
import loguru
import numpy as np
import copy
from ultralytics.data.augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)

from ultralytics.utils import DEFAULT_CFG
from PIL import Image
from . import split_utils
import random


logger = loguru.logger


def xywh2cxcywh(x, width, height):
    """
    Convert xywh to cxcywh format.
    """
    r = []
    r.append((x[0] + x[2]) / 2 / width)
    r.append((x[1] + x[3]) / 2 / height)
    r.append((x[2] - x[0]) / width)
    r.append((x[3] - x[1]) / height)
    return np.array(r, dtype=np.float32)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.shape[1], image.shape[0]

        image_id = target["image_id"]
        # image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["bboxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        # self.ids = self.ids[:1000]
        logger.info("CocoDetection: img_folder {} using {} images", img_folder, len(self.ids))
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(False)
        self.id2names = {}
        for item in self.coco.cats.values():
            self.id2names[item['id']] = item['name']
        logger.info("id 2 names {}", self.id2names)
        # use coco dataset to set yolo's labels
        self.labels = [

        ]

    def _load_image(self, id: int) -> np.array:
        path = self.coco.loadImgs(id)[0]["file_name"]
        # print(path)
        return cv2.imread(os.path.join(self.root, path))

    def __len__(self):
        # return 100
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
            pass

        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if isinstance(img, Image.Image):
            img = np.array(img)
            # convert to bgr
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            pass

        target['img'] = img
        target['cls'] = target['labels'].numpy()
        target['bboxes'] = target['bboxes'].numpy()
        target['original_shape'] = (img.shape[0], img.shape[1])
        
        # print(target)
        if self._transforms is not None:
            target = self._transforms(target)
            pass

        return target
    
    @property
    def class_id2names(self):
        return self.id2names



class SplitDataset(Dataset):
    def __init__(self, dataset, transforms=None, split_size=1024, shift=20, max_split=2, training=True):
        self.dataset = dataset
        self.training = training
        self.split_size = split_size
        self.shift = shift
        self.max_split = max_split
        self.transforms = transforms
        if self.training:
            self.num = len(dataset) * 5
        else:
            num = 0
            for i in range(len(dataset)):
                data = dataset[i]
                img = data['img']
                boxes = data['bboxes']
                cls = data['cls']
                splits = split_utils.split_image(img, boxes, cls, self.split_size, self.shift, self.max_split)
                num += len(splits)
                pass
            self.num = num
            pass
        self.buffer = []
        self.origin_idx = 0


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.training:
            if len(self.buffer) == 0:
                idx = random.randint(0, len(self.dataset) - 1)
                self.fill_buffer(idx)
                pass

            img, boxes, cls, startx, starty, image_id = self.buffer.pop(0)
            
            data = {
                'img': img,
                'bboxes': boxes,
                'cls': cls,
                'image_id': image_id,
            }
        else:
            if len(self.buffer) == 0:
                idx = self.origin_idx
                self.fill_buffer(idx)
                self.origin_idx += 1
                pass

            img, boxes, cls, startx, starty, image_id = self.buffer.pop(0)
            data = {
                'img': img,
                'bboxes': boxes,
                'cls': cls,
                'image_id': image_id,
            }
            pass

        if self.transforms is not None:
            data = self.transforms(data)
            pass

        return data
    
    def fill_buffer(self, idx):
        data = self.dataset[idx]
        
        img = data['img']
        boxes = data['bboxes']
        cls = data['cls']
        image_id = data['image_id']

        splits = split_utils.split_image(img, boxes, cls, self.split_size, self.shift, self.max_split)
        for i, s in enumerate(splits):
            img, boxes, cls, startx, starty = s
            sub_image_id = image_id * (self.max_split ** 2 + 1) + i

            if sub_image_id == 859:
                print("sub_image_id: {}".format(sub_image_id))
                pass
            self.buffer.append((img, boxes, cls, startx, starty, sub_image_id))
            pass
        pass