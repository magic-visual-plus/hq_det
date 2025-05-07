import cv2
import numpy as np
import torch
import torchvision.transforms.functional as VF
from PIL import Image
import random
from . import box_utils
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def ensure_cv2(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pass
    elif isinstance(img, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    
    return img

def letterbox_torch(img, bboxes, new_shape=(640, 640), fill=(0.44, 0.44, 0.44)):
    # new_shape: [H, W]
    # img: [C, H, W]
    # Scale ratio (new / old)
    shape = img.shape[1:3]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)  # only scale down, do not scale up (for better val mAP)

    # Compute padding
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = VF.resize(img, new_unpad)
        pass

    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))
    img_ = torch.ones((3, new_shape[0], new_shape[1]), dtype=img.dtype)
    img_[:, top : top + new_unpad[0], left : left + new_unpad[1]] = img

    boxes_ = bboxes.copy()
    boxes_[:, 0] = boxes_[:, 0] * r + left
    boxes_[:, 1] = boxes_[:, 1] * r + top
    boxes_[:, 2] = boxes_[:, 2] * r + left
    boxes_[:, 3] = boxes_[:, 3] * r + top
    return img_, boxes_

    pass


class Letterbox:
    def __init__(self, new_shape=(640, 640)):
        self.new_shape = new_shape
        self.scaleup = False
        self.auto = False
        self.scale_fill = False
        self.center = True
        pass

    def __call__(self, img, boxes):
        shape = img.shape[:2]
        new_shape = self.new_shape

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        
        # update boxes
        boxes = boxes.copy()
        boxes[:, 0] = boxes[:, 0] * ratio[0] + left
        boxes[:, 1] = boxes[:, 1] * ratio[1] + top
        boxes[:, 2] = boxes[:, 2] * ratio[0] + left
        boxes[:, 3] = boxes[:, 3] * ratio[1] + top
        boxes[:, 0] = np.clip(boxes[:, 0], 0, new_shape[1])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, new_shape[0])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, new_shape[1])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, new_shape[0])


        return img, boxes


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() <= self.p:
            img = data['img']
            bboxes = data['bboxes']

            img = cv2.flip(img, 1)
            bboxes_ = bboxes.copy()
            bboxes_[:, 0] = img.shape[1] - bboxes[:, 2]
            bboxes_[:, 2] = img.shape[1] - bboxes[:, 0]
            bboxes_[:, 0] = np.clip(bboxes_[:, 0], 0, img.shape[1])
            bboxes_[:, 2] = np.clip(bboxes_[:, 2], 0, img.shape[1])

            data['img'] = img
            data['bboxes'] = bboxes_
            pass
        return data
    pass

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() <= self.p:
            img = data['img']
            bboxes = data['bboxes']


            img = cv2.flip(img, 0)
            bboxes_ = bboxes.copy()
            bboxes_[:, 1] = img.shape[0] - bboxes[:, 3]
            bboxes_[:, 3] = img.shape[0] - bboxes[:, 1]
            bboxes_[:, 1] = np.clip(bboxes_[:, 1], 0, img.shape[0])
            bboxes_[:, 3] = np.clip(bboxes_[:, 3], 0, img.shape[0])

            data['img'] = img
            data['bboxes'] = bboxes_
            pass
        return data
    pass

class RandomRotate:
    def __init__(self, p=0.5):
        self.p = p
        self.transform = iaa.Rot90((1, 3))

    def __call__(self, data):
        if random.random() <= self.p:
            img = data['img']
            bboxes = data['bboxes']

            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                for box in bboxes
            ], shape=img.shape)
            image_aug, bbs_aug = self.transform(image=img, bounding_boxes=bbs)
            bboxes_ = np.zeros_like(bboxes)
            for i, box in enumerate(bbs_aug.bounding_boxes):
                bboxes_[i, 0] = box.x1
                bboxes_[i, 1] = box.y1
                bboxes_[i, 2] = box.x2
                bboxes_[i, 3] = box.y2
                pass

            data['img'] = image_aug
            data['bboxes'] = bboxes_
            pass
        return data
    pass

class RandomAffine:
    def __init__(self, p=0.5):
        self.p = p
        self.transform = iaa.Affine(shear=(-10, 10))

    def __call__(self, data):
        if random.random() <= self.p:
            img = data['img']
            bboxes = data['bboxes']

            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                for box in bboxes
            ], shape=img.shape)
            image_aug, bbs_aug = self.transform(image=img, bounding_boxes=bbs)    

            bboxes_ = np.zeros_like(bboxes)
            for i, box in enumerate(bbs_aug.bounding_boxes):
                bboxes_[i, 0] = box.x1
                bboxes_[i, 1] = box.y1
                bboxes_[i, 2] = box.x2
                bboxes_[i, 3] = box.y2
                pass

            data['img'] = image_aug
            data['bboxes'] = bboxes_
            pass

        return data
    pass


class RandomCrop:
    def __init__(self, p=0.5, min_size=0.5):
        self.p = p
        self.min_size = min_size
        pass

    def __call__(self, data):
        img = data['img']
        bboxes = data['bboxes']
        cls = data['cls']

        if random.random() < self.p:
            h, w = img.shape[:2]
            
            # choose x
            xrange = int(w * (1 - self.min_size))
            x1 = random.randint(0, xrange)
            x2 = random.randint(x1 + int(w * self.min_size), w)
            # choose y
            yrange = int(h * (1 - self.min_size))
            y1 = random.randint(0, yrange)
            y2 = random.randint(y1 + int(h * self.min_size), h)

            # crop
            img = img[y1:y2, x1:x2].copy()

            # remove boxes outside the crop
            mask = (bboxes[:, 0] < x2) & (bboxes[:, 2] > x1) & (bboxes[:, 1] < y2) & (bboxes[:, 3] > y1)
            bboxes = bboxes[mask]
            cls = cls[mask]

            # adjust boxes
            bboxes_ = bboxes.copy()
            bboxes_[:, 0] = np.clip(bboxes_[:, 0] - x1, 0, img.shape[1])
            bboxes_[:, 1] = np.clip(bboxes_[:, 1] - y1, 0, img.shape[0])
            bboxes_[:, 2] = np.clip(bboxes_[:, 2] - x1, 0, img.shape[1])
            bboxes_[:, 3] = np.clip(bboxes_[:, 3] - y1, 0, img.shape[0])
            bboxes = bboxes_
            pass

        data['img'] = img
        data['bboxes'] = bboxes
        data['cls'] = cls

        return data


class RandomNoise:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() <= self.p:
            img = data['img']

            img = img.astype(np.float32)
            noise = np.random.normal(0, 0.02, img.shape) * 255
            img = img + noise
            img = np.clip(img, 0, 255).astype(np.uint8)

            data['img'] = img
        return data
    pass


class RandomBlur:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        img = data['img']

        if isinstance(img, Image.Image):
            img = np.array(img)
            pass

        if np.random.rand() < self.p:
            ksize = np.random.randint(3, 7, 2) * 2 + 1
            img = cv2.GaussianBlur(img, ksize, 0)
            pass

        data['img'] = img
        return data
    pass


class RandomBrightness:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() <= self.p:
            img = data['img']

            if isinstance(img, Image.Image):
                img = np.array(img)
                pass

            if np.random.rand() < self.p:
                alpha = np.random.uniform(0.9, 1.1)
                img = cv2.convertScaleAbs(img, alpha=alpha)
                pass

            data['img'] = img
            pass
        return data
    pass


class RandomGrayScale:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() <= self.p:
            img = data['img']
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            data['img'] = img
            pass
        
        return data
    pass

class RandomResize:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() <= self.p:
            img = data['img']
            bboxes = data['bboxes']

            scale = np.random.uniform(0.5, 1.5)
            new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            bboxes = bboxes * scale

            data['img'] = img
            data['bboxes'] = bboxes
            pass

        return data
    pass


class RandomPerspective:
    def __init__(self, p=0.5):
        self.p = p
        self.transform = iaa.PerspectiveTransform(scale=(0.0, 0.1))

    def __call__(self, data):
        if random.random() <= self.p:
            img = data['img']
            bboxes = data['bboxes']

            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                for box in bboxes
            ], shape=img.shape)

            image_aug, bbs_aug = self.transform(image=img, bounding_boxes=bbs)

            data['img'] = image_aug
            bboxes_ = np.zeros_like(bboxes)
            for i, box in enumerate(bbs_aug.bounding_boxes):
                bboxes_[i, 0] = box.x1
                bboxes_[i, 1] = box.y1
                bboxes_[i, 2] = box.x2
                bboxes_[i, 3] = box.y2
                pass
            data['bboxes'] = bboxes_
            
        return data
    pass


class FilterSmallBox:
    def __init__(self, min_size=4):
        self.min_size = min_size

    def __call__(self, data):
        img = data['img']
        bboxes = data['bboxes']
        cls = data['cls']

        width = bboxes[:, 2] - bboxes[:, 0]
        height = bboxes[:, 3] - bboxes[:, 1]

        mask = (width >= self.min_size) & (height >= self.min_size)
        bboxes = bboxes[mask]
        cls = cls[mask]

        data['img'] = img
        data['bboxes'] = bboxes
        data['cls'] = cls

        return data
    pass

class RandomShuffleChannel:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() <= self.p:
            img = data['img']
            channels = [0, 1, 2]
            np.random.shuffle(channels)
            img = img[:, :, channels]

            data['img'] = img
        return data
    pass


class ToNumpy:
    def __call__(self, data):
        data['img'] = ensure_cv2(data['img'])
        return data
    pass

class Resize:
    def __init__(self, max_size=640):
        self.max_size = max_size

    def __call__(self, data):
        img = data['img']
        bboxes = data['bboxes']

        h, w = img.shape[:2]

        max_hw = max(h, w)
        scale = self.max_size / max_hw
        if max_hw > self.max_size:
            scale = self.max_size / max_hw
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            bboxes = bboxes * scale
        else:
            new_h, new_w = h, w
            pass

        data['img'] = img.copy()
        data['bboxes'] = bboxes

        return data

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
    
    def extend(self, transforms):
        if isinstance(transforms, list):
            self.transforms.extend(transforms)
            pass
        else:
            raise TypeError(f"Unsupported type: {type(transforms)}")
        pass


class ToTensor:
    def __call__(self, data):
        img = data['img']
        img = VF.to_tensor(img)

        data['img'] = img
        return data
    pass


class BGR2RGB:
    def __call__(self, data):
        img = data['img']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data['img'] = img
        return data
    pass


class Format:
    def __init__(self, input_size=(640, 640)):
        self.input_size = input_size

    def __call__(self, data):
        img = data['img']
        bboxes = data['bboxes']

        width, height = img.shape[1], img.shape[0]

        # convert xyxy to cxcywh
        bboxes_xyxy = bboxes.copy()
        bboxes_cxcywh_norm = box_utils.normalize(box_utils.xyxy2cxcywh(bboxes_xyxy), height, width)

        if isinstance(bboxes_xyxy, np.ndarray):
            bboxes_xyxy = torch.from_numpy(bboxes_xyxy).float()
            pass
        if isinstance(bboxes_cxcywh_norm, np.ndarray):
            bboxes_cxcywh_norm = torch.from_numpy(bboxes_cxcywh_norm).float()
            pass
        
        data['bboxes_xyxy'] = bboxes_xyxy
        data['bboxes_cxcywh_norm'] = bboxes_cxcywh_norm
        data['bboxes'] = bboxes_xyxy
        data['cls'] = torch.from_numpy(data['cls']).long()
        data['img'] = img
        data['batch_idx'] = torch.zeros((bboxes.shape[0],), dtype=torch.int64)
        return data

