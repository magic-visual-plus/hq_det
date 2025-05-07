import numpy as np

def xyxy2xywh(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.
    Args:
        boxes (np.ndarray): Bounding boxes in (x1, y1, x2, y2) format.
    Returns:
        np.ndarray: Bounding boxes in (x, y, w, h) format.
    """
    boxes_ = boxes.copy()
    if boxes_.ndim == 1:
        boxes_ = boxes_[None, :]
        squeeze = True
        pass
    else:
        squeeze = False
        pass

    x = boxes_[:, 0]
    y = boxes_[:, 1]
    w = boxes_[:, 2] - boxes_[:, 0]
    h = boxes_[:, 3] - boxes_[:, 1]
    boxes_ = np.stack((x, y, w, h), axis=1)
    if squeeze:
        boxes_ = boxes_[0, :]
        pass
    return boxes_


def xyxy2cxcywh(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.
    Args:
        boxes (np.ndarray): Bounding boxes in (x1, y1, x2, y2) format.
    Returns:
        np.ndarray: Bounding boxes in (cx, cy, w, h) format.
    """
    boxes_ = boxes.copy()
    if boxes.ndim == 1:
        boxes_ = boxes_[None, :]
        squeeze = True
        pass
    else:
        squeeze = False
        pass

    x = boxes_[:, 0]
    y = boxes_[:, 1]
    w = boxes_[:, 2] - boxes_[:, 0]
    h = boxes_[:, 3] - boxes_[:, 1]
    cx = x + w / 2
    cy = y + h / 2

    boxes_ = np.stack((cx, cy, w, h), axis=1)
    if squeeze:
        boxes_ = boxes_[0, :]
        pass

    return boxes_


def normalize(boxes, height, width):
    boxes_ = boxes.copy()
    if boxes_.ndim == 1:
        boxes_ = boxes_[None, :]
        squeeze = True
        pass
    else:
        squeeze = False
        pass

    boxes_[:, 0] = boxes_[:, 0] / width
    boxes_[:, 1] = boxes_[:, 1] / height
    boxes_[:, 2] = boxes_[:, 2] / width
    boxes_[:, 3] = boxes_[:, 3] / height

    boxes_ = np.clip(boxes_, 0, 1)

    boxes_[:, 2] = np.where(boxes_[:, 0] - boxes_[:, 2] * 0.5 < 0, boxes_[:, 0]*0.5, boxes_[:, 2])
    boxes_[:, 2] = np.where(boxes_[:, 0] + boxes_[:, 2] * 0.5 > 1, (1 - boxes_[:, 0]) * 0.5, boxes_[:, 2])
    boxes_[:, 3] = np.where(boxes_[:, 1] - boxes_[:, 3] * 0.5 < 0, boxes_[:, 1]*0.5, boxes_[:, 3])
    boxes_[:, 3] = np.where(boxes_[:, 1] + boxes_[:, 3] * 0.5 > 1, (1 - boxes_[:, 1]) * 0.5, boxes_[:, 3])

    if squeeze:
        boxes_ = boxes_[0, :]
        pass

    return boxes_


def cxcywh2xyxy(boxes):
    """
    Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    Args:
        boxes (np.ndarray): Bounding boxes in (cx, cy, w, h) format.
    Returns:
        np.ndarray: Bounding boxes in (x1, y1, x2, y2) format.
    """
    if boxes.ndim == 1:
        return np.array([boxes[0] - boxes[2] / 2, boxes[1] - boxes[3] / 2,
                         boxes[0] + boxes[2] / 2, boxes[1] + boxes[3] / 2], dtype=np.float32)
        pass

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    return np.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), axis=1)


def unnormalize(boxes, width, height):
    boxes_ = boxes.copy()
    if boxes_.ndim == 1:
        boxes_ = boxes_[None, :]
        squeeze = True
    else:
        squeeze = False
        pass

    boxes_[:, 0] = boxes_[:, 0] * width
    boxes_[:, 1] = boxes_[:, 1] * height
    boxes_[:, 2] = boxes_[:, 2] * width
    boxes_[:, 3] = boxes_[:, 3] * height

    if squeeze:
        boxes_ = boxes_[0, :]
        pass

    return boxes_


def iou_xyxy(box1, box2):
    x1 = np.maximum(box1[0], box2[:, 0])
    y1 = np.maximum(box1[1], box2[:, 1])
    x2 = np.minimum(box1[2], box2[:, 2])
    y2 = np.minimum(box1[3], box2[:, 3])
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    intersection = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou


def nms(boxes, cls, scores):
    # non_max_suppression

    # boxes: (N, 4)
    # cls: (N, )
    # scores: (N, )
    # return: (N, 4)

    # first, sort by scores
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    cls = cls[indices]
    scores = scores[indices]
    keep = []

    iou_thr = 0.5
    while len(boxes) > 0:
        # get the first box
        box = boxes[0]
        keep.append(indices[0])

        # calculate the iou of the first box with the rest
        iou = iou_xyxy(box, boxes[1:])
        # remove the boxes with iou > 0.5 and sample cls
        mask = np.logical_not((iou > iou_thr) & (cls[0] == cls[1:]))

        if mask.sum() == 0:
            # no more boxes to keep
            break

        boxes = boxes[1:][mask]
        cls = cls[1:][mask]
        scores = scores[1:][mask]
        indices = indices[1:][mask]
        pass

    return keep
    pass


    


