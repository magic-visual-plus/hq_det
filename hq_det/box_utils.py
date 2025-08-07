import numpy as np
import scipy.spatial.distance

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



def filter_invalid_boxes(boxes, cls, im_width, im_height):
    """
    Filter out invalid boxes that are outside the image boundaries.
    """

    boxes_ = boxes.copy()
    mask = (boxes_[:, 0] < im_width) & (boxes_[:, 1] < im_height) & \
              (boxes_[:, 2] > 0) & (boxes_[:, 3] > 0) & \
              (boxes_[:, 2] > boxes_[:, 0]) & (boxes_[:, 3] > boxes_[:, 1])
    boxes_ = boxes_[mask]
    boxes_[:, 0] = np.clip(boxes_[:, 0], 0, im_width)
    boxes_[:, 1] = np.clip(boxes_[:, 1], 0, im_height)
    boxes_[:, 2] = np.clip(boxes_[:, 2], 0, im_width)
    boxes_[:, 3] = np.clip(boxes_[:, 3], 0, im_height)
    cls_ = cls[mask]

    return boxes_, cls_


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

    iou_thr_different_cls= 0.6
    iou_thr_same_cls = 0.5
    while len(boxes) > 0:
        # get the first box
        box = boxes[0]
        keep.append(indices[0])

        # calculate the iou of the first box with the rest
        iou = iou_xyxy(box, boxes[1:])

        mask = (iou > iou_thr_same_cls) & (cls[0] == cls[1:])
        mask = mask | ((iou > iou_thr_different_cls) & (cls[0] != cls[1:]))

        mask = np.logical_not(mask)

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


def merge_two_boxes(box1, box2):
    """
    Merge two bounding boxes into one.
    Args:
        box1 (np.ndarray): First bounding box in (x1, y1, x2, y2) format.
        box2 (np.ndarray): Second bounding box in (x1, y1, x2, y2) format.
    Returns:
        np.ndarray: Merged bounding box in (x1, y1, x2, y2) format.
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def merge_nearby_boxes(boxes, cls, scores, area_thr=0.6):
    # merge nearby boxes
    # boxes: (N, 4)
    # cls: (N, )
    # scores: (N, )
    # return: (N, 4)

    ucls = np.unique(cls)
    boxes_ = []
    cls_ = []
    scores_ = []
    for c in ucls:
        cmask = cls == c
        boxes_c = boxes[cmask]
        scores_c = scores[cmask]
        cls_c = cls[cmask]
        if len(boxes_c) <= 1:
            boxes_.append(boxes_c)
            scores_.append(scores_c)
            cls_.append(cls_c)
            continue
        
        boxes_merged_before = boxes_c.copy()
        while True:
            boxes_merged_after, scores_c, cls_c = try_merge_nearby_boxes(boxes_merged_before, scores_c, cls_c, area_thr)
            if len(boxes_merged_after) == len(boxes_merged_before):
                boxes_.append(boxes_merged_after)
                scores_.append(scores_c)
                cls_.append(cls_c)
                break
            boxes_merged_before = boxes_merged_after.copy()
            pass
        pass
    boxes_ = np.concatenate(boxes_, axis=0)
    scores_ = np.concatenate(scores_, axis=0)
    cls_ = np.concatenate(cls_, axis=0)
    return boxes_, cls_, scores_


def try_merge_nearby_boxes(boxes, scores, cls, area_thr=0.6):
    if len(boxes) <= 1:
        return boxes, scores, cls

    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    dists = scipy.spatial.distance.cdist(centers, centers, metric='euclidean')
    dists = dists + np.eye(len(boxes)) * 1e10  # avoid self-distance
    near_index = np.argmin(dists, axis=1)

    for idx in range(len(boxes)):
        near_idx = near_index[idx]
        merged = merge_two_boxes(boxes[idx], boxes[near_idx])
        area1 = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])
        area2 = (boxes[near_idx][2] - boxes[near_idx][0]) * (boxes[near_idx][3] - boxes[near_idx][1])
        merged_area = (merged[2] - merged[0]) * (merged[3] - merged[1])
        if (area1 + area2) / merged_area > area_thr:
            # merge the two boxes
            new_box = merged
            new_score = (scores[idx] + scores[near_idx]) / 2
            new_cls = cls[idx]  # or cls[near_idx], they should be the

            boxes = np.delete(boxes, [idx, near_idx], axis=0)
            boxes = np.vstack((boxes, new_box))
            scores = np.delete(scores, [idx, near_idx])
            scores = np.append(scores, new_score)
            cls = np.delete(cls, [idx, near_idx])
            cls = np.append(cls, new_cls)
            return boxes, scores, cls
        pass

    return boxes, scores, cls
    pass

