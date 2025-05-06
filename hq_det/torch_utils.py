import torch
from mmdet.structures import DetDataSample


def batch_to_device(batch, device):
    """
    Move a batch of data to the specified device (GPU or CPU).
    
    Args:
        batch (dict): A dictionary containing the batch data.
        device (torch.device): The device to move the data to.
    
    Returns:
        dict: The batch data moved to the specified device.
    """

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [batch_to_device(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {key: batch_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, DetDataSample):
        return batch.to(device)
    else:
        return batch


def pad_image(img, bboxes, target_shape, pad_value=0.44):
    """
    Pad an image to the target shape with a specified padding value.
    
    Args:
        img (torch.Tensor): The input image tensor.
        target_shape (tuple): The target shape (height, width) for padding.
        pad_value (float): The value to use for padding.
    
    Returns:
        torch.Tensor: The padded image tensor.
    """
    h, w = img.shape[1:]
    if h == target_shape[0] and w == target_shape[1]:
        return img, bboxes
    
    pad_left = 0
    pad_top = 0
    pad_right = target_shape[1] - w
    pad_bottom = target_shape[0] - h
    img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

    bboxes_ = bboxes.clone()
    bboxes_[:, 0] = bboxes[:, 0] * w / target_shape[1]
    bboxes_[:, 1] = bboxes[:, 1] * h / target_shape[0]
    bboxes_[:, 2] = bboxes[:, 2] * w / target_shape[1]
    bboxes_[:, 3] = bboxes[:, 3] * h / target_shape[0]
    
    return img, bboxes