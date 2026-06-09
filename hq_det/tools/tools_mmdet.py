import torch
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


def collate_fn(batch):

    new_batch = {}
    new_batch['inputs'] = [torch.permute(torch.from_numpy(b['img']), (2, 0, 1)).contiguous() for b in batch]
    new_batch['image_id'] = [b['image_id'] for b in batch]
    new_batch['bboxes_xyxy'] = torch.cat([b['bboxes_xyxy'] for b in batch], 0)
    new_batch['cls'] = torch.cat([b['cls'] for b in batch], 0)
    new_batch['batch_idx'] = torch.cat([b['batch_idx']+i for i, b in enumerate(batch)], 0)

    data_samples = []

    for i, b in enumerate(batch):
        data_sample = DetDataSample(metainfo={
            'img_shape': (b['img'].shape[0], b['img'].shape[1]),
        })
        gt_instance = InstanceData()
        gt_instance.bboxes = b['bboxes_xyxy']
        gt_instance.labels = b['cls']
        data_sample.gt_instances = gt_instance
        data_samples.append(data_sample)
        pass

    new_batch['data_samples'] = data_samples
    return new_batch