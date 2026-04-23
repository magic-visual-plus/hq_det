import sys
from hq_det.models.rtdetr.core import YAMLConfig
from hq_det.models.rtdetr.solver import TASKS
import os
import torch

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


if __name__ == '__main__':

    cfg = YAMLConfig(sys.argv[2])
    # cfg.yaml_cfg['num_classes'] = 32
    # cfg.yaml_cfg[cfg.yaml_cfg['RTDETR']['decoder']]['num_classes'] = cfg.num_classes
    data_path = sys.argv[1]
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'valid')
    train_image_path = train_path
    train_ann_file = os.path.join(train_path, '_annotations.coco.json')
    val_image_path = val_path
    val_ann_file = os.path.join(val_path, '_annotations.coco.json')
    cfg.yaml_cfg['num_classes'] = 23
    cfg.yaml_cfg['remap_mscoco_category'] = False
    cfg.yaml_cfg['train_dataloader']['dataset']['img_folder'] = train_image_path
    cfg.yaml_cfg['train_dataloader']['dataset']['ann_file'] = train_ann_file
    cfg.yaml_cfg['train_dataloader']['num_workers'] = 8
    cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'] = val_image_path
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = val_ann_file
    cfg.yaml_cfg['val_dataloader']['num_workers'] = 8
    cfg.yaml_cfg['output_dir'] = '/root/autodl-tmp/rtdetrv2_pytorch/output'
    torch.multiprocessing.set_sharing_strategy('file_system')
    # print(model.decoder.num_classes)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    solver.fit()
    pass