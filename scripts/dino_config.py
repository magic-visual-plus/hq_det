from mmengine.config import read_base


with read_base():
    from mmdet.configs.dino.dino_4scale_r50_8xb2_12e_coco import *
    pass

data_root