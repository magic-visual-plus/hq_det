import sys
from hq_det.tools import train_dino_eva
from detectron2.config import LazyConfig

if __name__ == '__main__':
    cfg = LazyConfig.load('/root/autodl-tmp/hq_det/hq_det/models/dino_eva/configs/dino-eva-02/dino_eva_02_12ep.py')
    train_dino_eva.run(
        data_path='/root/autodl-tmp/dataset/gear_dataset_split_1106',
        output_path='/root/autodl-tmp/hq_det/output',
        num_epoches = 25,
        config_file = cfg,
        lr0=1e-4,
        load_checkpoint='/root/autodl-tmp/dino_eva_02_in21k_pretrain_vitdet_b_4attn_1024_lrd0p7_4scale_12ep.pth',
        eval_class_names=[],
        batch_size=1,
        image_size=1024,
        gradient_update_interval=16,
        lr_backbone_mult=1.0,
    )