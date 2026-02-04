import sys
from hq_det.tools import train_dino_eva

if __name__ == '__main__':
    train_dino_eva.run(
        data_path=sys.argv[1],
        output_path='output',
        num_epoches = 50,
        lr0=1e-4,
        load_checkpoint=sys.argv[2],
        eval_class_names=[],
        batch_size=1,
        image_size=1024,
        gradient_update_interval=2,
        lr_backbone_mult=1,
    )