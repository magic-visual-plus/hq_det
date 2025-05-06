import mmengine.config
import mmengine.runner
from mmdet.configs.dino import dino_4scale_r50_8xb2_12e_coco as dino_config
import sys
import os


if __name__ == '__main__':
    run_config = mmengine.config.Config.fromfile(dino_config.__file__)

    run_config.data_root = sys.argv[1]
    run_config.train_dataloader.dataset.data_root = run_config.data_root
    run_config.train_dataloader.dataset.ann_file = 'annotations/instances_val2017.json'
    run_config.train_dataloader.dataset.data_prefix = {'img': 'val2017/'}
    run_config.val_dataloader.dataset.data_root = run_config.data_root
    run_config.val_evaluator.ann_file = os.path.join(
        run_config.data_root, 'annotations/instances_val2017.json')
    run_config.test_dataloader = None
    run_config.test_cfg = None
    run_config.test_evaluator = None
    run_config.work_dir = sys.argv[2]

    print(run_config)
    runner = mmengine.runner.Runner.from_cfg(run_config)

    runner.train()
    pass