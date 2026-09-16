"""修改 TRAINING 参数后一键训练，或传入 DATA_ROOT CHECKPOINT 两个参数。

python scripts/run_train_pointdino.py
python scripts/run_train_pointdino.py data/my_points checkpoints/init.pth
旧的 config.py / --cfg-options 命令行方式仍可用，--help 查看。
"""
from pathlib import Path
import sys


from hq_det.tools import pointdino

import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

REPO_ROOT = Path(__file__).resolve().parents[1]

# 和 run_train_dino.py 相同，在此集中配置一次训练。
# 更换数据集主要修改 data_path、ann_file、image_dir；参见 POINTDINO_DATA_FORMAT.md。
TRAINING = dict(
    data_path=REPO_ROOT / 'data' / 'shanghaitech_part_b',
    output_path=REPO_ROOT / 'output' / 'pointdino_2',
    load_checkpoint=REPO_ROOT.parent / 'point_dino_stage2_step2_init.pth',
    train_ann_file='annotations/train.json',
    val_ann_file='annotations/val.json',
    train_image_dir='images/train',
    val_image_dir='images/val',
    class_names=None,              # 自动读取 JSON metainfo.classes，各 split 顺序相同
    num_epoches=100,
    lr0=1e-4,
    lr_backbone_mult=0.1,
    batch_size=1,
    eval_batch_size=1,
    image_size=None,               # 默认原图，不 resize；显式 (宽, 高) 或整数才缩放
    gradient_update_interval=1,
    num_data_workers=2,             # 调试数据时可设 0
    devices=[0],                  # None=默认设备；[]=CPU；[0]=指定 GPU
    config_path=None,              # None=Stage4 full-map；也可填 local 配置路径
    resume=False,                  # True 恢复 optimizer/epoch；初始化权重用 False
    score_threshold=0.5,
    lr_milestones=(70, 90),
    distance_thresholds=(5., 10.),  # 原图像素；最佳 checkpoint 按 f1@10px 保存
    cfg_options={
        'model.point_fidt_head.fidt_loss_weight': 0.0,
        'model.point_fidt_head.debug': False,},            
)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg.startswith('--') for arg in args) or (args and args[0].endswith('.py')):
        return pointdino.train_main(args)
    if len(args) > 2:
        raise ValueError('Expected at most DATA_ROOT CHECKPOINT, or use --help.')
    settings = dict(TRAINING)
    if args:
        settings['data_path'] = args[0]
    if len(args) == 2:
        settings['load_checkpoint'] = args[1]
    return pointdino.run(**settings)


if __name__ == '__main__':
    main()
