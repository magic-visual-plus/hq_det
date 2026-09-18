"""修改 TESTING 参数后一键测试，或传入 DATA_ROOT CHECKPOINT。

python scripts/test_pointdino.py
python scripts/test_pointdino.py data/my_points output/pointdino_2/best_model.pth
旧的 config.py CHECKPOINT / --cfg-options 命令行方式仍可用。
"""
from pathlib import Path
import sys

from hq_det.tools import pointdino, train_pointdino

REPO_ROOT = Path(__file__).resolve().parents[1]

TESTING = dict(
    data_path=REPO_ROOT / 'data' / 'pointdino',
    load_checkpoint=REPO_ROOT / 'output' / 'pointdino_2' / 'best_model.pth',
    output_path=REPO_ROOT / 'output' / 'pointdino_test',
    test_ann_file='valid/_annotations.coco.json',
    test_image_dir='valid',
    class_names=None,              # 必须与训练时类别顺序一致
    image_size=None,               # 默认原图；显式指定尺寸才启用 resize
    eval_batch_size=1,
    num_data_workers=2,
    devices=None,                  # None=默认设备；[]=CPU；[0]=指定 GPU
    config_path=None,              # 应与训练的配置相同
    score_threshold=0.5,
    distance_thresholds=(5., 10.),  # 原图像素
    cfg_options={},
)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg.startswith('--') for arg in args) or (args and args[0].endswith('.py')):
        return pointdino.evaluate_main(args)
    if len(args) > 2:
        raise ValueError('Expected at most DATA_ROOT CHECKPOINT, or use --help.')
    settings = dict(TESTING)
    if args:
        settings['data_path'] = args[0]
    if len(args) == 2:
        settings['load_checkpoint'] = args[1]
    return train_pointdino.test(**settings)


if __name__ == '__main__':
    main()
