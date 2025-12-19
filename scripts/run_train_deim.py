"""
运行 DEIM 模型训练的脚本
"""

import sys
import os
from hq_det.tools import train_deim


if __name__ == '__main__':
    devices = list(range(int(os.getenv("GPU_NUM", "1"))))
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
    config_path = sys.argv[3] if len(sys.argv) > 3 else None
    model_size = sys.argv[4] if len(sys.argv) > 4 else os.environ.get('MODEL_SIZE', 'l')
    series = sys.argv[5] if len(sys.argv) > 5 else os.environ.get('SERIES', 'dfine')
    model_type = sys.argv[6] if len(sys.argv) > 6 else os.environ.get('MODEL_TYPE', 'deim')
    
    if data_path is None:
        print("Usage: python run_train_deim.py <data_path> [checkpoint_path] [config_path] [model_size] [series] [model_type]")
        print("  model_size:")
        print("    For dfine series: 'n', 's', 'm', 'l', 'x' (default: 's')")
        print("    For rtdetrv2 series: 'r18vd', 'r34vd', 'r50vd', 'r50vd_m', 'r101vd' (default: 'r50vd')")
        print("  series: 'dfine' or 'rtdetrv2' (default: 'dfine')")
        print("  model_type: 'deim' or 'dfine'/'rtdetrv2' (default: 'deim')")
        sys.exit(1)
    
    train_deim.run(
        data_path=data_path,
        output_path='output',
        num_epoches=int(os.environ.get('NUM_EPOCHES', '70')),
        batch_size=int(os.environ.get('BATCH_SIZE', '6')),
        lr0=2e-4,
        warmup_epochs=2,
        load_checkpoint=checkpoint_path,
        eval_class_names=[],
        devices=devices,
        config_path=config_path,
        model_size=model_size,
        series=series,
        model_type=model_type,
        image_size=1024,
    )
    pass

