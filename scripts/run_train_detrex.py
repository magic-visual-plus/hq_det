"""
运行 Detrex 模型训练的脚本
"""

import sys
import os
from hq_det.tools import train_detrex


if __name__ == '__main__':
    devices = list(range(int(os.getenv("GPU_NUM", "1"))))
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
    config_path = sys.argv[3] if len(sys.argv) > 3 else None
    project = sys.argv[4] if len(sys.argv) > 4 else os.environ.get('PROJECT', 'dino')
    model_size = sys.argv[5] if len(sys.argv) > 5 else os.environ.get('MODEL_SIZE', 'r50')
    model_name = sys.argv[6] if len(sys.argv) > 6 else os.environ.get('MODEL_NAME', None)
    
    if data_path is None:
        print("Usage: python run_train_detrex.py <data_path> [checkpoint_path] [config_path] [project] [model_size] [model_name]")
        print("  project: 'dino', 'detr', 'deformable_detr', etc. (default: 'dino')")
        print("  model_size:")
        print("    For dino with ResNet: 'r50', 'r101' (default: 'r50')")
        print("    For dino with Swin: 'swin_tiny_224', 'swin_small_224', 'swin_large_384', etc.")
        print("    For dino with ConvNext: 'convnext_tiny_384', 'convnext_small_384', etc.")
        print("  model_name: Full model config name, e.g., 'dino_r50_4scale_12ep' (overrides project and model_size)")
        sys.exit(1)
    
    train_detrex.run(
        data_path=data_path,
        output_path='output',
        num_epoches=int(os.environ.get('NUM_EPOCHES', '12')),
        batch_size=int(os.environ.get('BATCH_SIZE', '4')),
        lr0=1e-4,
        warmup_epochs=0,
        load_checkpoint=checkpoint_path,
        eval_class_names=[],
        devices=devices,
        config_path=config_path,
        project=project,
        model_size=model_size,
        model_name=model_name,
        image_size=800,
    )
    pass


