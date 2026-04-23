"""
DEIM 模型训练启动脚本
"""

import sys
import os
from hq_det.tools import train_deim

def parse_arguments():
    """解析命令行参数和环境变量"""
    # 命令行参数
    data_folder = sys.argv[1] if len(sys.argv) > 1 else None
    pretrained = sys.argv[2] if len(sys.argv) > 2 else None
    custom_config = sys.argv[3] if len(sys.argv) > 3 else None
    size_option = sys.argv[4] if len(sys.argv) > 4 else os.environ.get('MODEL_SIZE', 's')
    series_option = sys.argv[5] if len(sys.argv) > 5 else os.environ.get('SERIES', 'rtdetrv2')
    type_option = sys.argv[6] if len(sys.argv) > 6 else os.environ.get('MODEL_TYPE', 'deim')
    
    if data_folder is None:
        print("用法: python run_train_deim.py <数据路径> [预训练权重路径] [配置文件路径] [模型尺寸] [系列] [模型类型]")
        print("示例: python run_train_deim.py /root/test")
        sys.exit(1)
    
    return data_folder, pretrained, custom_config, size_option, series_option, type_option

if __name__ == '__main__':
    # 获取GPU数量（默认为1）
    gpu_list = list(range(int(os.getenv("GPU_NUM", "1"))))
    
    # 解析参数
    data_dir, ckpt_path, yaml_path, model_scale, backbone_series, arch_type = parse_arguments()
    
    # 调用训练函数
    train_deim.run(
        data_path=data_dir,
        output_path='output',
        num_epoches= 70,
        batch_size=12,
        lr0=2e-4,
        warmup_epochs=2,
        load_checkpoint=ckpt_path,
        eval_class_names=['划伤', '划痕', '压痕', '吊紧', '异物外漏', '折痕', '抛线', '拼接间隙', '烫伤', '爆针线', '破损', ' 碰伤', '线头', '脏污', '褶皱(贯穿)', '褶皱（轻度）', '褶皱（重度）', '重跳针', '褶皱(贯穿)','脏污（彩色）', '脏污（颜料笔）', '褶皱(T型)'],
        devices=gpu_list,
        config_path=yaml_path,
        model_size=model_scale,
        series=backbone_series,
        model_type=arch_type,
        image_size=1024,
    )