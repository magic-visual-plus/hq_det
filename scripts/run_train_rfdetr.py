import os
import random
import numpy as np
import torch
from hq_det.monitor import LogRedirector
from hq_det.monitor import TrainingVisualizer
from hq_det.monitor import EmailSender 


def set_seed(seed=42):
    """设置随机种子以确保实验的可重复性
    
    Args:
        seed (int): 随机种子值，默认为42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed has been set to: {seed}")


def run_train_rfdetr(args, class_names):
    from hq_det.trainer import HQTrainerArguments
    from hq_det.tools.train_rfdetr import MyTrainer
    
    trainer = MyTrainer(
        HQTrainerArguments(
            data_path=args.data_path,   # 数据集路径
            num_epoches=args.num_epoches,  # 训练轮数
            warmup_epochs=args.warmup_epochs,  # 预热轮数
            num_data_workers=args.num_data_workers,  # 数据加载线程数
            lr0=args.lr0,  # 初始学习率
            lr_min=args.lr_min,  # 最小学习率
            lr_backbone_mult=0.1,
            batch_size=args.batch_size,  # 批量大小
            device=args.device,  # 设备
            checkpoint_path=args.output_path,  # 检查点路径
            output_path=args.output_path,  # 输出路径
            checkpoint_interval=args.checkpoint_interval,  # 检查点间隔
            image_size=args.image_size,  # 图像大小
            gradient_update_interval=args.gradient_update_interval,  # 梯度更新间隔
            max_grad_norm=0.1,  # 最大梯度范数
            model_argument={
                "model": args.load_checkpoint,  # 加载的模型路径
                "model_type": "base",  # 模型类型
                "lr_encoder": 1.5e-4,  # 编码器学习率
                "lr_component_decay": 0.7,  # 组件衰减率
            },
            eval_class_names=class_names,   # 评估类别名称
        )
    )

    trainer.run()

    return trainer
    

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RTMDet model')
    parser.add_argument('--data_path', '-d', type=str, required=True, help='Path to the training data')
    parser.add_argument('--output_path', '-o', type=str, required=True, help='Path to save outputs')
    parser.add_argument('--load_checkpoint', '-c', type=str, default=None, help='Path to load checkpoint')
    parser.add_argument('--num_epoches', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--warmup_epochs', '-w', type=int, default=2, help='Number of warmup epochs')
    parser.add_argument('--num_data_workers', '-j', type=int, default=8, help='Number of data workers')
    parser.add_argument('--lr0', '--initial-lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_min', '--min-lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--device', '--dev', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--checkpoint_interval', '--ckpt-int', type=int, default=-1, help='Checkpoint interval')
    parser.add_argument('--image_size', '-s', type=int, default=1024, help='Image size')
    parser.add_argument('--log_file', '-l', type=str, default=None, help='Path to save log file')
    parser.add_argument('--eval_class_names', type=str, default=None, help='Class names for evaluation')
    parser.add_argument('--experiment_info', type=str, default=None, help='Additional experiment information')
    parser.add_argument('--gradient_update_interval', '-g', type=int, default=1, help='Number of batches to accumulate gradients before updating')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (None for no seed)')

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_args()
    
    # 设置随机种子（如果提供）
    if args.seed is not None:
        set_seed(args.seed)
    
    if args.log_file is None:
        args.log_file = args.output_path + '/train.log'
    log_redirector = LogRedirector(args.log_file)
    if args.eval_class_names is None:
        class_names = []
    else:
        class_names = args.eval_class_names.split(',')
    
    trainer = run_train_rfdetr(args, class_names)

    csv_path = trainer.results_file
    pdf_path = args.output_path + '/results.pdf'    
    visualizer = TrainingVisualizer(input_file=csv_path, output_file=pdf_path)
    visualizer.load_data()
    visualizer.generate_report()

