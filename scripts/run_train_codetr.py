import time
import torch

from hq_det.monitor import LogRedirector
from hq_det.monitor import TrainingVisualizer
from hq_det.monitor import EmailSender 


def run_train_codetr(args, class_names):
    from hq_det.trainer import HQTrainerArguments
    from hq_det.tools.train_codetr import MyTrainer
    
    trainer = MyTrainer(
        HQTrainerArguments(
            data_path=args.data_path,
            num_epoches=args.num_epoches,
            warmup_epochs=args.warmup_epochs,
            num_data_workers=args.num_data_workers,
            lr0=args.lr0,
            lr_min=args.lr_min,
            batch_size=args.batch_size,
            device=args.device,
            checkpoint_path=args.output_path,
            output_path=args.output_path,
            checkpoint_interval=args.checkpoint_interval,
            image_size=args.image_size,
            model_argument={
                "model": args.load_checkpoint,
            },
            eval_class_names=class_names,
        )
    )

    trainer.run()

    return trainer
    

def get_args(use_kwargs=False, **kwargs):
    # Default values
    DEFAULT_NUM_EPOCHES = 100      # 迭代次数
    DEFAULT_WARMUP_EPOCHS = 2       # 预热迭代次数
    DEFAULT_NUM_DATA_WORKERS = 8    # 数据加载线程数
    DEFAULT_LR0 = 1e-3               # 初始学习率
    DEFAULT_LR_MIN = 5e-5            # 最小学习率
    DEFAULT_BATCH_SIZE = 4           # 批次大小
    DEFAULT_DEVICE = 'cuda:0'        # 设备
    DEFAULT_CHECKPOINT_INTERVAL = -1 # 检查点间隔
    DEFAULT_IMAGE_SIZE = 1024        # 图像大小

    if use_kwargs:
        class Args:
            def __init__(self, **kwargs):
                self.data_path = kwargs.get('data_path')
                self.output_path = kwargs.get('output_path')
                self.load_checkpoint = kwargs.get('load_checkpoint')
                self.num_epoches = kwargs.get('num_epoches', DEFAULT_NUM_EPOCHES)
                self.warmup_epochs = kwargs.get('warmup_epochs', DEFAULT_WARMUP_EPOCHS)
                self.num_data_workers = kwargs.get('num_data_workers', DEFAULT_NUM_DATA_WORKERS)
                self.lr0 = kwargs.get('lr0', DEFAULT_LR0)
                self.lr_min = kwargs.get('lr_min', DEFAULT_LR_MIN)
                self.batch_size = kwargs.get('batch_size', DEFAULT_BATCH_SIZE)
                self.device = kwargs.get('device', DEFAULT_DEVICE)
                self.checkpoint_interval = kwargs.get('checkpoint_interval', DEFAULT_CHECKPOINT_INTERVAL)
                self.image_size = kwargs.get('image_size', DEFAULT_IMAGE_SIZE)
                self.log_file = kwargs.get('log_file')
                self.eval_class_names = kwargs.get('eval_class_names')
                self.experiment_info = kwargs.get('experiment_info')
        
        return Args(**kwargs)
    
    import argparse
    parser = argparse.ArgumentParser(description='Train RTMDet model')
    parser.add_argument('--data_path', '-d', type=str, required=True, help='Path to the training data')
    parser.add_argument('--output_path', '-o', type=str, required=True, help='Path to save outputs')
    parser.add_argument('--load_checkpoint', '-c', type=str, default=None, help='Path to load checkpoint')
    parser.add_argument('--num_epoches', '-e', type=int, default=DEFAULT_NUM_EPOCHES, help='Number of epochs')
    parser.add_argument('--warmup_epochs', '-w', type=int, default=DEFAULT_WARMUP_EPOCHS, help='Number of warmup epochs')
    parser.add_argument('--num_data_workers', '-j', type=int, default=DEFAULT_NUM_DATA_WORKERS, help='Number of data workers')
    parser.add_argument('--lr0', '--initial-lr', type=float, default=DEFAULT_LR0, help='Initial learning rate')
    parser.add_argument('--lr_min', '--min-lr', type=float, default=DEFAULT_LR_MIN, help='Minimum learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--device', '--dev', type=str, default=DEFAULT_DEVICE, help='Device to use')
    parser.add_argument('--checkpoint_interval', '--ckpt-int', type=int, default=DEFAULT_CHECKPOINT_INTERVAL, help='Checkpoint interval')
    parser.add_argument('--image_size', '-s', type=int, default=DEFAULT_IMAGE_SIZE, help='Image size')
    parser.add_argument('--log_file', '-l', type=str, default=None, help='Path to save log file')
    parser.add_argument('--eval_class_names', type=str, default=None, help='Class names for evaluation')
    parser.add_argument('--experiment_info', type=str, default=None, help='Additional experiment information')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if args.log_file is None:
        args.log_file = args.output_path + '/train.log'
    log_redirector = LogRedirector(args.log_file)
    if args.eval_class_names is None:
        class_names = [
            '划伤', '划痕', '压痕', '吊紧', '异物外漏', '折痕', '抛线', '拼接间隙', 
            '烫伤', '爆针线', '破损', ' 碰伤', '线头', '脏污', '褶皱(贯穿)', 
            '褶皱（轻度）', '褶皱（重度）', '重跳针'
        ]
    else:
        class_names = args.eval_class_names.split(',')
    
    trainer = run_train_codetr(args, class_names)

    csv_path = trainer.results_file
    pdf_path = args.output_path + '/results.pdf'    
    visualizer = TrainingVisualizer(input_file=csv_path, output_file=pdf_path)
    visualizer.load_data()
    visualizer.generate_report()

    email_sender = EmailSender(
        sender_email='RookieEmail@163.com',
        sender_password='TFeLq9AKDdTjTsht'
    )
    email_sender.send_experiment_notification(
        receiver_email='jiangchongyang@digitalpredict.cn',
        experiment_name='CoDetr Training Results',
        attachments=[pdf_path, csv_path, args.log_file],
        additional_info=f"{args.experiment_info}\n"\
            f"PDF_PATH: {pdf_path}\n"\
            f"CSV_PATH: {csv_path}\n"\
            f"LOG_PATH: {args.log_file}"
    )