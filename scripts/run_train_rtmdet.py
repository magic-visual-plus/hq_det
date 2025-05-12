from hq_det.monitor.logio import LogRedirector


def run_train_rtmdet(args, class_names):
    from hq_det.trainer import HQTrainerArguments
    from hq_det.tools.train_rtmdet import MyTrainer
    
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
    

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RTMDet model')
    parser.add_argument('--data_path', '-d', type=str, required=True, help='Path to the training data')
    parser.add_argument('--output_path', '-o', type=str, required=True, help='Path to save outputs')
    parser.add_argument('--load_checkpoint', '-c', type=str, default=None, help='Path to load checkpoint')
    parser.add_argument('--num_epoches', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--warmup_epochs', '-w', type=int, default=2, help='Number of warmup epochs')
    parser.add_argument('--num_data_workers', '-j', type=int, default=8, help='Number of data workers')
    parser.add_argument('--lr0', '--initial-lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_min', '--min-lr', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--device', '--dev', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--checkpoint_interval', '--ckpt-int', type=int, default=-1, help='Checkpoint interval')
    parser.add_argument('--image_size', '-s', type=int, default=1024, help='Image size')
    parser.add_argument('--log_file', '-l', type=str, default=None, help='Path to save log file')
    parser.add_argument('--eval_class_names', type=str, default=None, help='Class names for evaluation')

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_args()
    if args.log_file:
        log_redirector = LogRedirector(args.log_file)
    if args.eval_class_names is None:
        class_names = [
            '划伤', '划痕', '压痕', '吊紧', '异物外漏', '折痕', '抛线', '拼接间隙', 
            '烫伤', '爆针线', '破损', ' 碰伤', '线头', '脏污', '褶皱(贯穿)', 
            '褶皱（轻度）', '褶皱（重度）', '重跳针'
        ]
    else:
        class_names = args.eval_class_names.split(',')
    run_train_rtmdet(args, class_names)
    