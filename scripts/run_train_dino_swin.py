import argparse
import os
from hq_det.tools import train_dino_swin

DEFAULT_CONFIG = 'dino_swin_tiny_224_4scale_12ep'


def parse_args():
    parser = argparse.ArgumentParser(description='Train DINO-Swin via hq_det')
    parser.add_argument('data_path', type=str, help='COCO-style dataset root containing train/ and valid/')
    parser.add_argument('load_checkpoint', type=str, nargs='?', default='',
                        help='checkpoint path (HQ ckpt or raw backbone .pth). Empty = train from scratch')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                        help=f'dino-swin config name (default: {DEFAULT_CONFIG}). '
                             f'Available: dino_swin_tiny_224_4scale_12ep, dino_swin_small_224_4scale_12ep, '
                             f'dino_swin_base_384_4scale_12ep, dino_swin_large_224_4scale_12ep, '
                             f'dino_swin_large_384_4scale_12ep, dino_swin_large_384_4scale_36ep, '
                             f'dino_swin_large_384_5scale_12ep, dino_swin_large_384_5scale_36ep')
    parser.add_argument('--output', type=str, default='output', help='output dir')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--gradient-update-interval', type=int, default=2)
    parser.add_argument('--lr-backbone-mult', type=float, default=0.1)
    parser.add_argument('--num-data-workers', type=int, default=12)
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    parser.add_argument('--checkpoint-name', type=str, default='ckpt.pth')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    output_path = os.path.join(args.output, args.config)
    train_dino_swin.run(
        data_path=args.data_path,
        output_path=output_path,
        num_epoches=args.epochs,
        lr0=args.lr,
        load_checkpoint=args.load_checkpoint,
        eval_class_names=[],
        batch_size=args.batch_size,
        image_size=args.image_size,
        gradient_update_interval=args.gradient_update_interval,
        lr_backbone_mult=args.lr_backbone_mult,
        num_data_workers=args.num_data_workers,
        devices=args.devices,
        checkpoint_name=args.checkpoint_name,
        config_name=args.config,
    )
