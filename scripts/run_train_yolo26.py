import argparse
import json
import os

from hq_det.tools import train_yolo26


def parse_devices(value):
    if value is None:
        return list(range(int(os.getenv("GPU_NUM", "1"))))
    return [int(item) for item in value.split(",") if item.strip()]


def get_args():
    parser = argparse.ArgumentParser(description="Train YOLO26 with HQ-DET")
    parser.add_argument("--data_path", "-d", required=True, help="Roboflow/COCO dataset path")
    parser.add_argument("--output_path", "-o", default="output", help="Output directory")
    parser.add_argument("--scale", "-m", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument(
        "--load_checkpoint",
        "-c",
        default=None,
        help="YOLO26 .pt/.pth checkpoint or model YAML",
    )
    parser.add_argument("--scratch", action="store_true", help="Build yolo26*.yaml instead of yolo26*.pt")
    parser.add_argument("--p2", action="store_true", help="Use YOLO26 P2 yaml/weights name")
    parser.add_argument("--num_epoches", "-e", type=int, default=100)
    parser.add_argument("--warmup_epochs", "-w", type=float, default=3.0)
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--image_size", "-s", type=int, default=1024)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument(
        "--optimizer",
        default="auto",
        choices=["auto", "MuSGD"],
        help="Frozen 8.4.7 recipe: auto always resolves to MuSGD",
    )
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--nbs", type=int, default=64)
    parser.add_argument("--warmup_momentum", type=float, default=0.8)
    parser.add_argument("--warmup_bias_lr", type=float, default=0.1)
    parser.add_argument("--cos_lr", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--non_deterministic", action="store_true")
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--dfl", type=float, default=1.5)
    parser.add_argument("--num_data_workers", "-j", type=int, default=8)
    parser.add_argument("--devices", default=None, help="Comma separated device ids, e.g. 0,1")
    parser.add_argument("--gradient_update_interval", type=int, default=1)
    parser.add_argument("--checkpoint_name", default="ckpt.pth")
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--eval_class_names", default=None, help="Comma separated class names")
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Rejected by the exact 8.4.7 recipe; retained for CLI compatibility",
    )
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_tau", type=float, default=2000.0)
    parser.add_argument("--amp", dest="enable_amp", action="store_true", help="Enable AMP")
    parser.add_argument("--no_amp", dest="enable_amp", action="store_false", help="Disable AMP")
    parser.set_defaults(enable_amp=True)
    parser.add_argument("--close_mosaic", type=int, default=10)
    parser.add_argument("--hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv_s", type=float, default=0.7)
    parser.add_argument("--hsv_v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale_gain", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--bgr", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--copy_paste", type=float, default=0.0)
    parser.add_argument("--copy_paste_mode", choices=["flip", "mixup"], default="flip")
    parser.add_argument(
        "--yolo_overrides",
        default="{}",
        help="JSON object with additional frozen recipe settings",
    )
    parser.add_argument("--augment_proba", type=float, default=0.3)
    parser.add_argument("--augment_split_size", type=int, default=-1)
    parser.add_argument("--augment_split_proba", type=float, default=0.5)
    parser.add_argument("--augment_foreground_path", default="")
    parser.add_argument("--augment_foreground_proba", type=float, default=0.8)
    parser.add_argument("--augment_force_resize", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_class_names = (
        None if args.eval_class_names is None else args.eval_class_names.split(",")
    )

    train_yolo26.run(
        data_path=args.data_path,
        output_path=args.output_path,
        scale=args.scale,
        load_checkpoint=args.load_checkpoint,
        num_epoches=args.num_epoches,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr0=args.lr0,
        lr_min=args.lr_min,
        lrf=args.lrf,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nbs=args.nbs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        cos_lr=args.cos_lr,
        seed=args.seed,
        deterministic=not args.non_deterministic,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        eval_class_names=eval_class_names,
        devices=parse_devices(args.devices),
        num_data_workers=args.num_data_workers,
        gradient_update_interval=args.gradient_update_interval,
        checkpoint_name=args.checkpoint_name,
        checkpoint_interval=args.checkpoint_interval,
        scratch=args.scratch,
        p2=args.p2,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        ema_tau=args.ema_tau,
        enable_amp=args.enable_amp,
        close_mosaic=args.close_mosaic,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale_gain=args.scale_gain,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        bgr=args.bgr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        cutmix=args.cutmix,
        copy_paste=args.copy_paste,
        copy_paste_mode=args.copy_paste_mode,
        yolo_overrides=json.loads(args.yolo_overrides),
        augment_proba=args.augment_proba,
        augment_split_size=args.augment_split_size,
        augment_split_proba=args.augment_split_proba,
        augment_foreground_path=args.augment_foreground_path,
        augment_foreground_proba=args.augment_foreground_proba,
        augment_force_resize=args.augment_force_resize,
    )
