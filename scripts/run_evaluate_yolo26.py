import argparse
import json

from hq_det.tools import train_yolo26


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO26 with the HQ-DET industrial evaluator"
    )
    parser.add_argument("--data_path", "-d", required=True, help="Dataset path")
    parser.add_argument("--model", "-c", required=True, help="YOLO26 .pt or .pth checkpoint")
    parser.add_argument("--output_path", "-o", default="output/yolo26_eval")
    parser.add_argument("--scale", "-m", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--p2", action="store_true")
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--image_size", "-s", type=int, default=1024)
    parser.add_argument("--num_data_workers", "-j", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--eval_class_names",
        default=None,
        help="Comma separated defect classes included by the industrial evaluator",
    )
    parser.add_argument(
        "--yolo_overrides",
        default="{}",
        help="JSON object with additional frozen recipe settings",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_class_names = (
        None
        if args.eval_class_names is None
        else args.eval_class_names.split(",")
    )
    _, metrics = train_yolo26.evaluate(
        data_path=args.data_path,
        model=args.model,
        output_path=args.output_path,
        scale=args.scale,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_data_workers=args.num_data_workers,
        device=args.device,
        eval_class_names=eval_class_names,
        p2=args.p2,
        yolo_overrides=json.loads(args.yolo_overrides),
    )
    print(metrics)
