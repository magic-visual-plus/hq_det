from ultralytics import YOLO
import sys


if __name__ == '__main__':
    model = YOLO(sys.argv[2])
    model.train(
        data=sys.argv[1],
        amp=False,
        optimizer='AdamW',
        lr0=1e-4,
        lrf=0.01,
        epochs=50,
        batch=4,
        imgsz=1024
    )
    pass