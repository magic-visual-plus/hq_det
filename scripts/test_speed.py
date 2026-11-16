import time
import sys
import cv2

from hq_det.models.gdino import hq_gdino
from hq_det.models.dino2 import hq_dino

if __name__ == "__main__":
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    batch_size = int(sys.argv[3])

    # model = hq_dino.HQDINO(model=sys.argv[1])
    model = hq_dino.HQDINO(model=sys.argv[1])
    # model = PatchClassificationModel(model=sys.argv[1])
    print(model.get_class_names())
    model.load_model(sys.argv[1])
    model.eval()
    model.to("cuda:0")
    start = time.time()
    img = cv2.imread(image_path)
    print(f"read image cost: {time.time() - start}")
    imgs = [img for _ in range(batch_size)]

    for i in range(100):
        start = time.time()
        res = model.predict(imgs, bgr=True, confidence=0.3,)
        print(time.time() - start)
        pass
    pass