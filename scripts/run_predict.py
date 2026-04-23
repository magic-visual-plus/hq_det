import sys
from hq_det.models.dino import hq_dino
from hq_det.models import rtdetr
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
import os
import torch
from hq_det import torch_utils
from ultralytics.utils import DEFAULT_CFG
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    
    model = hq_dino.HQDINO(model=sys.argv[1])
    # model = rtdetr.HQRTDETR(model=sys.argv[1])
    model.eval()
    
    model.to("cuda:0")

    filenames = os.listdir(input_path)

    filenames = [os.path.join(input_path, f) for f in filenames if f.endswith('.jpg') or f.endswith('.png')]

    for filename in tqdm(filenames):
        img = cv2.imread(filename)

        results = model.predict([img], bgr=True, confidence=0.3, max_size=1536)

        result = results[0]

        print(len(result.bboxes))
        for bbox in result.bboxes:
            img = cv2.rectangle(
                img.copy(),
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2
            )
            pass
        cv2.imwrite(os.path.join(output_path, os.path.basename(filename)), img)
        pass
    pass