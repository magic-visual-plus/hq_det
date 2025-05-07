import sys
from hq_det.models import dino
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
from hq_det import split_utils
import os
import torch
from hq_det import torch_utils
from ultralytics.utils import DEFAULT_CFG
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    
    model = dino.HQDINO(model=sys.argv[1])
    model.eval()
    
    model.to("cuda:0")

    filenames = os.listdir(input_path)

    filenames = [os.path.join(input_path, f) for f in filenames if f.endswith('.jpg') or f.endswith('.png')]

    for filename in tqdm(filenames):
        img = cv2.imread(filename)

        if '49fbab' in filename:
            print('filename:', filename)
            pass

        result = split_utils.predict_split(model, img, 0.2, 1024, 20, 2)

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