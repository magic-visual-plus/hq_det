import sys
from hq_det.models import rtdetr
from hq_det.models.dino import hq_dino
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
from hq_det import split_utils
import os
import torch
from hq_det import torch_utils
from ultralytics.utils import DEFAULT_CFG
import cv2
from tqdm import tqdm
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from hq_det import box_utils


FONT_PATH= '/root/autodl-tmp/simsun.ttc'

def putTextChinese(img, text, position, font_size, font_color):
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    font = ImageFont.truetype(FONT_PATH, font_size)
    draw = ImageDraw.Draw(pil_im)

    draw.text(position, text, font=font, fill=font_color)

    cv2_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return cv2_im

if __name__ == '__main__':
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # model = dino.HQDINO(model=sys.argv[1])
    model = rtdetr.HQRTDETR(model=sys.argv[1])
    model.eval()
    
    model.to("cuda:0")

    filenames = os.listdir(input_path)

    filenames = [os.path.join(input_path, f) for f in filenames if f.endswith('.jpg') or f.endswith('.png')]

    for filename in tqdm(filenames):
        img = cv2.imread(filename)

        if '1749385768499_20240704044647583' in filename:
            x = 11
            pass
        
        start = time.time()
        result = split_utils.predict_split(model, img, 0.3, 2048, 0, 100, bgr=True, add_global=False, box_area_thr=0.0)
        result.bboxes, result.cls, result.scores = box_utils.merge_nearby_boxes(result.bboxes, result.cls, result.scores, area_thr=0.6)
        print('time:', time.time() - start)

        for i, bbox in enumerate(result.bboxes):
            img = cv2.rectangle(
                img.copy(),
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                3
            )
            name = model.get_class_names()[result.cls[i]]
            img = putTextChinese(
                img, name, (int(bbox[0]-30), int(bbox[1]-30)), 20, (255, 0, 0)
            )
            pass
        cv2.imwrite(os.path.join(output_path, os.path.basename(filename)), img)
        pass
    pass