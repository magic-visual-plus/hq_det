import sys
from hq_det.models.dino2 import hq_dino
from hq_det.models.patch_classfication import PatchClassificationModel
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
    
    model = hq_dino.HQDINO(
        model=sys.argv[1],
        config_name="dino-5scale_swin-l_8xb2-12e_coco_resize.py"
    )
    # model = PatchClassificationModel(model=sys.argv[1])
    model.load_model(sys.argv[1])
    model.eval()
    class_names = model.get_class_names()
    print(class_names)
    model.to("cuda:0")
    os.makedirs(output_path, exist_ok=True)

    filenames = os.listdir(input_path)

    filenames = [os.path.join(input_path, f) for f in filenames if f.endswith('.jpg') or f.endswith('.png')]

    for filename in tqdm(filenames):
        img = cv2.imread(filename)

        results = model.predict([img], bgr=True, confidence=0.05)
        print(f"{filename}: {results[0]}")
        result = results[0]

        for bbox, score, c in zip(result.bboxes, result.scores, result.cls):
            img = cv2.rectangle(
                img.copy(),
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2
            )

            img = cv2.putText(
                img,
                f"{score:.2f}",
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
            name = class_names[c]
            img = cv2.putText(
                img,
                str(c),
                (int(bbox[0]), int(bbox[1]) - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            pass

        if result.bboxes.shape[0] > 0:
            cv2.imwrite(os.path.join(output_path, os.path.basename(filename)), img)
            pass
        pass
    pass