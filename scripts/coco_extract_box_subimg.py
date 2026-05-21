
import json
import os
import shutil
import sys
from tqdm import tqdm
import cv2
import numpy as np
from hq_det import coco_utils


def extract(input_path: str, output_path: str):
    coco_utils.extract_boxes(input_path, output_path)
    pass


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
    pass