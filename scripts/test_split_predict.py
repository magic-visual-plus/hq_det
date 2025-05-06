
import os
import sys
from hq_det.models import dino
import cv2
import numpy as np
import torch
from tqdm import tqdm
import time

def split_image(img, stride=1024, shift=20, max_split=3):
    max_size = stride * max_split - (max_split - 1) * shift
    h, w = img.shape[:2]

    if h > max_size or w > max_size:
        rate = max(h / max_size, w / max_size)
        img = cv2.resize(img, (int(w / rate), int(h / rate)))
        pass
    
    stride = stride - shift
    for i in range(0, img.shape[0], stride):
        if (i + shift) >= img.shape[0]:
            break
        for j in range(0, img.shape[1], stride):
            if (j + shift) >= img.shape[1]:
                break

            if i + stride > h:
                i = h - stride
            if j + stride > w:
                j = w - stride
            yield img[i:i + stride + shift, j:j + stride + shift]
            pass
        pass

    yield cv2.resize(img, (stride+20, stride+20))
    pass


if __name__ == "__main__":

    model = dino.HQDINO(model=sys.argv[1])
    image_path = sys.argv[2]

    img = cv2.imread(image_path)
    print(img.shape)
    device = "cuda:0"
    model.to(device)
    model.eval()
    batch_size = 4
    for i in tqdm(range(100)):
        imgs = list(split_image(img, stride=1024, shift=20, max_split=2))
        # for j in range(0, len(imgs), batch_size):
        #     imgs_batch = imgs[j:j + batch_size]
        #     model.predict(imgs_batch)
        #     pass
        start = time.time()
        model.predict(imgs)
        print("time:", time.time() - start)
        pass




