import sys, time, json
sys.path.insert(0, "/home/hq_det")
import numpy as np
import cv2
import torch

from hq_det.models.dino2 import hq_dino

ckpt = "/home/model.pth"
img_path = sys.argv[1] if len(sys.argv) > 1 else "/home/hq_det/hq_det/models/lwdetr/demo/000000496954.jpg"

model = hq_dino.HQDINO(model=ckpt)
model.load_model(ckpt)
model.eval()
model.to("cuda:0")

img = cv2.imread(img_path)
print("orig image:", img.shape)
# match predict(): resize to image_size bucket
ms = model.image_size
rate = ms / max(img.shape[0], img.shape[1])
img = cv2.resize(img, (int(img.shape[1]*rate), int(img.shape[0]*rate)))
print("resized:", img.shape, "image_size:", ms)

imgs = [img]
# warmup
for _ in range(15):
    res = model.predict(imgs, bgr=True, confidence=0.3)
times = []
for _ in range(50):
    t0 = time.time()
    res = model.predict(imgs, bgr=True, confidence=0.3)
    times.append((time.time()-t0)*1000)
times = sorted(times)
print(f"E2E_PREDICT_MS p50={times[25]:.1f} p90={times[45]:.1f} min={times[0]:.1f} max={times[-1]:.1f}")
print("n_dets:", len(res[0].bboxes), "sample_score:", float(res[0].scores.max()) if len(res[0].scores) else None)
# internal shapes for export design
with torch.no_grad():
    batch = model.imgs_to_batch(imgs)
    from hq_det import torch_utils
    batch = torch_utils.batch_to_device(batch, torch.device("cuda:0"))
    fwd = model.forward(batch)
    print("img_feats:", [tuple(f.shape) for f in fwd["img_feats"]])
    hi = fwd["head_inputs_dict"]
    for k, v in hi.items():
        print("head_inputs:", k, tuple(v.shape) if torch.is_tensor(v) else type(v))
    # timing breakdown
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        fwd = model.forward(batch)
    torch.cuda.synchronize()
    print(f"FORWARD_ONLY_MS {(time.time()-t0)*1000/20:.1f}")
    t0 = time.time()
    for _ in range(20):
        preds = model.postprocess(fwd, batch, 0.3)
    torch.cuda.synchronize()
    print(f"POSTPROCESS_MS {(time.time()-t0)*1000/20:.1f}")
    t0 = time.time()
    for _ in range(20):
        b = model.imgs_to_batch(imgs)
    print(f"PREPROC_CPU_MS {(time.time()-t0)*1000/20:.1f}")
print("BASELINE_DONE")
