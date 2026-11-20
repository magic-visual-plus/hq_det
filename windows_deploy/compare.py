"""Scan pool for images with detections, then run consistency on those."""
import base64, json, time, urllib.request
import numpy as np

imgs = [l.strip() for l in open(r"C:\deploy\imglist.txt", encoding="utf-8") if l.strip()]
print("pool:", len(imgs))

def call(port, b64, rid):
    body = json.dumps({"request_id": rid, "model_id": "x",
        "image_data": [{"image_data": b64, "source_type": "network_b64", "file_name": rid + ".jpg"}],
        "threshold": 0.3, "extra": {}}).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/api/predict", data=body,
        headers={"Content-Type": "application/json"}), timeout=300)
    return json.loads(r.read())

def iou(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1+aw, bx1+bw), min(ay1+ah, by1+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    u = aw*ah + bw*bh - inter
    return inter/u if u > 0 else 0

# scan up to 40 imgs, keep those with detections
with_det = []
for p in imgs[:40]:
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    try:
        resp = call(18901, b64, "scan")
        n = len(resp["results"][0]["predictions"])
    except Exception:
        continue
    if n > 0:
        with_det.append((p, n))
        print("HAS_DET", p.split("\\")[-1][:40], "n=", n, flush=True)
    if len(with_det) >= 10:
        break

print(f"\nfound {len(with_det)} imgs with detections, running consistency...")

lat_trt, lat_torch, eng_ms, tch_ms = [], [], [], []
s_trt, s_tch, ious, cnt_diff = [], [], [], []
for p, _ in with_det:
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    rt = call(18901, b64, "cmp1")
    rg = call(18900, b64, "cmp2")
    lat_trt.append(rt["cost_ms"]); lat_torch.append(rg["cost_ms"])
    eng_ms.append(rt["results"][0]["extra_info"]["engine_ms"])
    tch_ms.append(rg["results"][0]["extra_info"]["torch_ms"])
    d1 = rt["results"][0]["predictions"]
    d2 = rg["results"][0]["predictions"]
    cnt_diff.append(len(d1) - len(d2))
    used = set()
    for a in d1:
        best, bj = 0, None
        for j, b in enumerate(d2):
            if j in used or b["name"] != a["name"]:
                continue
            pa, pb = a["points"][0], b["points"][0]
            v = iou((pa["x"], pa["y"], pa["w"], pa["h"]), (pb["x"], pb["y"], pb["w"], pb["h"]))
            if v > best: best, bj = v, j
        if best >= 0.5 and bj is not None:
            used.add(bj)
            s_trt.append(a["confidence"]); s_tch.append(d2[bj]["confidence"])
            ious.append(best)

a1, a2 = np.array(lat_trt), np.array(lat_torch)
e, t = np.array(eng_ms), np.array(tch_ms)
print(f"\n=== latency on defect imgs (n={len(a1)}) ===")
print(f"TRT   cost p50={np.percentile(a1,50):.0f} engine p50={np.percentile(e,50):.0f}")
print(f"torch cost p50={np.percentile(a2,50):.0f} forward p50={np.percentile(t,50):.0f}")
st, sg = np.array(s_trt), np.array(s_tch)
print(f"\n=== consistency ({len(cnt_diff)} imgs, {len(st)} matched pairs) ===")
print(f"count diff: mean={np.mean(np.abs(cnt_diff)):.2f} max={max(np.abs(cnt_diff)) if cnt_diff else 0}")
if len(st):
    print(f"score diff: mean={np.abs(st-sg).mean():.4f} max={np.abs(st-sg).max():.4f}")
    print(f"box IoU: min={min(ious):.3f} mean={np.mean(ious):.3f}")
print("\nSCAN_DONE")
