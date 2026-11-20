#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Torch eager reference service on port 18900 — Swin-L batch=1.

Uses hq_dino.HQDINO full forward (same loader as the TRT service's head),
so it loads the ORIGINAL mmdet ckpt directly — no ckpt conversion needed.
HTTP API identical to trt_service_18901.py for fair comparison.
"""
import base64, json, os, sys, time, uuid
import numpy as np
import cv2
import torch

BASE = os.environ.get("TRT_BASE", r"D:\magic_fox_ai\resources\backend")
MODEL_DIR = os.path.join(BASE, "trt_models")
CKPT = os.path.join(MODEL_DIR, "daika_lungu_epoch12.pth")
sys.path.insert(0, BASE)

# same export patches as TRT service (eager path needs them for MDA fallback)
import mmcv.ops.multi_scale_deform_attn as msda
class _PyMSDAFn:
    @staticmethod
    def apply(value, value_spatial_shapes, level_start_index,
              sampling_locations, attention_weights, im2col_step=None):
        return msda.multi_scale_deformable_attn_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights)
msda.MultiScaleDeformableAttnFunction = _PyMSDAFn
import fairscale.nn.checkpoint.checkpoint_activations as _fsc
def _passthrough(original_forward, weak_self, offload_to_cpu, *args, **kwargs):
    return original_forward(weak_self(), *args, **kwargs)
_fsc._checkpointed_forward = _passthrough

from hq_det.models.dino2 import hq_dino

print("loading torch model...", flush=True)
t0 = time.time()
model = hq_dino.HQDINO(model=CKPT)
model.load_model(CKPT)
model.eval()
model.to("cuda:0")
mm = model.model
for m in mm.modules():
    if hasattr(m, "with_cp") and getattr(m, "with_cp"):
        m.with_cp = False
print(f"MODEL_READY in {time.time()-t0:.0f}s", flush=True)

def infer_bgr_bytes(img_bytes, conf):
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
    oh, ow = im.shape[:2]
    t0 = time.time()
    results = model.predict([im], bgr=True, confidence=conf)
    t1 = time.time()
    r = results[0]
    dets = []
    import json as _json
    _classes = _json.load(open(os.path.join(MODEL_DIR, "classes.json"), encoding="utf-8"))
    for j in range(len(r.scores)):
        x1, y1, x2, y2 = [float(v) for v in r.bboxes[j]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, ow), min(y2, oh)
        cls_id = int(r.cls[j])
        name = _classes[cls_id] if cls_id < len(_classes) else str(cls_id)
        dets.append({
            "name": name,
            "confidence": round(float(r.scores[j]), 5),
            "type": "bounding_box",
            "points": [{"x": round(x1, 1), "y": round(y1, 1),
                        "w": round(x2 - x1, 1), "h": round(y2 - y1, 1)}],
        })
    return dets, {"torch_ms": round((t1 - t0) * 1000, 1)}

def _decode_image_item(data, stype):
    if stype == "network_b64" or "," in data[:64]:
        if "," in data[:64]:
            data = data.split(",", 1)[1]
        return base64.b64decode(data)
    with open(data, "rb") as f:
        return f.read()

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._send(200, {"status": "ok", "model": "swin_l_torch_eager_b1", "port": 18900})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return
        t_start = time.time()
        try:
            threshold = float(req.get("threshold", 0.3))
            request_id = req.get("request_id") or str(uuid.uuid4())
            if isinstance(req.get("image_data"), str):
                img_bytes = _decode_image_item(req["image_data"],
                                               (req.get("source_type") or "network_b64").lower())
                dets, timing = infer_bgr_bytes(img_bytes, threshold)
                self._send(200, {
                    "request_id": request_id,
                    "model_name": "swin_l_torch_eager_b1",
                    "predictions": dets,
                    "extra_info": timing,
                    "cost_ms": round((time.time() - t_start) * 1000, 1),
                })
                return
            img_list = req.get("image_data") or []
            results = []
            for item in img_list:
                img_bytes = _decode_image_item(item.get("image_data", ""),
                                               (item.get("source_type") or "network_b64").lower())
                dets, timing = infer_bgr_bytes(img_bytes, threshold)
                results.append({"predictions": dets, "extra_info": timing})
            self._send(200, {
                "request_id": request_id,
                "model_name": "swin_l_torch_eager_b1",
                "results": results,
                "cost_ms": round((time.time() - t_start) * 1000, 1),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

if __name__ == "__main__":
    port = int(os.environ.get("TORCH_PORT", "18900"))
    print(f"SERVE_READY http://0.0.0.0:{port} ckpt={CKPT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
