#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TRT HTTP service for Swin-L (DINO) — model_sdk compatible.

POST /api/predict        ModelPredictParam   (image_data: str)
POST /api/batch_predict  ModelBatchPredictParam (image_data: List[ImageDataInfo])
GET  /health

Deployed on Windows (192.168.31.115:18901), engine fp16 batch=1.
"""
import base64, json, os, sys, time, uuid
import numpy as np
import cv2
import torch
import tensorrt as trt

BASE = os.environ.get("TRT_BASE", r"D:\magic_fox_ai\resources\backend")
MODEL_DIR = os.path.join(BASE, "trt_models")
PLAN = os.path.join(MODEL_DIR, "swin_l_b1_fp16.plan")
CLASSES = json.load(open(os.path.join(MODEL_DIR, "classes.json"), encoding="utf-8"))
H = W = 2048
CONF_DEFAULT = 0.3

# ---- export patches (same as offline pipeline; see TRT_DEPLOY.md pit 1/2) ----
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

from hq_det.models.dino2 import hq_dino  # noqa: E402

logger = trt.Logger(trt.Logger.WARNING)
with open(PLAN, "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()
names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
out_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
stream = torch.cuda.Stream()
print("ENGINE_READY", flush=True)

CKPT = os.path.join(MODEL_DIR, "daika_lungu_epoch12.pth")
model = hq_dino.HQDINO(model=CKPT)
model.load_model(CKPT)
model.eval()
model.to("cuda:0")
mm = model.model
for m in mm.modules():
    if hasattr(m, "with_cp") and getattr(m, "with_cp"):
        m.with_cp = False
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample

def make_ds():
    s = DetDataSample(metainfo={"img_shape": (H, W), "pad_shape": (H, W),
                                "batch_input_shape": (H, W), "scale_factor": 1.0})
    s.gt_instances = InstanceData()
    return [s]

def head_decode(hidden, refs, conf):
    if torch.is_tensor(refs):
        refs = list(torch.unbind(refs, dim=0))
    preds = mm.bbox_head.predict(hidden, refs, batch_data_samples=make_ds(), rescale=False)
    p = preds[0]
    mask = p.scores > conf
    return (p.bboxes[mask].detach().cpu().numpy(),
            p.scores[mask].detach().cpu().numpy(),
            p.labels[mask].detach().cpu().numpy())

def engine_run(img_t):
    ctx.set_tensor_address("img", img_t.data_ptr())
    outs = {}
    for n in out_names:
        shp = tuple(ctx.get_tensor_shape(n))
        outs[n] = torch.empty(shp, dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(n, outs[n].data_ptr())
    ctx.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    return outs[out_names[0]], outs[out_names[1]]

def infer_bgr_bytes(img_bytes, conf):
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    oh, ow = im.shape[:2]
    rate = max(H, W) / max(oh, ow)
    rw, rh = int(ow * rate), int(oh * rate)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:rh, :rw] = cv2.resize(im, (rw, rh))
    img_t = torch.permute(torch.from_numpy(canvas), (2, 0, 1)).contiguous().float().unsqueeze(0).cuda()
    t0 = time.time()
    hidden, refs = engine_run(img_t)
    t1 = time.time()
    b, s, l = head_decode(hidden, refs, conf)
    t2 = time.time()
    dets = []
    for j in range(len(s)):
        x1, y1, x2, y2 = b[j] / rate
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, ow), min(y2, oh)
        dets.append({
            "name": CLASSES[int(l[j])] if int(l[j]) < len(CLASSES) else str(int(l[j])),
            "confidence": round(float(s[j]), 5),
            "type": "bounding_box",
            "points": [{"x": round(float(x1), 1), "y": round(float(y1), 1),
                        "w": round(float(x2 - x1), 1), "h": round(float(y2 - y1), 1)}],
        })
    return dets, {"engine_ms": round((t1 - t0) * 1000, 1),
                  "head_ms": round((t2 - t1) * 1000, 1)}

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
            self._send(200, {"status": "ok", "model": "swin_l_trt_fp16_b1", "port": 18901})
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
            threshold = float(req.get("threshold", CONF_DEFAULT))
            request_id = req.get("request_id") or str(uuid.uuid4())
            # /api/predict: image_data is a plain string
            if isinstance(req.get("image_data"), str):
                img_bytes = _decode_image_item(req["image_data"],
                                               (req.get("source_type") or "network_b64").lower())
                dets, timing = infer_bgr_bytes(img_bytes, threshold)
                self._send(200, {
                    "request_id": request_id,
                    "model_name": "swin_l_trt_fp16_b1",
                    "predictions": dets,
                    "extra_info": timing,
                    "cost_ms": round((time.time() - t_start) * 1000, 1),
                })
                return
            # /api/batch_predict: image_data is a list
            img_list = req.get("image_data") or []
            results = []
            for item in img_list:
                img_bytes = _decode_image_item(item.get("image_data", ""),
                                               (item.get("source_type") or "network_b64").lower())
                dets, timing = infer_bgr_bytes(img_bytes, threshold)
                results.append({"predictions": dets, "extra_info": timing})
            self._send(200, {
                "request_id": request_id,
                "model_name": "swin_l_trt_fp16_b1",
                "results": results,
                "cost_ms": round((time.time() - t_start) * 1000, 1),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

if __name__ == "__main__":
    port = int(os.environ.get("TRT_PORT", "18901"))
    print(f"SERVE_READY http://0.0.0.0:{port} plan={PLAN}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
