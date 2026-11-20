#!/usr/bin/env python
"""WS serving for TRT-converted hq_det DINO detector.

Protocol (JSON over websocket):
  request : {"image_b64": <base64 of jpg/png bytes>, "confidence": 0.3}
  response: {"detections": [{"bbox": [x1,y1,x2,y2], "score": float,
                             "label": str}, ...],
             "time_ms": {"engine": float, "decode": float, "total": float},
             "error": null}

Boxes are returned in ORIGINAL image pixel coords (scale-back included).
Engine is shape-bound: images are resized (keep aspect) to max side and
padded to the plan shape; padded regions produce no detections above conf.

Usage:
  python trt_serve.py --ckpt /home/model.pth --plan dino_fp32.plan \
      --shape 2048x2048 --port 8900
"""
import argparse, asyncio, base64, json, time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import torch
import tensorrt as trt

def apply_export_patches():
    import mmcv.ops.multi_scale_deform_attn as msda
    class _PyMSDAFn:
        @staticmethod
        def apply(value, value_spatial_shapes, level_start_index,
                  sampling_locations, attention_weights, im2col_step=None):
            return msda.multi_scale_deformable_attn_pytorch(
                value, value_spatial_shapes, sampling_locations, attention_weights)
    msda.MultiScaleDeformableAttnFunction = _PyMSDAFn

class Engine:
    def __init__(self, plan_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(plan_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.out_names = [n for n in self.names
                          if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.stream = torch.cuda.Stream()
    def __call__(self, img_t):
        self.ctx.set_tensor_address("img", img_t.data_ptr())
        outs = {}
        for n in self.out_names:
            shp = tuple(self.ctx.get_tensor_shape(n))
            outs[n] = torch.empty(shp, dtype=torch.float32, device="cuda")
            self.ctx.set_tensor_address(n, outs[n].data_ptr())
        self.ctx.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return outs[self.out_names[0]], outs[self.out_names[1]]

class Server:
    def __init__(self, ckpt, plan_path, shape):
        apply_export_patches()
        from hq_det.models.dino2 import hq_dino
        self.model = hq_dino.HQDINO(model=ckpt)
        self.model.load_model(ckpt)
        self.model.eval()
        self.model.to("cuda:0")
        self.mm = self.model.model
        for m in self.mm.modules():
            if hasattr(m, "with_cp") and getattr(m, "with_cp"):
                m.with_cp = False
        pp = self.mm.data_preprocessor
        self.mean = pp.mean.detach().view(1, 3, 1, 1)
        self.std = pp.std.detach().view(1, 3, 1, 1)
        self.classes = self.model.get_class_names()
        self.H, self.W = shape
        self.engine = Engine(plan_path)
        self.idle_cache = {}

    def make_ds(self):
        from mmengine.structures import InstanceData
        from mmdet.structures import DetDataSample
        ds = []
        for _ in range(1):
            s = DetDataSample(metainfo={"img_shape": (self.H, self.W),
                                        "pad_shape": (self.H, self.W),
                                        "batch_input_shape": (self.H, self.W),
                                        "scale_factor": 1.0})
            s.gt_instances = InstanceData()
            ds.append(s)
        return ds

    def head_decode(self, hidden, refs, conf):
        if torch.is_tensor(refs):
            refs = list(torch.unbind(refs, dim=0))
        preds = self.mm.bbox_head.predict(hidden, refs,
                                          batch_data_samples=make_ds_of(self.H, self.W),
                                          rescale=False)
        p = preds[0]
        mask = p.scores > conf
        return (p.bboxes[mask].detach().cpu().numpy(),
                p.scores[mask].detach().cpu().numpy(),
                p.labels[mask].detach().cpu().numpy())

    def infer(self, img_bytes, conf):
        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        oh, ow = im.shape[:2]
        rate = max(self.H, self.W) / max(oh, ow)
        rw, rh = int(ow * rate), int(oh * rate)
        im = cv2.resize(im, (rw, rh))
        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        canvas[:rh, :rw] = im
        img_t = torch.permute(torch.from_numpy(canvas), (2, 0, 1)).contiguous().float().unsqueeze(0).cuda()
        t0 = time.time()
        hidden, refs = self.engine(img_t)
        t1 = time.time()
        b, s, l = self.head_decode(hidden, refs, conf)
        t2 = time.time()
        # scale back: canvas coords -> original coords; clip
        dets = []
        for j in range(len(s)):
            x1, y1, x2, y2 = b[j] / rate
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, ow), min(y2, oh)
            dets.append({"bbox": [round(float(x1), 1), round(float(y1), 1),
                                  round(float(x2), 1), round(float(y2), 1)],
                         "score": round(float(s[j]), 5),
                         "label": self.classes[int(l[j])] if int(l[j]) < len(self.classes) else str(l[j])})
        return {"detections": dets,
                "time_ms": {"engine": round((t1-t0)*1000, 1),
                            "decode": round((t2-t1)*1000, 1),
                            "total": round((t2-t0)*1000, 1)},
                "error": None}

def make_ds_of(H, W):
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    ds = []
    s = DetDataSample(metainfo={"img_shape": (H, W), "pad_shape": (H, W),
                                "batch_input_shape": (H, W), "scale_factor": 1.0})
    s.gt_instances = InstanceData()
    ds.append(s)
    return ds

async def handle(ws, server):
    peer = ws.remote_address
    print(f"[conn] {peer}", flush=True)
    try:
        async for msg in ws:
            try:
                req = json.loads(msg)
                img_b = base64.b64decode(req["image_b64"])
                conf = float(req.get("confidence", 0.3))
                t0 = time.time()
                resp = server.infer(img_b, conf)
                resp["time_ms"]["total"] = round((time.time()-t0)*1000, 1)
            except Exception as e:
                resp = {"detections": [], "error": f"{type(e).__name__}: {e}"}
            await ws.send(json.dumps(resp, ensure_ascii=False))
    except Exception as e:
        print(f"[disc] {peer}: {type(e).__name__}: {e}", flush=True)
    print(f"[disc] {peer}", flush=True)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--shape", required=True, help="HxW, must match plan")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8900)
    args = ap.parse_args()
    H, W = (int(v) for v in args.shape.lower().split("x"))
    server = Server(args.ckpt, args.plan, (H, W))
    print(f"SERVE_READY ws://{args.host}:{args.port} plan={args.plan} shape={H}x{W}", flush=True)
    async with __import__("websockets").serve(
            lambda ws: handle(ws, server), args.host, args.port, max_size=64 << 20):
        await asyncio.Future()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
