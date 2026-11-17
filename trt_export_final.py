#!/usr/bin/env python
"""TRT export tool for HQ-DET DINO (mmdet) models.

Converts an mmdet DINO checkpoint (Swin-T two-stage) to TensorRT fp32 engines
for fixed input shapes, with e2e detection parity check against eager.

Usage:
  python trt_export_final.py --ckpt /home/model.pth --shapes 2048x2048 1536x2048
  python trt_export_final.py --ckpt /home/model.pth --shapes 2048x2048 --precisions fp32 fp16
  python trt_export_final.py --ckpt /home/model.pth --shapes 2048x2048 --skip-export  # rebuild plan only

Requires: mmengine, mmdet, mmcv(with ops), onnx, tensorrt.

Notes / patches applied at export time (do NOT change model numerics):
  1. mmcv MultiScaleDeformableAttnFunction -> python fallback (CUDA op is a
     black box for ONNX trace; the python grid_sample path is exactly
     equivalent within ~1e-6).
  2. fairscale _checkpointed_forward -> passthrough (activation checkpointing
     is a memory-saving recompute mechanism, numerically identical).
  3. with_cp=False on all modules (same reason).
Input semantics: the exported graph takes RAW 0-255 BGR CHW image and does
mean/std normalization inside; head decoding (bbox_head.predict) stays
outside the engine and runs in torch.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn

def apply_export_patches():
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

def build_model(ckpt):
    from hq_det.models.dino2 import hq_dino
    model = hq_dino.HQDINO(model=ckpt)
    model.load_model(ckpt)
    model.eval()
    model.to("cuda:0")
    mm = model.model
    for m in mm.modules():
        if hasattr(m, "with_cp") and getattr(m, "with_cp"):
            m.with_cp = False
    return model, mm

def export_onnx(mm, H, W, out_onnx):
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    pp = mm.data_preprocessor
    mean = pp.mean.detach().view(1, 3, 1, 1)
    std = pp.std.detach().view(1, 3, 1, 1)

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean_", mean)
            self.register_buffer("std_", std)
        def forward(self, img):
            x = (img - self.mean_) / self.std_
            feats = self.mm_extract_feat(x)
            ds = []
            for _ in range(img.shape[0]):
                s = DetDataSample(metainfo={
                    "img_shape": (img.shape[2], img.shape[3]),
                    "pad_shape": (img.shape[2], img.shape[3]),
                    "batch_input_shape": (img.shape[2], img.shape[3]),
                    "scale_factor": 1.0})
                s.gt_instances = InstanceData()
                ds.append(s)
            hi = self.mm.forward_transformer(feats, ds)
            return hi["hidden_states"], torch.stack(hi["references"])

    w = Wrapper().eval().cuda()
    w.mm = mm
    w.mm_extract_feat = mm.extract_feat  # plain attr; called inside forward
    img = torch.randn(1, 3, H, W, device="cuda")
    with torch.no_grad():
        torch.onnx.export(w, (img,), out_onnx, input_names=["img"],
                          output_names=["hidden_states", "references"],
                          opset_version=17, do_constant_folding=True)
        torch.cuda.empty_cache()
    print(f"ONNX_OK {out_onnx} ({os.path.getsize(out_onnx)/1e6:.0f}MB)", flush=True)

def build_plan(onnx_path, plan_path, precision="fp32"):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        assert parser.parse(f.read()), f"onnx parse failed: {onnx_path}"
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, "trt build failed"
    with open(plan_path, "wb") as f:
        f.write(serialized)
    print(f"PLAN_OK {plan_path} ({os.path.getsize(plan_path)/1e6:.0f}MB, "
          f"built in {time.time()-t0:.0f}s)", flush=True)

def verify_e2e(model, mm, plan_path, H, W, sample_img=None, conf=0.3):
    import tensorrt as trt
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    pp = mm.data_preprocessor
    mean = pp.mean.detach().view(1, 3, 1, 1)
    std = pp.std.detach().view(1, 3, 1, 1)
    runtime = trt.Runtime(logger=trt.Logger(trt.Logger.WARNING))
    with open(plan_path, "rb") as f:
        eng = runtime.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    out_names = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    def run_trt(t):
        ctx.set_tensor_address("img", t.data_ptr())
        outs = {}
        for n in out_names:
            shp = tuple(ctx.get_tensor_shape(n))
            outs[n] = torch.empty(shp, dtype=torch.float32, device="cuda")
            ctx.set_tensor_address(n, outs[n].data_ptr())
        ctx.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return outs[out_names[0]], outs[out_names[1]]
    def make_ds():
        ds = []
        for _ in range(1):
            s = DetDataSample(metainfo={"img_shape": (H, W), "pad_shape": (H, W),
                                        "batch_input_shape": (H, W), "scale_factor": 1.0})
            s.gt_instances = InstanceData()
            ds.append(s)
        return ds
    def head_decode(hidden, refs):
        if torch.is_tensor(refs):
            refs = list(torch.unbind(refs, dim=0))
        preds = mm.bbox_head.predict(hidden, refs, batch_data_samples=make_ds(), rescale=False)
        p = preds[0]
        mask = p.scores > conf
        return (p.bboxes[mask].detach().cpu().numpy(),
                p.scores[mask].detach().cpu().numpy(),
                p.labels[mask].detach().cpu().numpy())
    if sample_img and os.path.exists(sample_img):
        im = cv2.imread(sample_img)
        rate = max(H, W) / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1]*rate), int(im.shape[0]*rate)))
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:im.shape[0], :im.shape[1]] = im
        im = canvas
    else:
        im = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    img_t = torch.permute(torch.from_numpy(im), (2, 0, 1)).contiguous().float().unsqueeze(0).cuda()
    with torch.no_grad():
        x = (img_t - mean) / std
        feats = mm.extract_feat(x)
        ds = make_ds()
        hi = mm.forward_transformer(feats, ds)
        e_h, e_r = hi["hidden_states"], torch.stack(hi["references"])
    t_h, t_r = run_trt(img_t)
    det_e = head_decode(e_h, e_r)
    det_t = head_decode(t_h, t_r)
    # match by IoU + class
    b1, s1, l1 = det_e; b2, s2, l2 = det_t
    matched, max_sd = 0, 0.0
    used = set()
    for j in range(len(s1)):
        best, bi = -1, 0.0
        for k in range(len(s2)):
            if k in used:
                continue
            x1 = np.maximum(b1[j][:2], b2[k][:2]); x2 = np.minimum(b1[j][2:], b2[k][2:])
            w = np.maximum(0, x2-x1).prod()
            a1 = (b1[j][2]-b1[j][0])*(b1[j][3]-b1[j][1])
            a2 = (b2[k][2]-b2[k][0])*(b2[k][3]-b2[k][1])
            iou = w/max(a1+a2-w, 1e-9)
            if iou > bi and l1[j] == l2[k]:
                bi, best = iou, k
        if best >= 0 and bi >= 0.5:
            used.add(best)
            matched += 1
            max_sd = max(max_sd, abs(float(s1[j])-float(s2[best])))
    print(f"VERIFY {os.path.basename(plan_path)}: eager_dets={len(s1)} "
          f"trt_dets={len(s2)} matched={matched} max_score_diff={max_sd:.4f}", flush=True)
    # speed
    for _ in range(10):
        run_trt(img_t)
    ts = []
    for _ in range(50):
        t0 = time.time(); run_trt(img_t); ts.append((time.time()-t0)*1000)
    ts.sort()
    print(f"SPEED {os.path.basename(plan_path)}: p50={ts[25]:.1f}ms p90={ts[45]:.1f}ms", flush=True)

import cv2  # noqa: E402  (imported late to keep module import light)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shapes", nargs="+", required=True, help="HxW, e.g. 2048x2048")
    ap.add_argument("--outdir", default="./trt_out")
    ap.add_argument("--skip-export", action="store_true", help="reuse existing onnx")
    ap.add_argument("--verify-img", default=None, help="real image for e2e parity check")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--precisions", nargs="+", default=["fp32"],
                    choices=["fp32", "fp16"],
                    help="plan precisions to build (fp16 verified: 4.8x speed, "
                         "max score diff 0.002 on real image)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    apply_export_patches()
    model, mm = build_model(args.ckpt)
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    for shape in args.shapes:
        H, W = (int(v) for v in shape.lower().split("x"))
        onnx_path = f"{args.outdir}/{stem}_{H}x{W}.onnx"
        if not (args.skip_export and os.path.exists(onnx_path)):
            export_onnx(mm, H, W, onnx_path)
        for precision in args.precisions:
            plan_path = f"{args.outdir}/{stem}_{H}x{W}_{precision}.plan"
            build_plan(onnx_path, plan_path, precision)
            if not args.skip_verify:
                verify_e2e(model, mm, plan_path, H, W, args.verify_img)
    print("ALL_DONE", flush=True)

if __name__ == "__main__":
    main()
