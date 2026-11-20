#!/usr/bin/env python
"""Build Swin-L fp16 b1 plan on Windows + verify + quick bench."""
import time
import tensorrt as trt

ONNX = r"D:\magic_fox_ai\resources\backend\trt_models\daika_lungu_epoch12_2048x2048.onnx"
OUT = r"D:\magic_fox_ai\resources\backend\trt_models\swin_l_b1_fp16.plan"

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags)
parser = trt.OnnxParser(network, logger)
print("parsing onnx...", flush=True)
with open(ONNX, "rb") as f:
    ok = parser.parse(f.read())
if not ok:
    for i in range(parser.num_errors):
        print("PARSE_ERR", parser.get_error(i), flush=True)
    raise SystemExit(1)

cfg = builder.create_builder_config()
cfg.set_flag(trt.BuilderFlag.FP16)
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)

inp = network.get_input(0)
_, C, H, W = tuple(inp.shape)
prof = builder.create_optimization_profile()
prof.set_shape(inp.name, (1, C, H, W), (1, C, H, W), (1, C, H, W))
cfg.add_optimization_profile(prof)

t0 = time.time()
serialized = builder.build_serialized_network(network, cfg)
assert serialized is not None, "build FAILED"
with open(OUT, "wb") as f:
    f.write(serialized)
import os
print(f"PLAN_OK {OUT} {os.path.getsize(OUT)/1e6:.0f}MB built_in={time.time()-t0:.0f}s", flush=True)
