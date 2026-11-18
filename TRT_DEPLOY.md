# HQ-DET TensorRT (TRT) 推理加速部署指南

> 适用范围：本仓库 `hq_det/models/dino2`（mmdet DINO + ResizeSwin 两阶段检测器）。
> 已在 RTX 4090（CUDA 12.8 / TensorRT 10.13）上完成两个模型的转换与验证（见"测试指标"）。
> **适用性结论先行**：Swin-T 级模型 TRT fp16 加速 2.64x（`--msda fallback`）；Swin-L 级模型 TRT fp16 加速 1.92x（`--msda batched`）——两个尺寸级别均有真实加速路径（见 §3）。
> 维护：TRT 转换问题优先看文末"原理与坑"，大部分问题都踩过。

## 1. 环境准备

### 1.1 Python 环境

Python >= 3.10（验证环境为 3.12 + torch 2.8.0+cu128）：

```bash
pip install --trusted-host mirrors.cloud.aliyuncs.com \
    mmengine==0.10.7 mmdet==3.3.0 onnx onnxruntime-gpu fairscale
```

### 1.2 mmcv（必须带 CUDA ops，无预编译轮子，需源码编译）

DINO 的可变形注意力依赖 mmcv 编译扩展；torch 2.8 没有 openmmlab 预编译轮子。
在**有 nvcc 的机器**上编译（无需 GPU，编译节点即可），约 40-60 分钟：

```bash
bash build_mmcv.sh   # CUDA_HOME=/usr/local/cuda-12.8 FORCE_CUDA=1 MMCV_WITH_OPS=1
```

要点：
- `--no-build-isolation`（隔离构建环境缺 torch 会直接失败）
- `TORCH_CUDA_ARCH_LIST` 按目标 GPU 设置（4090 = 8.9）
- 编译产物装进与运行相同的 python 环境（共享文件系统可直接复用）

### 1.3 TensorRT

```bash
pip install tensorrt-cu12
```

已知坑（NAT/代理环境）：`tensorrt-cu12` 是 stub 包，安装时要从
`pypi.nvidia.com` 拉真实 wheel，若 SSL 证书校验失败：
- 方案 A：`pip download tensorrt-cu12-libs / tensorrt-cu12-bindings` 后本地 `pip install --no-deps`
- 方案 B：`SSL_CERT_FILE` 指向包含代理 CA 的 bundle
验证：`python -c "import tensorrt; print(tensorrt.__version__)"`（验证版本 10.13.3.9）

## 2. 使用方法

### 2.1 一键转换 + 验证

```bash
PYTHONPATH=/home/hq_det python trt_export_final.py \
    --ckpt /path/to/model.pth \
    --shapes 2048x2048 1536x2048 \
    --precisions fp32 fp16 \
    --batch 4 \
    --export-device cpu \
    --verify-img /path/to/real.jpg \
    --outdir ./trt_out
```

参数：
- `--shapes`：**必须与实际推理输入形状一致**（见 2.3）。可多个。
- `--precisions`：`fp32`（默认）/ `fp16`，可同时建。
- `--batch`：固定 batch 维（默认 1）。产线攒批打分时按需设（如 4）。
  plan 文件名会带 `_b4` 后缀。
- `--export-device`：ONNX trace 的设备，默认 cuda。**大模型 + batch>1 时
  建议 cpu**（Swin-L batch=4 在 24G 卡上 trace 会 OOM，CPU 内存充足），
  只影响导出速度，不影响产物。
- `--msda`：`auto`（默认，按 backbone.embed_dims 自动选）/ `fallback` / `batched`。
- `--verify-img`：真实图片做端到端检测对拍（推荐，随机图会让 topk 对比失真）。
- `--skip-export`：ONNX 已存在时只重建 plan。
- `--skip-verify`：跳过对拍。

产物：`trt_out/<ckpt名>_<H>x<W>_<精度>.plan` + 对应 `.onnx`。

### 2.2 TRT 化的推理流程（客户端接入）

引擎只覆盖模型主体，前后处理留在 torch/numpy：

```python
# 1) 预处理（与 hq_det predict() 一致）：BGR 图 -> 等比 resize(max边=2048) -> CHW float32
# 2) 引擎前向：img(1,3,H,W) [0,255] -> hidden_states(6,1,900,256) + references(7,1,900,4)
# 3) 后处理（torch）：
refs = list(torch.unbind(refs, dim=0))
preds = model.model.bbox_head.predict(hidden, refs, batch_data_samples=ds, rescale=False)
```

完整示例见 `trt_export_final.py` 的 `verify_e2e()`（含 ds 构造、top-k 匹配、测速）。

### 2.3 输入形状与 plan 绑定（重要）

`predict()` 按比例 resize 到 max 边 = image_size(2048)，**不同宽高比的图输入形状不同**；
Swin 的窗口 pad 是动态操作，ONNX 只能固定形状导出。因此：

- **一个 plan 只对一个 (H, W)**；产线图尺寸固定时建对应形状即可（40s/个）
- 形状不匹配时引擎不报错但输出垃圾（实测踩过：1536×2048 plan 喂 2048×2048 图 → 0 检测）
- 上线前务必用 `--verify-img` 对拍确认

### 2.4 服务化（WS 常驻服务）

`trt_serve.py` 把 TRT 引擎包成 websocket 常驻服务（单实例绑定一个 ckpt + plan + 形状）：

```bash
python trt_serve.py --ckpt /path/to/model.pth \
    --plan trt_out/model_2048x2048_fp32.plan \
    --shape 2048x2048 --port 8900
```

客户端协议（JSON over websocket）：

```python
import asyncio, base64, json, websockets

async def infer(image_path, conf=0.3):
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    async with websockets.connect("ws://<host>:8900", max_size=64 << 20) as ws:
        await ws.send(json.dumps({"image_b64": b64, "confidence": conf}))
        return json.loads(await ws.recv())
# resp = {"detections": [{"bbox": [x1,y1,x2,y2], "score": .., "label": ".."}],
#         "time_ms": {"engine": .., "decode": .., "total": ..}, "error": null}
# bbox 为原始图片像素坐标（服务端已做 scale-back 与裁剪）
```

- 任意尺寸图片均可发：服务端等比 resize 到 max 边并 pad 到 plan 形状，pad 区域不产生检测
- 客户端示例见 `ws_client.py`
- 实测（4090，demo 图，WS 往返含 base64/resize/引擎/解码）：热身 **34~35ms/次**
- 与 phi_lerobot 的 policy_server 不同：检测服务不依赖 lerobot 栈，纯 websocket+JSON

## 3. 测试指标

RTX 4090, batch=1, 真实产线图（5120×5120），端到端检测对拍（IoU≥0.5 + 同类别，
检测数量/类别/框 IoU>0.99 完全一致）。eager 基线 = 原始模型（mmcv CUDA MDA 算子路径，
无导出补丁——见 §4 坑 6，性能基线禁止用补丁版 eager）。

| 模型 | eager | TRT fp32 (fallback) | TRT fp16 (fallback) | TRT fp32 (batched) | TRT fp16 (batched) |
|---|---|---|---|---|---|
| model A（Swin-T，48M） | 39.6ms | 29.7ms（**1.33x**）/ diff 0.0012 | **15.0ms（2.64x）** / diff 0.0020 | 31.9ms（1.24x）/ diff 0.0031 | 16.9ms（2.34x）/ diff 0.0020 |
| model B（Swin-L，~200M） | 168.1ms | 266.0ms（0.63x）/ diff 0.0010 | 172.1ms（1.02x）/ diff 0.0053 | **161.6ms（1.04x）** / diff 0.0015 | **87.6ms（1.92x）** / diff 0.0060 |

- score 最大偏差 = 端到端检测对拍中匹配框的 score 绝对差最大值（全部检测 IoU>0.99、类别一致）
- **选型**：Swin-T 级 → fp16 + `--msda fallback`（2.64x）；Swin-L 级 → fp16 + `--msda batched`（1.92x）；
  高精度保底场景 fp32：A 用 fallback（1.33x），B 用 batched（1.04x）

### batch 扩展实测（4090 vs 3090，model B fp16/baseline）

| 方案 | batch | 单图耗时 | FPS |
|---|---|---|---|
| 4090 原始 torch 基线（现产线） | 1 | 168.1ms | 5.95 |
| 4090 原始 torch 基线 | 4 | 163.8ms | 6.11 |
| 3090 TRT fp16 | 1 | 202.4ms | 4.94 |
| 3090 TRT fp16 | 4 | 200.1ms | 5.00 |
| 3090 TRT fp16 | 5 | ❌ 导出 OOM（24G 装不下 Swin-L 的 trace） | — |

**batch 无摊薄效应**（两边 1→4 都几乎持平）：2048×2048 输入 + Swin-L 的
计算量在 batch=1 时已喂满 SM，batch 增大只是线性排队；且 Swin-L 的导出
trace 在 24G 卡上 batch=5 就 OOM。batch 路线到头，换卡决策只看单图速度。

### 3090 替换 4090 的成本分析（按产线口径：3090+TRT vs 4090+原始 torch）

- 吞吐保留率：3090 TRT fp16 = 4090 原始基线的 **83%**（4.94 vs 5.95 FPS）
- AutoDL 租金：3090 ≈ 4090 的 60~70%
- **性价比 = 83% ÷ 65% ≈ 1.28x——成本敏感场景 3090+TRT fp16 可替换，省 ~25% 成本**
- 限制：单图延迟 202ms vs 168ms（慢 20%）；若 4090 也上 TRT fp16（87.6ms，
  15 FPS），性能优势回到 4090 侧（2.3x）——优化是普适的，两边都可以上
- 显存：两边同为 24G 级，batch 上限一致（Swin-L 上限 ~4）

### 3080（sm_86，20G）复测——明细

| 指标 | 数值 |
|---|---|
| 原始 eager 前向（warm） | 81.4ms（冷启动首帧 588.9ms，含 cudnn 算法搜索，常驻服务无此问题） |
| TRT fp32 引擎 | 73.6ms（仅比 eager 快 10%——fp32 下 TRT 优化空间小） |
| **TRT fp16 引擎** | **48.9ms（1.66x）** |
| 精度对拍 | fp32 3/3 diff 0.0028；fp16 3/3 diff 0.0072 |

**3080 比 3090 慢的硬件解释**：Swin 推理是算力+带宽混合负载，3080（GA102 20G 版）
带宽 760GB/s、SM 68 个——两项都低于 3090（936GB/s、82 SM），大特征图的显存搬运
是短板。**3080 已停产且性能不如 3090，不推荐作为产线卡**（在售卡里 4080 Super
同为 sm_89 且带宽 816GB/s，预估 fp16 ~22ms，是 4090 之下的最优选择）。

### 3090（sm_86）复测——plan 需按目标架构重新 build，ONNX 可直接复用

其他架构的 plan 同理需在目标架构上 build（A100 = sm_80、L40S/4090 = sm_89、
3090/3080 = sm_86、5090/5080 = sm_120）：ONNX 平台无关可直接复用，
在目标机器重跑 `--skip-export` 的 build + verify 即可。
A100（sm_80）实测数据待有环境后补充（TRT 10.13 / torch 2.8 cu128 均支持 sm_80，
管线无适配工作）。

| 模型 | eager | TRT fp32 | TRT fp16 | fp16 提速 |
|---|---|---|---|---|
| model A（Swin-T） | 77.5ms | 58.4ms | 31.6ms | **2.45x** |
| model B（Swin-L） | 304.4ms | 357.2ms | 202.4ms | **1.50x** |

- 精度与 4090 同级（A 3/3 diff 0.0054/0.0020；B 4/4 diff 0.0009/0.0055）
- 3090 ≈ 4090 的 43~47%（与 FP16 tensor core 算力比一致）
- 3090 的 TRT fp16 比同模型 4090 原始 eager 还快（31.6ms < 39.6ms）——旧卡+TRT 可追平新卡+torch

### `--msda` 两种导出模式的取舍

- `fallback`（默认）：逐 level grid_sample 展开——**Swin-T 级模型用这个**（MDA 占比小，
  展开算子不构成瓶颈，且省去 pad/cat 开销）
- `batched`：全部 level 融合成一次 grid_sample（pad 到最大形状 + 归一化坐标重映射）——
  **Swin-L 级模型必须用这个**（fallback 展开的数百个小算子在 TRT 里完全抵消卷积收益：
  fp32 0.63x / fp16 1.02x；batched 后 fp16 1.92x）
- 数值：batched 与 fallback 对拍等价（diff ~1e-5，坐标重映射舍入级）

### 适用性结论

- 两个尺寸级别现在都有真实加速路径（见上表"推荐配置"列）
- 如需进一步提速 Swin-L：下一个值得评估的方向是 FP16 + int8 校准

### MDA TRT plugin 路线调查记录（2026-11-18，结论：暂缓）

对 TRT 官方 plugin（`MultiscaleDeformableAttnPlugin_TRT`）做了完整可行性验证：

| 步骤 | 结果 |
|---|---|
| TRT 10.13 自带该 plugin（registry 确认，version "1" legacy） | ✅ 免编译 mmdeploy |
| custom_op + symbolic 导出 plugin 节点（INT32 元数据、4D value、静态 reshape） | ✅ |
| PARSE_OK / BUILD_OK / 运行 | ✅ |
| 数值对拍（强受控探针：value=arange、单点权重、指定坐标） | ❌ q 维语义不一致 |

探针发现：plugin 对 q0 的采样与 python 参考完全一致，但 q1 输出错误且
在不同 locs 配置下输出不变——指向 10.13 中 version "1"（legacy IPluginV2）
实现或其 ONNX parse 层的 q 维处理问题。已核对 NVIDIA 官方 CUDA kernel 源码
（索引语义与 mmcv 一致），问题在 plugin 封装层。

**结论**：batched GridSample（本分支方案）已拿到主要收益（Swin-T 2.64x /
Swin-L 1.92x）；plugin 路线等 mmdeploy 官方适配 TRT 10.x 新接口后再评估。
调查工件：AutoDL 3090 上 `/root/msda_plugin_step1.py`（单算子验证）、
`/root/msda_probe2.py`（探针）、`/root/step1_full.log`。

## 4. 原理与坑（为什么这么做）

1. **mmcv 可变形注意力 CUDA 算子是 ONNX trace 黑盒**
   eager 在 CUDA 上走 `MultiScaleDeformableAttnFunction`（编译算子），trace 后
   ONNX 图语义错误（表现为 hidden_states 与 eager 差 ~5，ORT/TRT 一致地错）。
   导出前 monkeypatch 为 `multi_scale_deformable_attn_pytorch`（纯 torch
   grid_sample 实现，数值差 ~1e-6），修复后各分段对拍 ≤1.3e-3。
2. **fairscale checkpoint / with_cp 必须透传或关闭**
   encoder 层被 `checkpoint_wrapper` 包装、Swin 开 `with_cp`，trace 时会
   崩溃或黑盒化。checkpoint 只是省显存的机制，透传不改变数值。
3. **topk query selection 对数值噪声敏感**
   DINO 的 pre_decoder 用 topk 选 proposal，fp32 重排引起的 ~1e-4 数值差会
   改变选中索引 → hidden_states 逐位差异大（~5），但**最终检测框/类别一致**
   （IoU>0.99，score 差 1e-3 级）。对拍必须用端到端检测指标，不能只看张量。
4. **对拍输入语义要一致**
   导出图内含归一化（输入 0-255），对拍时 eager 与引擎必须喂同一张原始图；
   不要"还原后"再喂（会人为制造 6+ 的假发散）。
5. **分段二分定位是排障利器**
   extract_feat / pre_transformer+encoder / pre_decoder / decoder 逐段导出
   对拍，能在一次运行内锁定第一处发散点。
6. **性能基线污染（本项目最大教训）**
   为 ONNX 对拍打的 eager 补丁（CUDA MDA 算子 → python fallback）会把 eager
   拖慢约 2 倍（Swin-L：真实 168ms vs 被污染 345ms）。曾据此错误上报
   "2.43x / 4.81x 加速"，后用同事在 4090D 上的产线数据（172ms）交叉验证才
   发现。**对拍参考与性能基线必须分开测**：对拍用补丁版 eager（语义对齐
   ONNX），性能基线用原始路径（无补丁）。

## 5. 产物清单（本分支）

| 文件 | 说明 |
|---|---|
| `trt_export_final.py` | 一键转换工具（本文档 §2，含对拍与测速） |
| `trt_serve.py` | WS 常驻推理服务（§2.4） |
| `ws_client.py` | 服务客户端示例 |
| `build_mmcv.sh` | mmcv 源码编译（§1.2） |
