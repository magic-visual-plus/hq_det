# HQ-DET TensorRT (TRT) 推理加速部署指南

> 适用范围：本仓库 `hq_det/models/dino2`（mmdet DINO + ResizeSwin 两阶段检测器）。
> 已在 RTX 4090（CUDA 12.8 / TensorRT 10.13）上完成两个模型的转换与验证（见"测试指标"）。
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
    --verify-img /path/to/real.jpg \
    --outdir ./trt_out
```

参数：
- `--shapes`：**必须与实际推理输入形状一致**（见 2.3）。可多个。
- `--precisions`：`fp32`（默认）/ `fp16`，可同时建。
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

## 3. 测试指标

RTX 4090, batch=1, fp32 引擎，端到端检测对拍（IoU≥0.5 + 同类别匹配）：

| 模型 | 输入 | eager | TRT fp32 | 提速 | TRT fp16 | fp16 提速 | score 最大偏差 (fp32/fp16) |
|---|---|---|---|---|---|---|---|
| model A（Swin-T，25 类） | 2048×2048 | 72.2ms | 29.7ms | **2.43x** | 15.0ms | **4.81x** | 0.0012 / 0.0020 |
| model B（Swin-B，TrashCan 系） | 2048×2048 | 334.3ms | 266.0ms | 1.26x | 172.1ms | **1.94x** | 0.0010 / 0.0053 |

对拍均在真实产线图片上进行，检测数量、类别、框（IoU>0.99）完全一致。

**结论**：Swin-T 级模型 fp16 收益显著；Swin-B 级模型因 transformer（可变形注意力）
占比高，fp32 收益有限，fp16 仍有 1.94x。如需更快：多尺度可变形注意力换成
TRT plugin（mmdeploy 有现成实现），encoder/decoder 预计可再提速 3-5x。

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

## 5. 产物清单（本分支）

| 文件 | 说明 |
|---|---|
| `trt_export_final.py` | 一键转换工具（本文档 §2） |
| `baseline_speed.py` | eager 基线测速（端到端 + 分段） |
| `build_mmcv.sh` | mmcv 源码编译（§1.2） |
