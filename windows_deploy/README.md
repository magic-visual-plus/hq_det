# Windows 部署指南（magic_fox_ai 平台 + RTX 3090）

目标机器：Windows 10/11 + RTX 3090 24G，已部署 magic_fox_ai 平台
（`D:\magic_fox_ai\resources\backend`，自带 Python 3.10 + torch 2.1.1+cu118 +
mmdet 3.3.0 + mmcv 2.1.0）。

本目录脚本在此环境验证通过（2026-11-19/20，驱动 610.88，TRT 10.13.3.9）。

## 1. 模型文件准备

放到 `D:\magic_fox_ai\resources\backend\trt_models\`：

| 文件 | 说明 |
|---|---|
| `daika_lungu_epoch12.pth` | Swin-L 原始 ckpt（mmdet 格式，877MB） |
| `daika_lungu_epoch12_2048x2048.onnx` | b1 导出（922MB，平台无关，可直接从 OSS 拿） |
| `swin_l_b1_fp16.plan` | 本机 build 的 fp16 引擎（470MB，**必须目标卡本地 build**） |
| `classes.json` | 25 类缺陷名单（从 ckpt `meta.dataset_meta.CLASSES` 导出） |

`classes.json` 导出：

```python
import torch, json
c = torch.load(ckpt, map_location="cpu", weights_only=False)
json.dump(c["meta"]["dataset_meta"]["CLASSES"],
          open("classes.json", "w", encoding="utf-8"), ensure_ascii=False)
```

## 2. 安装 TensorRT（Windows 专用坑）

NVIDIA pypi 上 `tensorrt-cu12` 顶层包只有 Linux sdist，Windows 装不上。
用 NVIDIA 索引只装 bindings + libs（都是 win_amd64 wheel）：

```bat
python.exe -m pip install --no-deps --only-binary :all: ^
    --index-url https://pypi.nvidia.com ^
    tensorrt-cu12-bindings==10.13.3.9 tensorrt-cu12-libs==10.13.3.9
```

wrapper 缺失，用本目录 `tensorrt_shim.py` 补——复制为
`site-packages\tensorrt.py`（8 行转发模块）。

## 3. 本机 build plan

```bat
python.exe build_plan_win.py
```

约 5 分钟（Swin-L fp16，470MB）。plan 编码源卡 SM 数，**换卡必须重建**（坑 7）。

## 4. 启动服务

```bat
D:\magic_fox_ai\resources\backend\python.exe -X utf8 C:\deploy\trt_service_18901.py
```

- 端口 18901（TRT fp16 b1），与平台自身 18900 服务共存
- 接口与 `model_sdk` 客户端兼容：
  `ModelManager(api_url="http://127.0.0.1:18901", api_key=...)` →
  `model.predict/batch_predict`
- `image_data` 支持 base64（`network_b64`）或本机路径（`local_file`），
  同机调用传路径可省掉 base64 开销（~90ms）
- torch eager 参照服务：`torch_service_18900.py`，端口 18900——注意与平台
  自身 18900 端口冲突，仅做性能对比时临时启动
- **已知坑**：PowerShell `Start-Process -WindowStyle Hidden` 分离启动在
  部分机器上对 torch 服务静默失败（TRT 服务正常）；建议用前台窗口或
  计划任务方式常驻

## 5. 压测 / 对比

```bat
python.exe -X utf8 stress.py 1 20 18901    # 单线程压测（勿用多线程，b1 引擎会排队超时）
python.exe -X utf8 compare.py              # 需两个服务同时在线
```

实测数据（3090 24G Windows，5120×5120 产线图，阈值 0.3）见 TRT_DEPLOY.md
「Windows 部署实测」一节。

## 6. 关键结论

- TRT fp16 b1 端到端 p50 = **239~255ms**（引擎 199ms + base64/HTTP ~50ms），
  torch eager p50 = 385ms，**TRT 快 38%**
- 与 4090 上的精度对拍一致：检出类别/框与 torch 参照 IoU mean 0.897
  （本对比里 torch 侧是 hq_dino 全前向 + 自带 resize，与 TRT 服务端
  resize 有轻微实现差异；4090 同管线对拍为 IoU>0.99）
- DETECTION_BATCH_SIZE 已在 `ml_main.py` 改为 1（b4 实测全线劣化，见主文档）
