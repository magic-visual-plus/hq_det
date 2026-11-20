# TRT 部署手册（Windows 平台机器 · 可复用流程）

> 适用：magic_fox_ai 平台机器（Windows）+ NVIDIA 独立显卡（已在 3090 24G
> 验证；3080/4080 同流程）。目标：把 hq_det 的 DINO/Swin 检测模型以
> TensorRT fp16 服务化，精度与原生 torch 等价、速度 1.4~1.65x。
>
> 单机部署耗时约 **30~40 分钟**（其中 plan 构建 2~7 分钟）。

## 0. 部署流程总览（checklist）

```
[1] 模型文件就位（ckpt + ONNX + classes.json）        ~10 分钟（含下载）
[2] TensorRT 安装（bindings/libs wheel + shim）        ~5 分钟
[3] 目标卡本地构建 plan（fp16）                        2~7 分钟（按卡型）
[4] 部署服务脚本 + 启动 + 健康检查                     ~5 分钟
[5] 精度验收（与平台 torch 服务对拍 200 张）           ~5 分钟
[6] 生产流量切换                                       ~5 分钟
```

## 1. 模型文件准备

放到 `D:\magic_fox_ai\resources\backend\trt_models\`：

| 文件 | 来源 | 说明 |
|---|---|---|
| `<模型名>.pth` | OSS/训练产物 | 原始 mmdet ckpt（平台 DET0309 加载器直接可用） |
| `<模型名>_<HxW>.onnx` | OSS（平台无关，各机共用） | hq_det 的 `trt_export_final.py` 导出（在任意一台 Linux 机器做一次即可） |
| `swin_l_b1_fp16.plan` | **目标卡本地 build** | 见 §3；plan 编码源卡 SM 数，**跨卡不可拷贝** |
| `classes.json` | ckpt 导出 | 类别名单，见下 |

classes.json 导出：

```python
import torch, json
c = torch.load(ckpt, map_location="cpu", weights_only=False)
json.dump(c["meta"]["dataset_meta"]["CLASSES"],
          open("classes.json", "w", encoding="utf-8"), ensure_ascii=False)
```

## 2. TensorRT 安装（Windows 专用坑）

NVIDIA pypi 上 `tensorrt-cu12` 顶层包只有 Linux sdist，Windows 直接 pip 装不上。
只装 bindings + libs（都有 win_amd64 wheel），wrapper 用 shim 补：

```bat
python.exe -m pip install --no-deps --only-binary :all: ^
    --index-url https://pypi.nvidia.com ^
    tensorrt-cu12-bindings==10.13.3.9 tensorrt-cu12-libs==10.13.3.9
copy tensorrt_shim.py  <python的site-packages>\tensorrt.py
```

验证：`python -c "import tensorrt; print(tensorrt.__version__)"`

## 3. Plan 构建（每张卡必须本机做）

```bat
python.exe build_plan_win.py
```

- 脚本内改 ONNX 文件名/输出名即可换模型；换输入尺寸需重新导出对应 ONNX
- 实测构建耗时：4080S ~7 分钟、3090 ~5 分钟（fp16）；fp32 plan 更快（~2 分钟）
- **坑 7：plan 编码了源卡 SM 数**，4090 的 plan 在 3090/3080 上无法运行
  （驱动再新也不行），每个卡型必须本地 build；ONNX 可直接复用
- 显存参考：Swin-L fp16 推理仅 ~2.4GB，**10G 卡够用**；build 峰值 8-10GB

## 4. 服务部署与启动

```bat
python.exe trt_service_18901.py
```

- 端口 18901（可改脚本内 TRT_PORT 默认值）；与平台 18900 服务共存
- 接口与 model_sdk 客户端完全兼容（`/api/predict` + `/api/batch_predict`）
- `image_data` 支持 base64 或本机路径（`local_file`，同机调用可省 ~90ms）
- 已内置与平台对齐的预处理（**GaussianBlur 3x3**，缺了会精度漂移，见 §8）
- 开机自启（管理员）：

```bat
schtasks /create /tn "TRTService" /tr "D:\magic_fox_ai\resources\backend\python.exe -X utf8 C:\deploy\trt_service_18901.py" /sc onstart /ru SYSTEM /rl HIGHEST
```

  （或放一个 .bat 到开始菜单"启动"文件夹；注意进程启动到可服务约 40 秒）

## 5. 精度验收（每台新机器必做）

两个服务并行（平台 torch 18900 + TRT 18901），200 张业务图对拍：

```bat
python.exe cmp200b.py 200
```

合格标准（3090 实测基准）：

| 指标 | 合格线 | 3090 实测 |
|---|---|---|
| 检出数量完全一致率 | ≥ 95% | 96% |
| 匹配对分数差 mean | ≤ 0.01 | 0.0048 |
| 匹配框 IoU mean | ≥ 0.95 | 0.957 |

**若分数差在 0.04 量级：预处理没对齐**（查 GaussianBlur 和 padding，见 §8），
不是 fp16 精度问题（可用 fp32 plan 复测验证：fp32 若同样漂移即排除精度）。

## 6. 生产流量切换

1. 平台服务（18900）保持在线作为回退，TRT（18901）并行
2. 客户端切换：model_sdk 的 `api_url` 改指 18901 即可
   （model_id 随意填，TRT 服务不校验）：

```python
mng = ModelManager(api_url="http://127.0.0.1:18901",
                   api_key="584ff3a5-2bff-4e5f-98e7-7af7e7600cf1")
model = mng.model(model_id="swin_l_trt")
```

3. 建议灰度：先切 10% 流量观察一天（对比检出数量/类别统计），再全量
4. 回退：api_url 改回 18900 即时生效
5. 注意：平台 `det03.py` 已打 padding 补丁（§8），切换前后平台行为已对齐

## 7. 平台侧配套改动（一次性，迁移新机器时同步检查）

| 文件 | 改动 | 备份 |
|---|---|---|
| `ml_main.py` | `DETECTION_BATCH_SIZE` 4 → **1**（b4 实测全线劣化） | git |
| `model_snapshot\md_snapshot.pkl` | 模型 ckpt 路径指向老机器的改为本机路径 | `.bak2` |
| `site-packages\...\det03.py` | DET0307-0309 分支加 "resize 长边 2048 + pad 方形"（与 TRT/训练对齐，坐标按 rate 还原） | `det03.py.bak` |
| license | 硬件指纹绑定，换机器必须重新生成（`hardware info not match` 报错即此） | — |

## 8. 已知坑速查（都是踩过的）

| 坑 | 现象 | 解法 |
|---|---|---|
| plan 跨卡不通用 | CUDA error 720 / "device only supports 80 SMs" | 目标卡本地 build（无捷径） |
| Windows 装 TRT | 顶层 tensorrt-cu12 只有 Linux sdist | 只装 bindings+libs wheel + shim |
| bindings 对 py 版本上限 | `_PyThreadState_UncheckedGet` 报错 | 用 py3.10/3.12 |
| 精度漂移 0.04 量级 | 与 torch 服务对比分数差大、数量差多 | 预处理没对齐：GaussianBlur + 非方形图 pad（不是 fp16 问题，fp32 复测同漂移即可证实） |
| b1 引擎多线程压测 | 请求排队直至超时 | 压测/服务一律单并发 |
| 中文的 cv2.imread | Windows OpenCV 打不开非 ASCII 路径 | `open(path,'rb')` + `cv2.imdecode` |
| 分离启动静默失败 | Start-Process 后进程消失 | 前台窗口跑，或计划任务 |
| 检出坐标 | 平台 batch_predict 的 Point 含 x,y,w,h | 对拍脚本 IoU 按 x,y,w,h 口径算（勿用 xyxy） |

## 9. 实测数据汇总（汇报用）

见 `TRT_DEPLOY.md` §6「Windows 部署实测」。核心数字（3090，5120×5120 产线图，
200 张，与平台 torch eager 对拍）：

- **精度**：检出数量一致率 96%；分数差 mean 0.0048（fp16 噪声级）；IoU 0.957；类别零错乱
- **速度**：端到端 252ms vs 293ms（**快 1.4x**）；纯推理 198ms vs ~370ms（**快 1.9x**）
- **资源**：推理显存 ~2.4GB（10G 卡可用）；单并发服务

各卡型单图性能参考（TRT fp16 b1，Linux 实测）：4090 15ms / 4080S 21ms /
3080 34.7ms / 3090 34.1ms（Swin-T）；Swin-L：4090 87.6ms / 4080S 113.3ms /
3090 199.5ms。Windows 端到端数字含 HTTP+base64 开销 ~55ms。
