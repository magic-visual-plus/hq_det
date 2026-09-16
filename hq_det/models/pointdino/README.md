# PointDINO 集成与验证报告

PointDINO 是独立的点检测模型，使用 `PointDINO*` 类和注册名，不修改
HQ-Det 原有 DINO、DINOHead、GroundingDINO、CoDETR 或公共数据变换。
预测结果为 `InstanceData(points, scores, labels)`，不是检测框。

新增的可编辑入口与数据说明：`scripts/run_train_pointdino.py` 的 `TRAINING`
及 `scripts/test_pointdino.py` 的 `TESTING` 可集中设置自定义数据路径、JSON、
图片目录、类别与训练参数。完整数据目录/JSON 字段要求见
`scripts/POINTDINO_DATA_FORMAT.md`；本次用户 Stage2 权重的实际训练检查见
`scripts/POINTDINO_CHECKPOINT_CHECK.md`。原有命令行模式仍保留。

## 来源与目标架构

实际目标目录为 `hq_det`（远端 `magic-visual-plus/hq_det`），分支 `main`，
分析基线提交 `e87dd15`。源为 `dinov3_dino_mmdet` 的
`mmdet_dino_2d_stage4` 分支，提交 `aef51733`，与其父提交 `cfd5d3a9`
比较识别 PointDINO 修改。开始移植时两个工作区均干净；不提交 commit。

HQ-Det 是统一包装框架，实际 DINO/DeformableDETR 核心来自已安装的
MMDetection，不是目标内的 `mmdet/` 源码。分析使用的模板为：

- `hq_det/models/dino/`、`dino2/`：DINO 配置、checkpoint 和 HQ 包装方式。
- `hq_det/models/codetr/`：独立模型子包、局部导入与 MMDetection Registry。
- `hq_det/models/gdino/`：已有 GroundingDINO 接入方式的交叉检查。

`hq_det/__init__.py` 和 `hq_det/models/__init__.py` 都为空，框架没有必须修改的
中央模型工厂。新包的 `__init__.py` 导入独立组件，新配置用 `custom_imports`
加载新包；类分别注册到 `mmdet.registry` 的 `MODELS`、`TASK_UTILS`、
`TRANSFORMS` 和 `METRICS`。没有 `force=True` 或 monkey patch。

HQ 通用 trainer 的 bbox 数据、后处理和评价不适合直接处理点预测；其实际
optimizer/scheduler 也不读取 DINO config 的完整训练设置。因此本集成使用
目标已有 MMEngine Runner 接入方式，新增专用 train/test 入口，使点数据、
所有 loss、FIDT、optimizer、scheduler 和 evaluator 都经过完整源训练流程。

## 依赖图与类映射

```text
PointDINO
├── ResNet50 → ChannelMapper → DeformableDetrTransformerEncoder
│                              + SinePositionalEncoding
├── PointDINOTransformerDecoder
│   └── 原生 decoder layer / MultiScaleDeformableAttention / MLP
├── PointDINOCdnQueryGenerator
│   └── 原生 CDN 的 label noise / groups / attention mask
├── PointDINOHead
│   ├── HungarianAssigner(FocalLossCost + PointDINOL1Cost)
│   └── FocalLoss + L1Loss + 像素 Euclidean loss
└── 可选 PointDINOFIDTAuxHead（只在 loss 中共享最高分辨率 neck feature）

BaseDetDataset → PointDINO load / resize / flip / crop / pack
              → DetDataPreprocessor → model → PointDINOMetric
```

| 源组件 | 目标独立类 | 移植内容 |
|---|---|---|
| 修改后的 `DINO` | `PointDINO` | 二维 proposal、query、DN、FIDT 接入 |
| 修改后的 `DINOHead` | `PointDINOHead` | 二维回归、点 target、matching/DN/encoder loss、预测 |
| 修改后的 `DinoTransformerDecoder` | `PointDINOTransformerDecoder` | 二维位置编码和逐层 reference refinement |
| `PointCdnQueryGenerator` | `PointDINOCdnQueryGenerator` | 正负点噪声、二维 DN 拼接 |
| `PointFIDTAuxHead` | `PointDINOFIDTAuxHead` | FIDT target、full-map/local MSE |
| `PointL1Cost` | `PointDINOL1Cost` | 归一化点 L1 匹配 cost |
| `LoadAnnotations` 的点改动 | `PointDINOLoadAnnotations` | `point` / `point_label` / ignore flags |
| `Resize` / `RandomFlip` / `RandomCrop` 的点改动 | `PointDINOResize` / `PointDINORandomFlip` / `PointDINORandomCrop` | 同步变换点与标签 |
| `PackDetInputs` 的点映射 | `PointDINOPackDetInputs` | `gt_instances.points` 与 labels |
| `PointMetric` | `PointDINOMetric` | 阈值内一对一匹配与点检测指标 |

直接复用原生 ResNet、ChannelMapper、encoder、decoder layer、attention、
positional encoding、HungarianAssigner、FocalLossCost、FocalLoss、L1Loss、
BaseDetDataset、DetDataPreprocessor 和 MMEngine hooks/runtime。
没有 PointDINO 专属 bbox coder、额外点特征采样层或自定义 optimizer。
父类本来支持二维 head forward、标准初始化及部分 DN loss 聚合，保留继承，
只覆盖不同的方法。

特别确认：本次源 Stage4 PointDINO 使用 **ResNet50**。
源 `projects/dinov3_dino` 的 ViT、bridge、DDP、hooks 和 optimizer 是其他实验，
不在此模型的依赖链中。目标没有等价 DINOv3，但本次不需要新增 DINOv3
wrapper、外部模型包或 DINOv3 权重。源 GroundingDINO、EMA、text transform
等无关改动也没有迁移。

## 算法与配置

- 二阶段 encoder proposal、head regression、DN 与每层 reference 均为二维
  `(x, y)`，不构造人工 `(w, h)`；保留 top-k、detach 与 look-forward-twice。
- 点 DN 在归一化坐标中加入随机方向噪声：正半径 `[0,s)`、负半径 `[s,2s)`，
  `s=0.01`；保留 label noise `0.5` 和 dynamic DN budget `100`。
- Hungarian cost：Focal `2.0`、Point L1 `20.0`。后者是匹配权重，
  不等于 L1 训练权重 `5.0`。
- decoder matching、DN、encoder 均使用分类、点 L1 和 Euclidean loss。
  Euclidean 项为正样本的 `sqrt(dx_pixel² + dy_pixel² + 1e-6)` 之和，
  除以同步正样本数和 `8`，再乘 `0.20`。六层 decoder 共 39 项基础 loss。
- FIDT 使用最近 GT 距离 `D`：`1 / (D ** (0.02 * D + 0.75) + 1)`。
  最高分辨率 neck feature 经 `256→64/GN8` 分支、2 倍上采样输出单通道图。
  full-map 对有效图像区域计算 MSE，local 仅选择距 GT 不超过 `16` 像素的格点。
  保留权重 `1.0`、chunk `4096/256`、debug 统计；`fidt_mse` 等 detached
  统计不会加入总 loss。`enabled=False` 或零权重完全跳过分支构造。
- 预测保持 sigmoid 类别展开后的 top-k，无 NMS，输出像素点并支持原图缩放还原。

主配置：900 queries，四层级特征，encoder/decoder 各六层，单类，输入
`1024×768`，ResNet50 `frozen_stages=1`。AdamW `lr=1e-4`、
`weight_decay=1e-4`、backbone LR 倍率 `0.1`、梯度裁剪 `0.1`；12 epochs，
第 11 epoch 学习率乘 `0.1`；train batch 2、val/test batch 1、seed 0。
保存最优 checkpoint 使用 `point/f1@8px`。`loss_iou` 保留父类构造配置，
点损失路径不计算 IoU。

三处明确适配，不改变正常源训练的数值算法：

1. `PointDINOHead.split_outputs` 保留源对 `dn_meta=None` 的处理，
   避免原生 MMDetection 3.3 在关闭 DN 时访问空字典。
2. 空 GT 导致零 DN query 时，用 query **数量**而非 batch 大小判断，
   给 label embedding 添加零值计算图连接，防止它在分布式训练中成为未用参数。
3. 验证管线先 resize 图片，再加载原始点标注，使 GT 与 `rescale=True` 预测
   使用同一坐标系。源 native `1024×768` 行为不变，其他尺寸也能正确评价。

## 使用方法

从仓库根目录操作，在已有 HQ 环境安装本项目：

```bash
python -m pip install -e .
```

验证基线是 Python 3.10、PyTorch 2.1、torchvision 0.16、MMCV 2.1.0、
MMEngine 0.10.7、MMDetection 3.3.0；MMCV 必须与 PyTorch/平台匹配。
数据转换还使用 SciPy 与 Pillow。PointDINO 不依赖源仓库的 Python 路径。

数据目录默认：

```text
data/shanghaitech_part_b/
    train_point.json
    test_point.json
    train_data/images/
    train_data/ground_truth/
    test_data/images/
    test_data/ground_truth/
```

从原始 ShanghaiTech MAT 转换，点坐标保持原值，不进行取整或额外坐标偏移：

```bash
python scripts/pointdino_convert_shanghaitech.py --img-dir data/shanghaitech_part_b/train_data/images --gt-dir data/shanghaitech_part_b/train_data/ground_truth --out data/shanghaitech_part_b/train_point.json
python scripts/pointdino_convert_shanghaitech.py --img-dir data/shanghaitech_part_b/test_data/images --gt-dir data/shanghaitech_part_b/test_data/ground_truth --out data/shanghaitech_part_b/test_point.json
```

标准 BaseDetDataset JSON 格式如下；空标注使用 `instances: []`。

```json
{"metainfo":{"classes":["point"]},"data_list":[{"img_id":0,"img_path":"IMG_1.jpg","width":1024,"height":768,"instances":[{"point":[120.25,80.5],"point_label":0,"ignore_flag":0}]}]}
```

训练、从源二维初始化 checkpoint 训练、local FIDT、测试：

```bash
python scripts/run_train_pointdino.py --data-root data/shanghaitech_part_b
python scripts/run_train_pointdino.py --data-root data/shanghaitech_part_b --load-from checkpoints/point_dino_stage2_step2_init.pth
python scripts/run_train_pointdino.py hq_det/models/pointdino/configs/pointdino_r50_shanghaitech_stage4_local_12e.py --data-root data/shanghaitech_part_b
python scripts/test_pointdino.py hq_det/models/pointdino/configs/pointdino_r50_shanghaitech_stage4_12e.py work_dirs/pointdino_r50_shanghaitech_stage4_12e/epoch_12.pth --data-root data/shanghaitech_part_b
```

`--resume CHECKPOINT` 恢复模型、optimizer 和训练进度；仅 `--resume` 从 work-dir
自动续训。`--load-from` 只初始化权重。可用 `--cfg-options` 覆盖配置：

```bash
python scripts/run_train_pointdino.py --cfg-options model.point_fidt_head.enabled=False
python scripts/run_train_pointdino.py --cfg-options model.use_dn=False model_wrapper_cfg.find_unused_parameters=True
python scripts/run_train_pointdino.py --cfg-options model.backbone.init_cfg=None train_dataloader.num_workers=0 train_dataloader.persistent_workers=False
```

关闭 DN 时生成器参数仍保留以兼容源 state_dict；DDP 使用
`model_wrapper_cfg.find_unused_parameters=True`。单进程不需要该配置。
启用 AMP、自动学习率缩放或分布式 launcher 是显式可选项，默认仍使用源设置。
Windows 入口将 multiprocessing start method 适配为 `spawn`。

默认 `load_from=None`，不会引用源服务器的绝对路径。backbone 使用标准
`torchvision://resnet50` 初始化；离线环境可设为本地权重或 `None`。
要复现原实验初始化，必须提供相应的 **二维 PointDINO** Stage2 checkpoint；
普通四维 DINO checkpoint 的回归头和位置 MLP 形状不兼容，不能直接当作它使用。
类重命名不改变 parameter key，已有二维 PointDINO state_dict 可沿用。

## 验证

在独立临时 Python 环境、目标仓库根目录执行：

```bash
python -m pytest -q tests/test_pointdino_integration.py tests/test_pointdino_fidt.py tests/test_pointdino_smoke.py tests/test_pointdino_data.py tests/test_pointdino_runner.py tests/test_pointdino_cli.py tests/test_pointdino_scripts.py
```

测试使用实际 stock MMDetection、MMCV CPU 扩展和 PyTorch 运算，无虚构模型或
attention 替身。只有 CLI 参数传递测试使用隔离参数检查。

2026-09-15 初始集成验证为 46 项测试。加入可编辑脚本和自定义数据配置后，
最终执行结果为 **53 passed，3 subtests passed**，耗时 26.34 秒。
首轮发现的 DN 关闭空值访问已修复并通过回归；所有下表列出的检查均已实际通过。

| 检查 | 内容 |
|---|---|
| import / registry | 各 PointDINO 注册名及原生 DINO/Head/transform identity |
| config / full build | full-map 和 local 原始 900 queries、6+6 层完整模型构建 |
| dummy tensor forward | 真正 backbone→neck→encoder→decoder→head 的二维输出 |
| loss / backward | 六层 decoder 的 39 项基础 loss，FIDT loss，finite 与梯度传播 |
| DN | 径向正负噪声；启用/关闭 DN；空 GT 和混合有点/空点 batch |
| FIDT | 公式、最近距离、padding、local 半径、空掩码、chunk 和梯度 |
| predict / test_step | 点、分数、标签输出；FIDT 推理零调用；开关分支推理逐位一致 |
| native shape | 768×1024 输入的真实 backbone/neck 与 192×256 FIDT 输出 |
| Runner | 真实 JSON/PNG→dataloader→AdamW train→val→保存→重新加载→test |
| existing model | HQ 原有 DINO 配置加载和完整 build，维持四维回归及原类 identity |
| data / metric | resize/flip/crop/ignore/empty targets 与点匹配指标 |

烟雾 forward 使用 256×256 输入、30 queries、两层 encoder 和全部六层 decoder，
保留 R50、宽度 256、四层级特征、真实 matcher 和 DN。
Runner 合成训练使用 128×128、10 queries、encoder/decoder 各一层，执行一个真实
训练更新及验证/测试；这不是 ShanghaiTech 完整训练或精度结果。

额外只读对照源实现：100 组随机变换数据逐数组完全相等；100 张随机样本的
200 次 metric 匹配及聚合完全相等；合成 MAT→converter→BaseDetDataset 流程通过。
这些源代码对照只在移植验证时运行，发布代码和上述测试不读取源仓库。

核心源数值对照使用相同 seed 和真实 stock MMDetection，临时加载源类时去除
其 registry 装饰器以避免污染：初始化后 RNG 与 571 个 state_dict 张量全部
逐项完全相等；128×160 双 GT、R50、四层级、两层 encoder、六层 decoder、
30 queries 下，full-map 的 41 个输出项和 local 的 43 个输出项最大绝对误差
均为 `0.0`（含 detached debug 项）；DN query/mask/meta 和预测 points/scores/labels
也完全相同。该对照没有在目标中保留对源仓库的运行依赖。

## 文件与影响范围

新增文件（均相对目标仓库）：

```text
hq_det/models/pointdino/__init__.py
hq_det/models/pointdino/pointdino.py
hq_det/models/pointdino/pointdino_head.py
hq_det/models/pointdino/pointdino_layers.py
hq_det/models/pointdino/pointdino_fidt_head.py
hq_det/models/pointdino/pointdino_match_cost.py
hq_det/models/pointdino/pointdino_transforms.py
hq_det/models/pointdino/pointdino_metric.py
hq_det/models/pointdino/configs/pointdino_r50_shanghaitech_stage4_12e.py
hq_det/models/pointdino/configs/pointdino_r50_shanghaitech_stage4_local_12e.py
hq_det/models/pointdino/README.md
hq_det/models/pointdino/LICENSE
hq_det/tools/pointdino.py
scripts/run_train_pointdino.py
scripts/test_pointdino.py
scripts/pointdino_convert_shanghaitech.py
tests/test_pointdino_integration.py
tests/test_pointdino_smoke.py
tests/test_pointdino_fidt.py
tests/test_pointdino_data.py
tests/test_pointdino_runner.py
tests/test_pointdino_cli.py
tests/test_pointdino_scripts.py
scripts/POINTDINO_DATA_FORMAT.md
scripts/POINTDINO_CHECKPOINT_CHECK.md
scripts/POINTDINO_FILE_AUDIT.md
```

唯一修改的已有文件是 `pyproject.toml`，新增 PointDINO 包的 README/LICENSE
打包规则；原有依赖、包规则及已有模型行为不变。核心及公共导入文件均无需修改。
新包沿用移植文件中的 OpenMMLab copyright，
附源 Apache-2.0 LICENSE；具体独立类改造与 API 适配已在上文列明。

## 外部条件与验证边界

已验证环境为 **CPU**：PyTorch `2.1.0+cpu`、torchvision `0.16.0+cpu`、
MMCV `2.1.0`、MMEngine `0.10.7`、MMDetection `3.3.0`。
该环境安装在系统临时目录，未修改用户已有 Python 环境，也未加入仓库。
MMDetection 实际来自临时环境的 site-packages。

尚未进行真实 ShanghaiTech 12-epoch 训练、精度对比、CUDA/AMP 或多卡 DDP 实测。
这些是数据/权重和环境验证边界，不代表这些项目已经测试通过。
真实训练需要用户数据、适用的预训练/初始化权重；CUDA 运行另需与实际 GPU、
PyTorch 匹配的 CUDA/MMCV 构建。本配置没有 DINOv3 外部条件。
