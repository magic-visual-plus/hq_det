# YOLO26 在 HQ-DET 框架下的综合配置报告

> 检查日期：2026-07-15  
> 检查范围：当前本地目录中的 YOLO26 训练、评测、数据集、DDP 和 Ultralytics 8.4.7 recipe  
> 版本要求：`ultralytics==8.4.7`


## 1. 总体架构

YOLO26 没有直接调用 Ultralytics 的 `DetectionTrainer` 或 `DetectionValidator`，而是采用以下组合方式：

```text
scripts/run_train_yolo26.py
    -> hq_det.tools.train_yolo26.run()
        -> Yolo26Trainer
            -> HQ-DET 训练生命周期、日志、checkpoint 和工业评测
            -> UltralyticsV847Recipe
                -> augmentation / dataloader / MuSGD
                -> scheduler / warmup / AMP / EMA
                -> loss / runtime / postprocess
            -> HQYOLO26
                -> 固定版本的 YOLO26 模型、criterion 和底层算子
```

框架自己持有训练编排和接口，Ultralytics 8.4.7 只提供与模型格式强绑定的模型结构、criterion 和底层增强/NMS 原语。

### 1.1 对其他模型的隔离

- 只有 `Yolo26Trainer` 会显式创建 `UltralyticsV847Recipe`。
- 通用 `HQTrainer`、普通 `YoloTrainer`、RT-DETR、DINO、RF-DETR 等不会自动切换到 MuSGD 或该 recipe。
- `HQYOLODataset` 只有在显式传入 `augmentation_strategy`、`metadata_mode` 等参数时才进入 YOLO26 新数据路径。
- 全局依赖锁定为 `ultralytics==8.4.7`。这不会自动改变其他 trainer 的调用逻辑，但同一环境中所有直接导入 Ultralytics 的代码都会使用 8.4.7，这是进程级依赖边界。

## 2. 框架级接口

接口定义在 `hq_det/training/interfaces.py`，8.4.7 具体实现位于 `hq_det/training/recipes/ultralytics_v847/`。

### 2.1 接口清单

| 接口 | 主要方法 | 作用 | 其他模型复用情况 |
| --- | --- | --- | --- |
| `AugmentationStrategy` | `build(dataset)`、`close_mosaic(dataset, hyp=None)` | 构建训练/验证增强图并关闭混合增强 | 接口可复用；当前实现要求 YOLO 风格 dataset，其他数据结构需适配 |
| `DataLoaderStrategy` | `build(dataset, **kwargs)` | 构建可复现、支持 sampler 的 dataloader | PyTorch Dataset 基本可直接复用 |
| `OptimizerStrategy` | `build(model, hyp, context)` | 构建 optimizer，返回累积步数、缩放 decay、实际 LR 等 | 接口可复用；当前参数分组含 YOLO 检测头规则 |
| `SchedulerStrategy` | `build(...)`、`apply_warmup(...)` | 构建 epoch scheduler，并逐 batch warmup | 可复用，需要提供约定的 `hyp` 字段 |
| `EMAStrategy` | `build(model, decay, tau)` | 创建参数和浮点 buffer 的 EMA 副本 | 标准 PyTorch 模型可直接复用 |
| `PrecisionStrategy` | `resolve_amp(...)`、`build_scaler(...)` | 检查 AMP 并创建 GradScaler | CUDA PyTorch 模型可复用 |
| `RuntimeStrategy` | `initialize_seed(...)`、`prepare_model(...)` | 设置随机种子、确定性行为和冻结层 | 可复用；冻结层命名规则需匹配模型 |
| `LossStrategy` | `compute(...)`、`on_epoch_end(...)` | 接入模型 loss 和 progressive loss 生命周期 | 当前实现要求模型提供 `compute_loss()`/`update_epoch()` |
| `PostprocessStrategy` | `process(...)` | 转换为 HQ-DET `PredictionResult` | 当前实现要求模型提供 `postprocess()` |
| `DetectionTrainingRecipe` | `build_config(args)`、`validate_arguments(args)` | 组合全部策略，形成版本化训练配方 | 新模型可实现新 recipe 或选择性组合现有组件 |

### 2.2 接口数据类

```python
OptimizerContext(
    num_classes,
    dataset_size,
    epochs,
    global_batch_size,
    nominal_batch_size,
)

OptimizerBuildResult(
    optimizer,
    accumulate,
    scaled_weight_decay,
    iterations,
    name,
    learning_rate,
    momentum,
)

SchedulerBuildResult(scheduler, lr_lambda)
```

### 2.3 8.4.7 具体实现

| 类/方法 | 作用 |
| --- | --- |
| `UltralyticsV847Recipe` | 组合并校验完整 8.4.7 detection recipe |
| `UltralyticsV847Augmentation` | 固化 8.4.7 增强顺序和 `close_mosaic` |
| `UltralyticsV847DataLoader` | 提供 `InfiniteDataLoader`、worker 复用和固定 seed |
| `UltralyticsV847Optimizer` | 实现 `auto -> MuSGD`、参数分组、三倍 head LR、accumulate 和 decay 缩放 |
| `MuSGD` | 固化 Muon + SGD 更新和五步 Newton-Schulz 正交化 |
| `UltralyticsV847Scheduler` | linear/cosine scheduler 和逐 batch warmup |
| `UltralyticsV847EMA` / `ModelEMA847` | 实现 8.4.7 EMA |
| `UltralyticsV847Precision` | AMP 兼容性检查和 GradScaler |
| `UltralyticsV847Runtime` | seed、确定性 CUDA 行为和冻结层 |
| `UltralyticsV847Loss` | 调用 YOLO26 criterion 并推进 progressive loss |
| `UltralyticsV847Postprocess` | 将输出接入 HQ-DET 工业评测结构 |

### 2.4 其他模型如何调用

复用完整 recipe：

```python
from hq_det.training.recipes.ultralytics_v847 import UltralyticsV847Recipe

recipe = UltralyticsV847Recipe()  # 检查 ultralytics 是否严格为 8.4.7
recipe.validate_arguments(args)
hyp = recipe.build_config(args)
```

只复用 MuSGD optimizer：

```python
from hq_det.training import OptimizerContext
from hq_det.training.recipes.ultralytics_v847 import UltralyticsV847Optimizer

context = OptimizerContext(
    num_classes=num_classes,
    dataset_size=len(train_dataset),
    epochs=epochs,
    global_batch_size=batch_size,
    nominal_batch_size=64,
)
result = UltralyticsV847Optimizer().build(model, hyp, context)
optimizer = result.optimizer
accumulate = result.accumulate
```

只复用 EMA：

```python
from hq_det.training.recipes.ultralytics_v847 import UltralyticsV847EMA

ema = UltralyticsV847EMA().build(model, decay=0.9999, tau=2000.0)
# 每次真实 optimizer.step() 后调用
ema.update(model)
```

只复用 dataloader：

```python
from hq_det.training.recipes.ultralytics_v847 import UltralyticsV847DataLoader

loader = UltralyticsV847DataLoader().build(
    dataset,
    batch_size=batch_size,
    workers=8,
    collate_fn=dataset.collate_fn,
    shuffle=True,
    sampler=distributed_sampler,
    rank=rank,
)
```

接口本身是模型无关的，但 augmentation、loss、postprocess 和部分 optimizer 分组规则带有 YOLO 约定。非 YOLO 模型应实现同一接口或增加薄适配层，不应直接假设全部具体组件天然兼容。

## 3. 数据集配置

### 3.1 目录格式

```text
dataset_root/
    train/
        _annotations.coco.json
        *.jpg / *.png
    valid/
        _annotations.coco.json
        *.jpg / *.png
```

COCO JSON 中的 `file_name` 必须能相对对应 split 目录找到图片。

### 3.2 类别 ID 映射

当前实现会按训练集原始 COCO category ID 排序，并映射为连续的 `0..nc-1`。验证集按类别名称对齐训练集映射，而不是假设 train/valid 原始 category ID 相同。

```text
COCO category ID: {1, 3, 7}
YOLO class ID:    {0, 1, 2}
```

重复类别名称、未知验证类别和越界 ID 会提前报错。映射同时写入 `<output_path>/yolo26_recipe_manifest.json`。

### 3.3 `image_id` 修复

8.4.7 的某些增强只保留 `im_file`，不保证保留自定义 `image_id`。当前 `_finalize_sample()` 依次：

1. 使用增强结果中的数值 `image_id`。
2. 将数字字符串转换为整数。
3. 若得到图片路径等非数值对象，metadata 模式从 `_image_ids[index]` 恢复原 COCO image ID。
4. 非 metadata 模式无法恢复时使用 dataset index。

因此不会再出现：

```text
ValueError: invalid literal for int() with base 10: '/path/to/image.jpg'
```

### 3.4 标注处理

- 过滤 `iscrowd=1`、非法框和零面积框。
- 框裁剪到图像边界，去除重复的 `class + bbox` 标注。
- 转换为 normalized `xywh`，collate 后再次检查 `0 <= cls < nc`。
- 训练使用 square batch；验证使用 `rect=True`、stride 32、pad 0.5。


## 4. YOLO26 支持范围

### 4.1 模型系列

支持完整 YOLO26 检测系列：`yolo26n`、`yolo26s`、`yolo26m`、`yolo26l`、`yolo26x`，通过 `--scale n|s|m|l|x` 选择。

### 4.2 模型来源

- `.pt`：加载可信的 Ultralytics YOLO26 checkpoint。
- `.pth`：加载 state_dict，并根据 `--scale` 重建对应 YAML 结构。
- `.yaml` / `.yml`：从模型结构开始训练。
- `--scratch`：使用 `yolo26{scale}.yaml`，不加载标准预训练权重。
- `--p2`：使用 `yolo26{scale}-p2.yaml` 或用户提供的 P2 checkpoint。

当前集成针对 bbox detection，不提供 segmentation、pose、OBB 或 classification 训练入口。

### 4.3 已支持工作

- 单卡训练和 PyTorch DDP 多卡数据并行。
- 训练中逐 epoch 工业评测及独立工业评测。
- EMA、best model 选择和 `.pt`/`.pth` 双格式保存。
- Python API 批量图片推理，返回 `PredictionResult`。
- end-to-end YOLO26 输出和普通 NMS 输出后处理。

## 5. 冻结的 8.4.7 训练配置

### 5.1 基础默认值

| 配置 | 默认值 | 说明 |
| --- | ---: | --- |
| `epochs` | 100 | 由 `--num_epoches` 覆盖 |
| `imgsz` | 1024（CLI） | 自动向上取整到 32 的倍数 |
| `batch` | 4（CLI） | 表示全局 batch，不是单卡 batch |
| `amp` | `True` | CUDA 上先执行 8.4.7 `check_amp()` |
| `seed` | 0 | 每个 rank 使用 `seed + 1 + rank` |
| `deterministic` | `True` | 可用 `--non_deterministic` 关闭 |
| `optimizer` | `auto` | `auto` 和显式 `MuSGD` 均只使用 MuSGD |
| `lr0` | 0.01 | 显式 `MuSGD` 使用；`auto` 会按规则重算 |
| `lrf` | 0.01 | 最终 LR 系数 |
| `momentum` | 0.937 | 显式 MuSGD 使用；auto 固定为 0.9 |
| `weight_decay` | 0.0005 | 按全局 batch 与 accumulate 缩放 |
| `nbs` | 64 | nominal batch size |
| `warmup_epochs` | 3.0 | warmup 至少 100 iteration |
| `warmup_momentum` | 0.8 | warmup momentum 起点 |
| `warmup_bias_lr` | 0.1 | auto optimizer 会改为 0.0 |
| `box` | 7.5 | box loss gain |
| `cls` | 0.5 | classification loss gain |
| `dfl` | 1.5 | DFL loss gain；日志第三项不是 GIoU |
| `max_grad_norm` | 10.0 | 固定梯度裁剪阈值 |
| `close_mosaic` | 10 | 最后 10 个 epoch 关闭混合增强 |
| `iou` | 0.7 | NMS IoU threshold |
| `max_det` | 300 | 每张图最大检测数 |

### 5.2 数据增强默认值和顺序

```text
Mosaic                         p=1.0
CopyPaste                      p=0.0
RandomPerspective + LetterBox  degrees=0.0, translate=0.1, scale=0.5,
                               shear=0.0, perspective=0.0
MixUp                          p=0.0
CutMix                         p=0.0
Albumentations                 p=1.0（内部变换由配置决定）
RandomHSV                      h=0.015, s=0.7, v=0.4
VerticalFlip                   p=0.0
HorizontalFlip                 p=0.5
Format                         normalized xywh + batch_idx
```

验证增强为 `LetterBox(scaleup=False) -> Format`。

到达 `epochs - close_mosaic` 时，同时将 Mosaic、CopyPaste、MixUp 和 CutMix 概率设置为 0，并重建 transform/dataloader iterator。

旧 HQ 增强参数 `augment_proba`、`augment_split_*`、`augment_foreground_*` 仍保留在 CLI 中兼容旧脚本，但 YOLO26 8.4.7 recipe 不执行这些增强。

### 5.3 MuSGD 选择逻辑

```text
accumulate = max(round(nbs / global_batch), 1)

scaled_weight_decay =
    weight_decay * global_batch * accumulate / nbs

iterations =
    ceil(dataset_size / max(global_batch, nbs)) * epochs
```

`--optimizer auto`：

```text
lr_fit = round(0.002 * 5 / (4 + nc), 6)
momentum = 0.9
warmup_bias_lr = 0.0

iterations > 10000:
    lr = 0.01
    Muon/SGD 系数 = 0.1/1.0

iterations <= 10000:
    lr = lr_fit
    Muon/SGD 系数 = 0.5/0.5
```

`--optimizer MuSGD` 使用用户传入的 `lr0` 和 `momentum`，Muon/SGD 系数仍按 iterations 长短选择。

参数分为 weight、normalization、bias、matrix/Muon 四类，再按普通 LR 和三倍 LR 拆分，共八组。匹配检测头等特定名称的参数使用三倍 LR。

### 5.4 Scheduler、warmup、EMA 和 loss

- 默认 linear：LR 系数从 `1.0` 下降到 `lrf`。
- `--cos_lr`：使用 8.4.7 one-cycle cosine 形式从 `1.0` 下降到 `lrf`。
- warmup iteration：`max(round(warmup_epochs * batches_per_epoch), 100)`。
- warmup 逐 batch 插值 accumulate、普通组 LR、bias LR 和 momentum。
- 每次真实 optimizer step 后更新 EMA。

EMA 固定要求：

```text
decay = 0.9999
tau = 2000.0
decay(x) = 0.9999 * (1 - exp(-x / 2000))
```

验证、best model 和常规 checkpoint 均使用 EMA 模型。YOLO26 criterion 每个 epoch 结束后调用一次 `update()`，推进 end-to-end progressive loss。

## 6. DDP 数据并行配置

### 6.1 当前实现

多卡训练使用 PyTorch `DistributedDataParallel`：

- 每个进程、每张 GPU 各保存一份完整模型。
- `DistributedSampler` 将训练数据拆分给不同 rank。
- 每个 epoch 调用 `sampler.set_epoch(epoch)`。
- rank 0 执行完整验证，其他 rank 在 barrier 等待。
- `sync_bn=True` 为框架继承默认值，多卡时转换为 `SyncBatchNorm`。
- `find_unused_parameters=True`。

这是模型复制后的数据并行，不是把模型层拆到不同 GPU 的模型并行。

### 6.2 全局 batch 语义

YOLO26 的 `--batch_size` 是全局 batch：

```text
rank_batch = global_batch / world_size
```

四卡使用 `--batch_size 32` 时，每卡 batch 为 `32 / 4 = 8`。这在 batch 规模上等价于单卡 batch 32，不等价于单卡 batch 8。多卡速度也不会严格达到 4 倍，因为还有梯度同步、rank 0 验证、数据读取和通信开销。

### 6.3 DDP loss 修复

```python
backward_loss = loss
if torch.distributed.is_initialized():
    backward_loss = loss * torch.distributed.get_world_size()

scaler.scale(backward_loss).backward()
return loss, info
```

YOLO criterion 的 loss 已按当前 rank batch 聚合，而 DDP 默认对各 rank 梯度求平均。反传前乘 `world_size` 可抵消该梯度平均，与 8.4.7 多卡训练语义对齐。返回原始 `loss`，所以 `results.csv` 和进度条不会被卡数人为放大。

## 7. 工业评测配置

评测有意不改成官方 class-aware validator，仍使用 HQ-DET 的工业通用定义：

1. 先按 `eval_class_names` 选择缺陷类别；未指定时使用全部类别。
2. 选中的 GT 和预测类别统一折叠为类别 `0`（`ng`）。
3. 只判断缺陷是否被检测到，不要求缺陷类别判断正确。
4. mAP 为 class-agnostic COCO AP@[0.50:0.95]。
5. precision、recall、F1、FNR 和 confidence 从 IoU=0.50 的 PR 数据中选择最佳 F1 点。
6. best model 依据该工业 mAP 保存。

训练结果字段包括：

```text
mAP, precision, recall, f1_score, fnr, confidence
train/box_loss, train/cls_loss, train/dfl_loss
val/box_loss, val/cls_loss, val/dfl_loss
lr/pg*
```

因此 HQ-DET 工业 mAP 与官方 Ultralytics class-aware mAP50-95 不是同一个指标，不能按数值直接一一比较。

## 8. 启动指令

以下命令均在服务器项目目录执行：

```bash
cd /root/hq_det_original
```

### 8.1 环境检查

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

必须输出 `8.4.7`。如果 PyTorch 2.6+ 加载可信 `.pt` 时出现 `weights_only` 错误，可在确认 checkpoint 来源可信后执行：

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

### 8.2 单卡训练 YOLO26m

```bash
python scripts/run_train_yolo26.py \
  --data_path "/autodl-fs/data/liudongdong/weiben_keti_split/" \
  --output_path "/root/hq_det_original/checkpoints/yolo26m_hq_train/" \
  --scale m \
  --load_checkpoint "/root/hq_det_original/checkpoints/yolo/yolo26m.pt" \
  --num_epoches 135 \
  --batch_size 8 \
  --image_size 1280 \
  --optimizer auto \
  --devices 0
```

### 8.3 四卡 DDP，每卡 batch 8

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  --master_port=29517 \
  scripts/run_train_yolo26.py \
  --data_path "/autodl-fs/data/liudongdong/weiben_keti_split/" \
  --output_path "/root/hq_det_original/checkpoints/yolo26m_hq_train_v2/" \
  --scale m \
  --load_checkpoint "/root/hq_det_original/checkpoints/yolo/yolo26m.pt" \
  --num_epoches 135 \
  --batch_size 32 \
  --image_size 1280 \
  --optimizer auto \
  --devices 0,1,2,3
```

这里 `32` 是全局 batch，四个 rank 各自读取 8 张图片。`--batch_size` 必须能被进程数整除；多卡必须用 `torchrun` 启动，只用普通 `python` 加多个 `--devices` 不构成正确的多进程 DDP。

### 8.4 四卡 DDP，保持全局 batch 8

若要与单卡 `batch_size=8` 保持相同全局 batch，而不是每卡都用 8：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  --master_port=29517 \
  scripts/run_train_yolo26.py \
  --data_path "/autodl-fs/data/liudongdong/weiben_keti_split/" \
  --output_path "/root/hq_det_original/checkpoints/yolo26m_global_batch8/" \
  --scale m \
  --load_checkpoint "/root/hq_det_original/checkpoints/yolo/yolo26m.pt" \
  --num_epoches 135 \
  --batch_size 8 \
  --image_size 1280 \
  --optimizer auto \
  --devices 0,1,2,3
```

此时每卡 batch 为 2。

### 8.5 从 YAML 开始训练

```bash
python scripts/run_train_yolo26.py \
  --data_path "/autodl-fs/data/liudongdong/weiben_keti_split/" \
  --output_path "/root/hq_det_original/checkpoints/yolo26m_scratch/" \
  --scale m \
  --scratch \
  --num_epoches 135 \
  --batch_size 8 \
  --image_size 1280 \
  --optimizer auto \
  --devices 0
```

P2 结构在该命令上增加 `--p2`。

### 8.6 独立工业评测

```bash
python scripts/run_evaluate_yolo26.py \
  --data_path "/autodl-fs/data/liudongdong/weiben_keti_split/" \
  --model "/root/hq_det_original/checkpoints/yolo26m_hq_train/best_model.pt" \
  --output_path "/root/hq_det_original/checkpoints/yolo26m_hq_train/eval/" \
  --scale m \
  --batch_size 8 \
  --image_size 1280 \
  --device 0
```

只评测部分缺陷时增加：

```bash
--eval_class_names "划伤（轻度）,气孔,裂纹"
```

### 8.7 Python 训练与评测接口

```python
from hq_det.tools import train_yolo26

trainer = train_yolo26.run(
    data_path="/path/to/dataset",
    output_path="/path/to/output",
    scale="m",
    load_checkpoint="/path/to/yolo26m.pt",
    num_epoches=135,
    batch_size=8,
    image_size=1280,
    optimizer="auto",
    devices=[0],
)

trainer, metrics = train_yolo26.evaluate(
    data_path="/path/to/dataset",
    model="/path/to/best_model.pt",
    output_path="/path/to/eval",
    scale="m",
    batch_size=8,
    image_size=1280,
    device=0,
)
```

### 8.8 Python 推理接口

当前没有单独的 `run_predict_yolo26.py`，但模型类已提供批量推理：

```python
import cv2

from hq_det.models.yolo26 import HQYOLO26

class_id2names = {
    0: "划伤（轻度）",
    1: "气孔",
    # 按 yolo26_recipe_manifest.json 补齐其余连续类别
}

model = HQYOLO26(
    class_id2names=class_id2names,
    model="/path/to/best_model.pt",
    scale="m",
)
model.to("cuda:0").eval()

image = cv2.imread("/path/to/image.jpg")
result = model.predict([image], bgr=True, confidence=0.25)[0]
print(result.bboxes, result.scores, result.cls)
```

## 9. 输出文件

```text
<output_path>/
    results.csv
    yolo26_recipe_manifest.json
    best_model.pt
    best_model.pth
    ckpt.pt
    ckpt.pth
    plots/
        epoch*/pr_curve.csv
```

- `results.csv`：每个 epoch 的工业指标、train/val loss 和参数组 LR。
- `yolo26_recipe_manifest.json`：版本、模型来源、类别映射、全局 batch、optimizer 派生值和完整 frozen config。
- `best_model.*`：按工业 mAP 选择的最佳 EMA 权重。
- `ckpt.*`：最近一个 epoch 保存的 EMA 权重。
