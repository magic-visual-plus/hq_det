# PointDINO 数据格式与一键启动说明

## 1. 是否与以前的 JSON 一致

与源 PointDINO 的 `BaseDetDataset` 点标注 JSON **完全一致**，已有源
PointDINO 的 `train_point.json` / `test_point.json` 不需要改变字段。

HQ 原有 DINO 的 COCO 框标注（`images`、`annotations`、`categories`、`bbox`）
是另一种格式，不能直接作为这个点模型的输入。此处使用
`metainfo` + `data_list` + `instances.point`，不需要 `bbox`、`area`、
`segmentation`、`iscrowd`。不会自动把框中心当作你的点监督。

## 2. 数据集文件结构

脚本默认与原 ShanghaiTech 结构一致：

```text
data/shanghaitech_part_b/              ← data_path
├── train_point.json                  ← train_ann_file
├── test_point.json                   ← val_ann_file / test_ann_file
├── train_data/
│   ├── images/                       ← train_image_dir
│   │   ├── IMG_1.jpg
│   │   └── IMG_2.jpg
│   └── ground_truth/                 ← 仅 MAT 转换时需要
│       ├── GT_IMG_1.mat
│       └── GT_IMG_2.mat
└── test_data/
    ├── images/                       ← val_image_dir / test_image_dir
    │   └── IMG_1.jpg
    └── ground_truth/                 ← 仅 MAT 转换时需要
        └── GT_IMG_1.mat
```

准备好图片与 JSON 后，训练、测试都不读取 MAT，`ground_truth/` 可以没有。
验证集和测试集可共用一个 JSON，也可以分开。其他点数据集可以按自己习惯组织：

```text
data/my_points/                       ← data_path
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── images/
    ├── train/
    │   ├── a.jpg
    │   └── b.png
    ├── val/
    │   └── c.jpg
    └── test/
        └── d.jpg
```

对应训练参数：

```python
data_path='data/my_points',
train_ann_file='annotations/train.json',
val_ann_file='annotations/val.json',
train_image_dir='images/train',
val_image_dir='images/val',
```

对应测试参数：

```python
data_path='data/my_points',
test_ann_file='annotations/test.json',
test_image_dir='images/test',
```

路径解析规则：

```text
标注文件 = data_path / ann_file
图片文件 = data_path / image_dir / JSON 中的 img_path
```

例如 `data_path=data/my_points`、`train_image_dir=images/train`、
`img_path=a.jpg`，实际图片就是 `data/my_points/images/train/a.jpg`。
若 JSON 写的是 `images/train/a.jpg`，则设置 `train_image_dir=''`，
避免重复拼接。标注文件和图片目录也可用绝对路径；JSON 中建议使用相对路径，
便于换机器。图片支持 OpenCV 可读取的 JPG、PNG 等格式。

## 3. 完整可用 JSON 示例

一个 JSON 管理一个 split 的所有图片。下面包含有标注和没有目标的两张图片：

```json
{
  "metainfo": {
    "classes": ["point"]
  },
  "data_list": [
    {
      "img_id": 1,
      "img_path": "IMG_1.jpg",
      "width": 1024,
      "height": 768,
      "instances": [
        {
          "point": [120.25, 80.5],
          "point_label": 0,
          "ignore_flag": 0
        },
        {
          "point": [600.0, 300.75],
          "point_label": 0,
          "ignore_flag": 0
        }
      ]
    },
    {
      "img_id": 2,
      "img_path": "IMG_2.jpg",
      "width": 640,
      "height": 480,
      "instances": []
    }
  ]
}
```

| 字段 | 类型 | 要求与含义 |
|---|---|---|
| `metainfo` | object | 保留此对象，用于数据集元信息 |
| `metainfo.classes` | string array | 非空类别列表，ID 按列表顺序从 0 开始；默认 `['point']` |
| `data_list` | array | 图片记录列表；每个 split 一个列表 |
| `img_id` | integer | 图片标识，建议在同一 split 内唯一；不同 split 可重复 |
| `img_path` | string | 图片路径，按上面的 image_dir 规则拼接 |
| `width` | positive integer | 原始图片实际宽度，不是 resize 后宽度 |
| `height` | positive integer | 原始图片实际高度，不是 resize 后高度 |
| `instances` | array | 每个元素是一个点；没有目标时明确写 `[]` |
| `instances[i].point` | `[x, y]` | 原图像素坐标，可为小数，必须是有限数值 |
| `instances[i].point_label` | integer | 从 0 到类别数减 1；单类全部写 0 |
| `instances[i].ignore_flag` | 0 or 1 | 0=有效，1=忽略；可省略，默认 0，建议显式写 0 |

请按上表完整提供字段。`metainfo.classes` 在显式传 `class_names` 时可以由
参数提供，但为了数据可移植，推荐每个 JSON 都保存它。`instances` 即使省略
会被底层读取为空，也应明确写出，避免误把漏标图片当作负样本。

坐标约定：x 向右、y 向下，基于原图像素，通常满足
`0 <= x < width`、`0 <= y < height`。不要提前归一化为 0～1，不要提前
按训练输入尺寸缩放，不要把 `[x,y]` 写成 `[y,x]`。
训练管线会同步 resize/flip 图片和点；推理会把点还原到原图坐标后评价。
源 ShanghaiTech 转换工具会原样保留 MAT 点坐标，不额外取整或自动加减 1。

各 split 必须使用相同的 `classes` 顺序。示例：

```json
"classes": ["scratch_point", "defect_point"]
```

对应 `point_label=0` / `1`。新入口会自动设置模型类别数。
当前提供的初始化权重是**单类**，用于其他单类点任务可以直接加载；
若类别数变化，使用匹配类别数的 PointDINO checkpoint，或设
`load_checkpoint=None` 从 backbone 预训练初始化。不要把单类完整权重当作
多类头的完整初始化。原 PointDINOMetric 保留按位置汇总的匹配方式，
不输出分类别 AP；此行为没有因新增入口而改变。

## 4. 训练脚本如何修改

打开 `scripts/run_train_pointdino.py`，编辑 `TRAINING = dict(...)`。
常用参数已像原 `run_train_dino.py` 一样集中列出：

- 数据：`data_path`、两个 `ann_file`、两个 `image_dir`、`class_names`。
- 模型：`load_checkpoint`、`config_path`、`image_size=(宽,高)`。
- 训练：`num_epoches`、`batch_size`、`eval_batch_size`、`lr0`、
  `lr_backbone_mult`、`gradient_update_interval`、`num_data_workers`。
- 输出与续训：`output_path`、`resume`。
- 设备：`devices=None` 自动使用环境默认设备，`[]` 为 CPU，`[0]` 指定 GPU。
- 评价：`score_threshold`、`distance_thresholds`（正整数像素阈值）。

`config_path=None` 使用 Stage4 full-map；local FIDT 使用
`hq_det/models/pointdino/configs/pointdino_r50_shanghaitech_stage4_local_12e.py`。
`cfg_options` 提供最终覆盖，例如关闭 FIDT：

```python
cfg_options={'model.point_fidt_head.enabled': False}
```

默认 `num_epoches=12`，在第 11 epoch 衰减；修改 epochs 后，默认衰减点跟随
为倒数第一个 epoch。通过 `lr_milestones=[...]` 可以明确指定衰减点。
原 weight decay `1e-4`、梯度裁剪 `0.1`、DN、点匹配和 loss 设置保持不变。

已经执行 `python -m pip install -e .` 的环境中，从仓库根目录运行：

```bash
python scripts/run_train_pointdino.py
```

也可只用命令行覆盖数据根目录和权重：

```bash
python scripts/run_train_pointdino.py data/my_points checkpoints/point_init.pth
```

该命令仅覆盖这两个参数，其余仍取 `TRAINING`。因此自定义 ann/image 子目录
也需要在 `TRAINING` 中配置。默认权重路径通过仓库位置推导到上一层的
`point_dino_stage2_step2_init.pth`，对应本次提供的权重文件。

完整 checkpoint 的新参数入口会跳过冗余的 torchvision backbone 下载。
`resume=False` 只加载模型权重；`resume=True` 尝试恢复训练状态，checkpoint
需要含 optimizer/epoch 等状态。本次 `stage2_step2_init.pth` 是初始化用途，
应使用 `resume=False`。

多卡通过标准 `torchrun` 启动，例如 `torchrun --nproc_per_node=2
scripts/run_train_pointdino.py`，配合 `devices=[0,1]`。单进程不会悄悄忽略
请求的多张卡。CUDA/多卡是否可用还取决于运行环境，此次仅 CPU 验证。

## 5. 测试脚本如何修改

打开 `scripts/test_pointdino.py`，编辑 `TESTING = dict(...)`。
设置测试 `data_path`、`test_ann_file`、`test_image_dir`、训练后的
`load_checkpoint` 和相同的 `config_path` / 类别顺序：

```bash
python scripts/test_pointdino.py
python scripts/test_pointdino.py data/my_points output/pointdino/epoch_12.pth
```

默认测试权重是新训练输出目录下的 `epoch_12.pth`。若训练 epochs 不等于 12，
需要改为实际生成的 checkpoint。测试入口只需要测试 JSON 和图片，
不要求训练集文件存在。输出点精确率、召回率、F1、定位误差、TP/FP/FN；
阈值单位是原图像素。

原 CLI 形式仍保留（出现 `--` 参数或首参数为 `.py` config 时使用该模式）：

```bash
python scripts/run_train_pointdino.py --help
python scripts/run_train_pointdino.py --data-root data/shanghaitech_part_b --load-from ../point_dino_stage2_step2_init.pth --cfg-options model.backbone.init_cfg=None
python scripts/test_pointdino.py hq_det/models/pointdino/configs/pointdino_r50_shanghaitech_stage4_12e.py output/pointdino/epoch_12.pth --data-root data/shanghaitech_part_b
```

原 CLI 模式使用 config 参数，不读取脚本的 `TRAINING` / `TESTING` 字典。

## 6. MAT 转 JSON

只有原始 ShanghaiTech MAT 标注需要使用转换工具；手工生成上述 JSON 的
其他点数据集不需要 MAT。

```bash
python scripts/pointdino_convert_shanghaitech.py --img-dir data/shanghaitech_part_b/train_data/images --gt-dir data/shanghaitech_part_b/train_data/ground_truth --out data/shanghaitech_part_b/train_point.json
python scripts/pointdino_convert_shanghaitech.py --img-dir data/shanghaitech_part_b/test_data/images --gt-dir data/shanghaitech_part_b/test_data/ground_truth --out data/shanghaitech_part_b/test_point.json
```
