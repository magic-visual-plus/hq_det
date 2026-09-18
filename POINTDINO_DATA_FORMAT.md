# PointDINO 数据格式与训练入口

PointDINO 的标准数据结构与 HQ 其他模型一致，使用 COCO 风格的
`images`、`annotations`、`categories`，每条标注使用 `point: [x, y]`
表示目标位置。`bbox` 不能代替 `point`；只有框的标注会报错，不会自动取框中心。

`weiben_keti_split` 仅作为目录和 JSON 结构参考。本功能不修改该数据集，
也不将其中的框转换成点。请把真实点标注放入独立的 `data/pointdino` 目录。

## 数据目录

脚本默认 `data_path=REPO_ROOT / 'data' / 'pointdino'`，其中 `REPO_ROOT`
是 `hqdet` 仓库目录：

```text
hqdet/
└── data/
    └── pointdino/
        ├── train/
        │   ├── _annotations.coco.json
        │   ├── image_1.jpg
        │   └── image_2.png
        ├── valid/
        │   ├── _annotations.coco.json
        │   └── image_3.jpg
        └── test/                        # 可选独立测试集
            ├── _annotations.coco.json
            └── image_4.jpg
```

默认训练读取 `train`，验证和测试读取 `valid`。使用独立测试集时，在
`scripts/test_pointdino.py` 的 `TESTING` 中设置：

```python
test_ann_file='test/_annotations.coco.json',
test_image_dir='test',
```

其他目录布局可通过 `train_ann_file`、`val_ann_file`、`test_ann_file` 和
对应的 `image_dir` 设置。路径解析规则是：

```text
标注文件 = data_path / ann_file
图片文件 = data_path / image_dir / images[i].file_name
```

例如 `train_image_dir='train'`、`file_name='image_1.jpg'`，图片路径为
`data/pointdino/train/image_1.jpg`。如果 `file_name` 已包含 `train/`，应把
`train_image_dir` 设为空字符串，避免重复拼接。建议 JSON 使用相对文件名。

## 完整 JSON 示例

每个 split 保存一个 JSON。下面有两个类别、两张图片；第二张图片没有标注，
是合法负样本。示例图片尺寸仅用于说明，并不限制实际输入尺寸。

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image_1.jpg",
      "width": 960,
      "height": 720
    },
    {
      "id": 2,
      "file_name": "image_2.png",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 101,
      "image_id": 1,
      "category_id": 7,
      "point": [120.25, 80.5],
      "iscrowd": 0
    },
    {
      "id": 102,
      "image_id": 1,
      "category_id": 42,
      "point": [600.0, 300.75],
      "ignore_flag": 0
    }
  ],
  "categories": [
    {"id": 42, "name": "defect_point"},
    {"id": 7, "name": "scratch_point"}
  ]
}
```

| 字段 | 要求 |
|---|---|
| `images`、`annotations`、`categories` | 三个列表都要提供；无目标 split 的 `annotations` 可以为空 |
| `images[i].id` | 整数，同一 split 内唯一 |
| `images[i].file_name` | 非空图片文件名，按 `image_dir` 拼接 |
| `images[i].width`、`height` | 建议完整填写原始图片的实际宽高，使用正整数 |
| `annotations[i].id` | 整数，同一 split 内唯一 |
| `annotations[i].image_id` | 指向本 JSON 中已定义的图片 ID |
| `annotations[i].category_id` | 指向本 JSON 中已定义的类别 ID |
| `annotations[i].point` | 两个有限数值 `[x, y]`，允许小数，使用原图像素坐标 |
| `categories[i].id` | 唯一整数，不要求连续或从 0 开始 |
| `categories[i].name` | 非空显示名称；不同类别 ID 可以使用相同名称 |
| `iscrowd`、`ignore_flag` | 可选，值为 0 或 1；0 为有效点，1 为忽略点 |

`area`、`segmentation`、`bbox` 不是点检测必需字段。每张负样本图片仍应放在
`images` 中，只需不为它创建标注。若同时提供 `ignore_flag` 和 `iscrowd`，
以 `ignore_flag` 为准；省略 `ignore_flag` 时读取 `iscrowd`，两者都省略则为 0。
忽略点与有效点分开，不作为有效 GT 参与训练与点指标计算。

坐标约定：x 向右、y 向下，通常满足 `0 <= x < width`、
`0 <= y < height`。不要提前归一化、缩放或把 `[x, y]` 写成 `[y, x]`。

## 类别映射与初始化权重

内部训练标签按 `categories.id` 从小到大映射为 `0, 1, ...`。
上述示例始终是 `7 → 0`、`42 → 1`，与 `categories` 列表的排列顺序无关。
映射按类别 ID 区分，不按名称去重；名称相同但 ID 不同仍是两个类别。

train、valid、test 必须保持相同的类别 ID 和 ID 到名称的映射。某个 split
没有某一类目标时，也要在 `categories` 中保留该类别。`class_names=None`
表示从标注读取；显式提供名称时，顺序必须与类别 ID 排序后的顺序一致。

本次提供的 `point_dino_stage2_step2_init.pth` 是**单类**初始化权重，不能直接
加载上述两类示例的完整模型。类别数或参数形状不兼容时会报错，不会自动跳过
不匹配的分类头。多类任务应提供匹配类别数的 PointDINO 权重，或设置
`load_checkpoint=None` 使用 backbone 初始化。默认点指标按位置汇总，
不是分类别 COCO bbox AP。

## 原图尺寸与评价坐标

训练和测试默认 `image_size=None`，保留每张图片的原始尺寸；基础 full-map
和 local 配置的 train/val/test 管线也没有 Resize。同一 batch 的图片尺寸
不同时，只在右侧、底部 padding，不缩放图片或点坐标。

只有显式设置 `image_size=640` 或 `image_size=(1024, 768)` 才启用 resize。
整数表示正方形，二元组顺序为 `(宽, 高)`；指定尺寸时不保持原宽高比例。
训练先加载点再同步 resize 图片和点；验证、测试先 resize 图片，再加载原图
GT，预测点则还原到原图坐标。未 resize 时使用 `scale_factor=(1, 1)`。

默认评价距离为原图像素的 **5 px 和 10 px**，分数阈值为 `0.5`，最优模型
按 `point/f1@10px` 选择。显式 resize 后，距离阈值仍表示原图像素；这不保证
指标数值不变，因为输入内容的采样尺度改变后，模型预测也可能改变。

默认模型保留源实现的 900 queries。过小图片可能产生不足 900 个特征候选，
触发源模型的 top-k 限制；需要时由用户显式设置 `image_size` 或调整 queries，
入口不会为此自动缩放图片。

## 训练、测试与续训

在已有 HQ Python 环境中安装本项目，再从仓库根目录运行：

```bash
python -m pip install -e .
python scripts/run_train_pointdino.py
```

编辑训练脚本的 `TRAINING` 字典可以集中配置数据、权重和训练参数。
当前保留的用户配置是 100 epochs、学习率 `1e-4`、backbone 倍率 `0.1`，
完成第 70、90 轮后学习率乘以 `0.1`（第 71、91 轮开始使用新的学习率），
FIDT 权重为 `0.0` 且关闭 debug。
这与基础模型配置文件中的源 12-epoch 实验预设不同；无参数启动使用脚本字典。
Euclidean loss 的 `/8` 和 local FIDT 的 `16` 像素半径不因评价阈值改变。

也可以只覆盖数据根目录和初始化权重：

```bash
python scripts/run_train_pointdino.py data/my_points checkpoints/point_init.pth
```

这只覆盖这两个参数，其余设置仍取 `TRAINING`。默认数据目录是独立的
`data/pointdino`，需要用户提供真实点标注和图片。

测试前编辑 `TESTING`，选择与训练相同的模型配置、类别映射和训练后权重：

```bash
python scripts/test_pointdino.py
python scripts/test_pointdino.py data/my_points output/pointdino_2/best_model.pth
```

测试默认使用 `valid`，也可以按前面的设置指向可选 `test`；不要求训练图片存在。
输出包括点精确率、召回率、F1、定位误差、TP/FP/FN，以及 `metrics.json`。

训练输出目录包含 `best_model.pth`、最新的 `ckpt.pth`、按配置保留的
`epoch_N.pth`、`pointdino_config.py` 和 `results.csv`。`checkpoint_name`
可以修改最新 checkpoint 的文件名。

`resume=False` 只初始化模型权重；`resume=True` 从训练 checkpoint 恢复
optimizer、scheduler、scaler、epoch 和已保存的训练状态。
单类 Stage2 文件用于初始化，应使用 `resume=False`。完整 checkpoint 加载时
会关闭冗余的 backbone pretrained 下载；它不意味着不兼容参数会被跳过。

带 `.py` config 或 `--` 参数的命令行模式仍可用：

```bash
python scripts/run_train_pointdino.py --help
python scripts/test_pointdino.py --help
```

该模式从 config 读取设置，不读取 `TRAINING` / `TESTING` 字典；它同样通过
`PointDINOTrainer` 接入 HQ 的训练、测试流程。

## 旧标注兼容

已有 `metainfo` + `data_list` + `instances.point/point_label` 文件仍可读取，
用于兼容早期 PointDINO 数据。新数据请采用上面的 COCO 点格式。
两种格式都使用原图像素点，不从框产生点，也不会改写输入 JSON 或图片。
