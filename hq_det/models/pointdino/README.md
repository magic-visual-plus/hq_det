# PointDINO 在 HQ-Det 中的接入

PointDINO 通过 HQ 的数据、模型和 trainer 接口训练与测试，检测核心保持独立的
`PointDINO*` 类与 registry 名称。标准数据使用 COCO 的
`images/annotations/categories`，把目标几何标注写成 `point: [x, y]`；
预测结果为 `InstanceData(points, scores, labels)`。

完整目录、字段及可运行入口说明见
[PointDINO 数据格式](../../../scripts/POINTDINO_DATA_FORMAT.md)。
`weiben_keti_split` 仅作结构参考，不读取它作为默认训练集，不修改数据，
也不把其中的框转换为点。新数据默认放在仓库的 `data/pointdino/`。

## HQ 继承关系

| PointDINO 组件 | HQ 基类 | 接入内容 |
|---|---|---|
| `PointDINODataset` | `CocoDetection` | 复用 HQ 图片读取与数据集接口，读取点标注 |
| `PointDINODatasetProvider` | `HQDatasetProvider` | 通过统一 provider 创建 train/valid/test 数据集 |
| `PointDINOModel` | `HQModel` | 完整模型 forward、loss、predict、checkpoint 接口 |
| `PointDINOTrainerArguments` | `HQTrainerArguments` | 增加点模型配置、原图尺寸与续训参数 |
| `PointDINOTrainer` | `HQTrainer` | 继承 `run`、setup、`train_step`、`optimizer_step` 等公共流程 |

专属 trainer 适配点数据、完整点损失、点指标、参数分组、源 epoch scheduler
以及 checkpoint 策略。配置仍用于构建 PointDINO 核心和优化器，训练主循环
由 HQTrainer 执行；可编辑脚本与 config CLI 都进入这一继承链。

公共 `hq_det/trainer.py` 只从原 `run()` 中提取四个扩展 hook：
`_before_training`、`_epoch_range`、`_finish_epoch`、`_step_epoch_scheduler`。
默认 hook 保留原流程和调用顺序，PointDINO 在自己的子类中覆盖。
原有 DINO、DINOHead、GroundingDINO、CoDETR 等模型核心实现没有因此改变。

HQ 的框增强模块改为在父类构建训练/验证增强时按需导入。PointDINO 使用自己的
点增强实现，因此不会因导入 HQTrainer 而加载 imgaug；这避免了服务器 NumPy 2
移除 `np.sctypes` 后，未使用的 imgaug 在入口导入阶段报错。原有框模型调用
原增强方法时仍使用原实现。这项调整不代表整个训练环境已完成 NumPy 2 兼容验证。

## 数据与尺寸

```text
data/pointdino/
    train/_annotations.coco.json
    train/<图片文件>
    valid/_annotations.coco.json
    valid/<图片文件>
    test/_annotations.coco.json     # 可选
    test/<图片文件>
```

每条 annotation 包含 `id`、`image_id`、`category_id`、`point: [x, y]`。
`area`、`segmentation` 和 `bbox` 不是必需字段；缺失 `point` 时不会从 bbox
推导点。负样本保留在 `images` 中，不为其创建 annotations。

类别 ID 可不连续，按 ID 排序映射到模型内部的连续标签；不同 ID 的名称可以
重复。各 split 必须保留一致的 ID 到名称映射，包括该 split 未出现的类别。
旧 `metainfo/data_list` 点 JSON 作为兼容格式继续支持。

`ignore_flag` 优先于 `iscrowd`，二者省略时默认 0；值 1 表示忽略点。
点使用原图像素坐标，不需要用户归一化或预先缩放。

默认 `image_size=None`，train/val/test 均保留原图尺寸。同 batch 的不同尺寸
图片只在右侧、底部 padding。仅显式整数或 `(宽, 高)` 才启用 resize：
训练同步变换图片和点；验证、测试先变换图片再加载原图 GT，预测还原到原图。
未 resize 时 pack 提供 `scale_factor=(1, 1)`。

默认指标距离为原图的 5 px、10 px，分数阈值 `0.5`，最优模型按
`point/f1@10px` 保存。显式 resize 不改变距离阈值的坐标单位，但可能改变
模型预测与指标数值，不能保证精度不变。默认 900 queries 的源模型仍要求
足够的特征候选；小图不满足 top-k 条件时，需要用户显式设置输入尺寸或 queries，
不会自动放大输入。

## 独立检测核心与算法

源 PointDINO 来自 `dinov3_dino_mmdet` 的 `mmdet_dino_2d_stage4` 实现。
当前模型使用 ResNet50；源项目中其他 DINOv3 实验不在本模型的依赖链内，
运行不需要源仓库、DINOv3 wrapper 或 DINOv3 权重。

```text
PointDINOModel(HQModel)
└── PointDINO
    ├── ResNet50 → ChannelMapper → 原生 deformable encoder
    ├── PointDINOTransformerDecoder
    ├── PointDINOCdnQueryGenerator
    ├── PointDINOHead
    │   ├── HungarianAssigner(FocalLossCost + PointDINOL1Cost)
    │   └── FocalLoss + L1Loss + 像素 Euclidean loss
    └── 可选 PointDINOFIDTAuxHead
```

| 源定制组件 | 独立类 |
|---|---|
| 修改后的 `DINO` | `PointDINO` |
| 修改后的 `DINOHead` | `PointDINOHead` |
| 修改后的 `DinoTransformerDecoder` | `PointDINOTransformerDecoder` |
| `PointCdnQueryGenerator` | `PointDINOCdnQueryGenerator` |
| `PointFIDTAuxHead` | `PointDINOFIDTAuxHead` |
| `PointL1Cost` | `PointDINOL1Cost` |
| 点标注 load/pack/resize/flip/crop | 对应 `PointDINO*` transform |
| `PointMetric` | `PointDINOMetric` |

直接复用原生 ResNet、ChannelMapper、encoder、decoder layer、attention、
positional encoding、HungarianAssigner、FocalLossCost、FocalLoss、L1Loss
和 DetDataPreprocessor。注册通过新包的 `__init__.py` 与配置
`custom_imports` 完成，没有覆盖已有 registry 或修改原 DINO 算法。

保留的核心参数与逻辑：

- proposal、reference refinement、head 回归和 DN 均为二维点；保留 top-k、
  detach 和 look-forward-twice。
- DN 正负点使用径向噪声，`point_noise_scale=0.01`，label noise `0.5`，
  dynamic DN budget `100`。
- Hungarian Focal cost `2.0`、Point L1 cost `20.0`；点 L1 训练权重 `5.0`。
- Euclidean 项按像素距离计算，除以正样本数和 **8**，再乘 `0.20`。
  评价阈值改为 5/10 px 不改变这个 loss 的 `/8`。
- FIDT target 为 `1 / (D ** (0.02 * D + 0.75) + 1)`，`D` 为最近 GT 距离。
  分支使用 `256→64/GN8` 和两倍上采样；local 支持半径仍为 **16 px**。
  零权重或 `enabled=False` 跳过分支；debug 统计不加入训练总 loss。
- 推理输出点、分数、标签，不做 NMS；FIDT 不参与预测。

## 训练配置与启动

先在已有 HQ 环境中安装项目：

```bash
python -m pip install -e .
python scripts/run_train_pointdino.py
python scripts/test_pointdino.py
```

编辑两个脚本中的 `TRAINING`、`TESTING`，配置数据路径、权重、类别和设备。
当前训练脚本保留用户设置：100 epochs，学习率 `1e-4`，第 70、90 epoch
衰减，backbone 学习率倍率 `0.1`，FIDT 权重 `0.0`、debug 关闭。
AdamW weight decay `1e-4` 和梯度裁剪 `0.1` 保留。

基础 `pointdino_r50_shanghaitech_stage4_12e.py` 及 local 配置保留源
12-epoch 实验预设，供选择和覆盖。脚本无参数启动使用字典中的用户设置；
带 config/`--` 参数启动使用 config CLI 设置，同样调用 PointDINOTrainer。
两种方式均默认原图输入。

```bash
python scripts/run_train_pointdino.py data/my_points checkpoints/point_init.pth
python scripts/test_pointdino.py data/my_points output/pointdino_2/best_model.pth
python scripts/run_train_pointdino.py --help
```

本次单类 Stage2 初始化权重不能直接加载多类头；不会自动跳过 shape 不匹配
参数。应使用匹配类别数的 PointDINO checkpoint，或设置
`load_checkpoint=None` 使用 backbone 预训练初始化。

完整 checkpoint 加载会关闭冗余的 pretrained 下载，保持参数名称不变。
`resume=False` 仅加载权重；`resume=True` 恢复 optimizer、scheduler、scaler、
epoch 和保存的训练状态。初始化文件缺少训练状态时不能用作完整续训文件。

训练保存 `best_model.pth`、最新 `ckpt.pth`（可通过 `checkpoint_name` 改名）、
按保留策略保存的 `epoch_N.pth`，并输出 `pointdino_config.py`、`results.csv`。
测试输出 `metrics.json`。模型选优使用 `point/f1@10px`。

## 文件位置与验证边界

主要接口文件为 `hq_det/pointdino_data.py`、`hq_det/pointdino_dataset.py`、
`hq_det/models/pointdino/pointdino_hq.py`、`hq_det/tools/train_pointdino.py`。
检测核心与独立变换位于本目录，脚本参数适配位于 `hq_det/tools/pointdino.py`。
保留移植代码中的 OpenMMLab copyright，许可证见本目录的 [LICENSE](LICENSE)。

本轮在独立 CPU 环境实际验证：Python 3.10、Torch 2.1.0+cpu、MMCV 2.1.0、
MMEngine 0.10.7、MMDetection 3.3.0，使用临时合成 COCO 点数据，不改写用户数据。

- HQ Dataset/Provider/Model/Trainer 继承关系及共享方法身份检查通过。
- 中文路径、空标注、ignore、非连续类别 ID、重复显示名、跨 split 映射检查通过；
  原图及显式 resize 管线与此前点数据管线的张量结果一致。
- 实际调用两个脚本的 `main([])`，加载用户提供的单类 Stage2 权重，使用完整
  R50、900 queries、6 层 encoder 和 6 层 decoder，在不同尺寸原图上完成两个
  优化器更新、验证、checkpoint 保存、重新加载和测试，loss 与指标均为有限值。
  该检查保留脚本的 FIDT 零权重设置；启用 FIDT 时另行验证了 loss/backward
  及其分支和共享 neck 的非零梯度。
- 小型配置执行 2 epochs、每轮 3 个 batch、累积间隔 2，确认每轮正确更新两次；
  从第 1 轮恢复至第 2 轮，与连续训练的全部模型张量、AdamW 状态、scheduler
  和训练统计逐项一致。
- 公共 HQTrainer 新 hook 展开后与原 `run()` 的 AST 一致，其余原有方法未变；
  导入新组件后，原 DINO 配置、registry 和完整 model build 通过。

以上是功能检查，不是完整数据集精度结果。检查脚本和合成输出位于系统临时目录，
没有恢复用户已删除的 tests 目录或审计文档。

实际训练需要点标注、图片与匹配的权重；CUDA、AMP、多卡及完整训练精度应在
对应环境单独验证。PointDINO 的运行依赖包括 HQ 环境、MMDetection、MMEngine
及匹配 PyTorch 的 MMCV 扩展，没有 DINOv3 外部依赖。
