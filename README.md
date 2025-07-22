# HQ-DET: High Quality Image Detection Framework

HQ-DET是一个轻量级的目标检测框架，集成了多个主流的目标检测模型，并提供了统一的训练、评估和对比实验接口。

## 项目特点

- 支持多个主流目标检测模型
- 统一的训练和评估接口
- 完整的实验对比分析
- 高质量的实现和优化

## 支持的模型

- RT-DETRv2 R50
- RTMDet Large
- DINO R50
- YOLOv12 Large
- RF-DETR Base
- CO-DETR R50

## 环境要求

- Python >= 3.6
- CUDA支持
- 其他依赖项见 `pyproject.toml`

## 安装

### linux
```bash
git clone [repository_url]
cd hq_det

# 安装依赖
pip install -e .
```

### windows
```bat
python.exe -m pip install --upgrade pip
python.exe -m pip install openmim
python.exe -m mim install mmcv==2.1.0 mmdet==3.3.0 mmengine==0.10.7
python.exe -m pip install peft==0.10.0 transformers==4.43.3 tokenizers<0.19
git clone [repository_url]
pip install -e .
```

## 项目结构

```
hq_det/
├── benchmark/          # 模型对比实验
├── hq_det/            # 核心代码
├── scripts/           # 训练和评估脚本
├── test/              # 测试代码
└── test_notebook/     # Jupyter notebooks
```

## 使用方法

### 训练模型

```bash
# 使用示例脚本进行训练
python scripts/run_train_dino.py [dataset_dir] [model_path]
```

## 实验对比

详细的模型对比实验请参考 [benchmark](./benchmark) 目录：

- 所有模型在相同的硬件环境(RTX 4090)下进行训练和测试
- 训练时间统一对齐到12小时
- 详细的对比分析请参考 [benchmark.ipynb](./benchmark/benchmark.ipynb)

## 实验结果

| 模型 | mAP | FPS |
|------|-----|-----|
| RT-DETRv2 R50 | 44.4 | 43.83 |
| RTMDet Large | 41.6 | 23.09 |
| DINO R50 | 39.8 | 24.75 |
| YOLOv12 Large | 35.1 | - |
| RF-DETR Base | 44.5 | 21.97 |
| CO-DETR R50 | 38.1 | 6.12 |

**FPS测试说明：**
- 测试环境：RTX 4090 GPU
- 输入尺寸：1024×1024
- 批处理大小：1
- 预热轮数：100
- 测试轮数：1000
- 使用FP32精度进行推理
- 包含了数据加载和预处理时间


各模型的详细实验结果可在对应目录下查看：
- 训练日志：`nohup.out`
- 评估结果：`results.csv`

## 贡献

欢迎提交Issue和Pull Request来帮助改进项目。

## 许可证

本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

- 作者：Xiaochuan Zou
- 邮箱：zouxiaochuan@163.com 
