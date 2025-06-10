# 目标检测模型对比实验

本目录包含了多个主流目标检测模型的对比实验结果。我们对比了以下几个模型：

## 实验模型

1. **RT-DETRv2 R50**
2. **RTMDet Large**
3. **DINO R50**
4. **YOLOv12 Large**
5. **RF-DETR Large**
6. **CO-DETR R50**

## 实验设置

- 所有模型在相同的硬件环境(RTX 4090)下进行训练和测试, 训练时间对齐到12h
- 使用相同的评估指标进行对比
- 详细的训练日志和实验结果可在各模型对应的目录中查看

## 结果分析
### 对比分析
详细的对比分析请参考 [benchmark.ipynb](./benchmark.ipynb) 文件，其中包含了模型性能的对比图表和分析。

### 各实验结果
详细的实验结果分析请参考各模型目录下的结果文件：
- 训练日志：`nohup.out`
- 评估结果：`results.csv`

完整的实验结果文件可以从以下链接下载：
https://rookie-ai.oss-cn-hangzhou.aliyuncs.com/exper-0606.zip?OSSAccessKeyId=LTAI5tH4rq3aaiq8pzYXArsN&Expires=3.6e%2B97&Signature=KFM5BV0Y2W87IMe8P9I6Iu8Yqcc%3D

## 注意事项

1. 所有实验使用相同的评估数据集
2. 实验环境保持一致，确保公平对比
3. 详细的训练参数和配置可在各模型目录下查看

