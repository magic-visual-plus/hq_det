from mmengine.model import BaseModule

import torch
import torch.nn as nn
from mmdet.models import SwinTransformer, ResNet

class LearnableResize(BaseModule):
    def __init__(self, channels=3, scale_factor=2):
        super().__init__()
        # 这是一个标准的卷积，不是 Depthwise
        self.conv = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=scale_factor, 
            stride=scale_factor, 
            bias=False
        )
        
        # --- 精确的初始化逻辑 ---
        with torch.no_grad():
            # 1. 先把所有权重设为 0（包括跨通道的连接）
            self.conv.weight.zero_()
            
            # 2. 只对对角线上的通道（in_ch == out_ch）进行 Resize 初始化
            init_value = 1.0 / (scale_factor ** 2)
            for i in range(channels):
                # 每一个对应通道的卷积核设为平均分布
                self.conv.weight[i, i, :, :] = init_value
                
    def forward(self, x):
        return self.conv(x)


class ResizeSwinTransformer(SwinTransformer):
    def __init__(self, image_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 在 backbone 之前添加一个可学习的 Resize 模块
        assert image_size % 1024 == 0
        self.learnable_resize = LearnableResize(scale_factor=image_size // 1024)
        print(image_size // 1024, "x Resize applied in ResizeSwinTransformer")

    def forward(self, x):
        # 先通过可学习的 Resize 模块调整输入特征图的尺寸
        x = self.learnable_resize(x)
        # 然后再通过 SwinTransformer 的前向传播
        return super().forward(x)


class ResizeResNet(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 在 backbone 之前添加一个可学习的 Resize 模块
        self.learnable_resize = LearnableResize(scale_factor=4)

    def forward(self, x):
        # 先通过可学习的 Resize 模块调整输入特征图的尺寸
        x = self.learnable_resize(x)
        # 然后再通过 SwinTransformer 的前向传播
        return super().forward(x)