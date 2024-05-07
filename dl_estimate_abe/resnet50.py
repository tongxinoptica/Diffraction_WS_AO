import torch
import torch.nn as nn
from torchvision import models


# 定义一个新的 ResNet 类来修改输出
class ResNet50(nn.Module):
    def __init__(self, num_classes=135):
        super(ResNet50, self).__init__()
        # 加载预训练的 ResNet-50
        original_model = models.resnet50(weights=None)

        # 除了全连接层之外复制所有层
        self.features = nn.Sequential(*list(original_model.children())[:-2])

        self.conv_reducer = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 135, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(135),
            nn.ReLU(inplace=True),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # 通过修改后的ResNet模型
        x = self.features(x)
        x = self.conv_reducer(x)
        # 调整输出以匹配所需的输出维度 (4, 135, 1, 1)
        x = self.global_avg_pool(x)
        return x
