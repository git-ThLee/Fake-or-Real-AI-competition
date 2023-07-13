import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# MobileNet 모델 정의 (Dropout 추가)
class MobileNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):  # 이진 분류 설정 및 Dropout 비율 추가
        super(MobileNet, self).__init__()

        def conv_dw(in_planes, out_planes, stride):
            return nn.Sequential(
                nn.Conv2d(in_planes, in_planes, 3, stride, 1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            # ... more layers ...
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout 계층 추가
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x