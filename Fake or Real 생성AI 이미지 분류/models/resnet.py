import torch
import torch.nn as nn
import timm

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = timm.create_model('resnet18', pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 625),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(625, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.is_sigmoid = True

    def forward(self, x):
        x = self.model(x)
        if self.is_sigmoid:
            x = torch.sigmoid(x)
        return x