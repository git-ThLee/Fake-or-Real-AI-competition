import torch
import torch.nn as nn

class Custom(nn.Module):
    def __init__(self):
        super(Custom, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding='valid')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding='valid')
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(p=0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=492032, out_features=256)  # 수정된 부분
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
