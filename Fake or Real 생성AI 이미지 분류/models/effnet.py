from torch import nn
import timm

class EffNet(nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        self.model = timm.create_model(model_name='efficientnet_b1', pretrained=True)

        # self.model.classifier = nn.LazyLinear(1)
        # self.is_sigmoid = True

        # 원래의 classifier 제거
        self.model.classifier = nn.Identity()
        
        # 드롭아웃 추가
        self.dropout = nn.Dropout(0.5)

        # 새로운 classifier에 드롭아웃 적용
        self.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(1280, 1)
        )
        self.is_sigmoid = True

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        if self.is_sigmoid:
            x = nn.Sigmoid()(x)
        return x

# class EffNet(nn.Module):
#     def __init__(self):
#         super(EffNet, self).__init__()
#         self.model = timm.create_model(model_name='efficientnet_b0', pretrained=True)
#         self.model.classifier = nn.LazyLinear(1)
#         self.is_sigmoid = True

#     def forward(self, x):
#         x = self.model(x)
#         if self.is_sigmoid:
#             x = nn.Sigmoid()(x)
#         return x

