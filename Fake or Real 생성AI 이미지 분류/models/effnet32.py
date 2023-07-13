from torch import nn
import timm

class EffNet32(nn.Module):
    def __init__(self):
        super(EffNet32, self).__init__()
        self.model = timm.create_model(model_name='efficientnet_b0', pretrained=True)
        self.model.classifier = nn.LazyLinear(1)
        self.is_sigmoid = True

    def forward(self, x):
        x = self.model(x)
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

