import torch.nn as nn
import torchvision.models

architectures = {
    'v2': torchvision.models.mobilenet_v2,
    'v3_large': torchvision.models.mobilenet_v3_large,
    'v3_small': torchvision.models.mobilenet_v3_small,
}


class MobileNet(nn.module):
    def __init__(self, arch):
        super().__init__()
        assert arch in architectures.keys(), f'{arch} not in possible options: {architectures.keys()}'
        model = architectures[arch](pretrained=True)
        self.features = model.features

    def forward(self, x):
        x = self.features(x)
        return x
