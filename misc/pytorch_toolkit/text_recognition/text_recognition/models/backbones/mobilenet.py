import torch.nn as nn
import torchvision.models

architectures = {
    'v2': torchvision.models.mobilenet_v2,
    'v3_large': torchvision.models.mobilenet_v3_large,
    'v3_small': torchvision.models.mobilenet_v3_small,
}


class MobileNet(nn.Module):
    def __init__(self, arch, in_channels=1):
        super().__init__()
        assert arch in architectures.keys(), f'{arch} not in possible options: {architectures.keys()}'
        model = architectures[arch](pretrained=True)
        self.features = model.features
        if in_channels == 1:
            first_conv = self.features[0][0]
            self.features[0][0] = nn.Conv2d(1, first_conv.out_channels, first_conv.kernel_size, first_conv.stride, first_conv.padding, False)

    def forward(self, x):
        x = self.features(x)
        return x
