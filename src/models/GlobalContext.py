import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class GlobalNet(nn.Module):
    def __init__(self, C5_size, feature_size):
        super(GlobalNet, self).__init__()

        self.I4_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.I4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.I3_1 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.I3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def freeze_modules(self, freeze):
        for n, w in self.named_parameters():
            w.requires_grad = not freeze

    def forward(self, C5):
        batch, channel, _, _ = C5.size()

        I4_1 = self.I4_1(C5)
        I4_up = F.interpolate(I4_1, scale_factor=2, mode='bilinear', align_corners=True)
        I4_2 = self.I4_2(I4_up)
        I4_relu = self.relu(I4_2)

        I3_1 = self.I3_1(I4_relu)
        I3_up = F.interpolate(I3_1, scale_factor=2, mode='bilinear', align_corners=True)
        I3_2 = self.I3_2(I3_up)

        return I3_2
