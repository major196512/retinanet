import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P7 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def freeze_modules(self, freeze):
        for n, w in self.named_parameters():
            w.requires_grad = not freeze

    def forward(self, inputs, global_feature):
        C3, C4, C5 = inputs

        P3_1 = self.P3_1(C3)
        P3_cat = P3_1 + global_feature
        P3_2 = self.P3_2(P3_cat)
        P3_down = self.pool(P3_2)

        P4_1 = self.P4_1(C4)
        P4_cat = P4_1 + P3_down
        P4_2 = self.P4_2(P4_cat)
        P4_down = self.pool(P4_2)

        P5_1 = self.P5_1(C5)
        P5_cat = P5_1 + P4_down
        P5_2 = self.P5_2(P5_cat)

        P6 = self.P6(P5_2)

        P7_1 = self.relu(P6)
        P7_2 = self.P7(P7_1)

        return [P3_2, P4_2, P5_2, P6, P7_2]
