import torch
import torch.nn as nn
import torch.nn.functional as F
from .extension import *

class Baseline(nn.Module):

    def __init__(self, n_classes=10):
        super(Baseline, self).__init__()

        self.conv = nn.Conv2d(3, 3, kernel_size = 3)

        self.relu = nn.ReLU(inplace = True)

        self.linear = nn.Linear(2700, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class Sample1(nn.Module):

    def __init__(self, n_classes=10):
        super(Sample1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.Conv2d(12, 6, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.Conv2d(6, 6, kernel_size = 3),
            nn.ReLU(inplace = True)
        )
        self.flatten = Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(4056, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def sample(**kwargs):
    return Sample1(**kwargs)


