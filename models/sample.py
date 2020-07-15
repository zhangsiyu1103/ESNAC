import torch
import torch.nn as nn
import torch.nn.functional as F
from .extension import *

class Baseline(nn.Module):

    def __init__(self, n_classes=10):
        super(Baseline, self).__init__()

        self.conv = nn.Conv2d(3, 3, kernel_size = 3)

        self.relu = nn.ReLU(inplace = True)
        self.flatten = Flatten()

        self.linear = nn.Linear(2700, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x



class GroundTruth0(nn.Module):

    def __init__(self, n_classes=10):
        super(GroundTruth0, self).__init__()
        self.flatten = Flatten()

        self.linear1 = nn.Linear(3072, 4096)

        self.linear2 = nn.Linear(4096, n_classes)

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class GroundTruth1(nn.Module):

    def __init__(self, n_classes=10):
        super(GroundTruth1, self).__init__()
        self.flatten = Flatten()

        self.linear1 = nn.Linear(3072, 128)

        self.linear2 = nn.Linear(128, n_classes)

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class GroundTruth2(nn.Module):

    def __init__(self, n_classes=10):
        super(GroundTruth2, self).__init__()
        self.flatten = Flatten()

        self.linear1 = nn.Linear(108, 128)

        self.linear2 = nn.Linear(128, n_classes)

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x



class GroundTruth3(nn.Module):

    def __init__(self, n_classes=10):
        super(GroundTruth3, self).__init__()
        self.flatten = Flatten()

        self.linear1 = nn.Linear(108, 512)

        self.linear2 = nn.Linear(512, n_classes)

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x



class GroundTruth4(nn.Module):

    def __init__(self, n_classes=10):
        super(GroundTruth4, self).__init__()

        self.conv = nn.Conv2d(3, 3, kernel_size = 3)

        self.relu1 = nn.ReLU(inplace = True)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(48, 512)

        self.linear2 = nn.Linear(512, n_classes)

        self.relu2 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x




class Sample1(nn.Module):

    def __init__(self, n_classes=10):
        super(Baseline, self).__init__()

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
            nn.Linear(5400, 5400),
            nn.ReLU(inplace = True),
            nn.Linear(5400, 5400),
            nn.ReLU(inplace = True),
            nn.Linear(5400, n_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class Sample0(nn.Module):

    def __init__(self, n_classes=10):
        super(Sample0, self).__init__()

        self.flatten = Flatten()

        self.linear1 = nn.Linear(3072, 8192)

        self.linear2 = nn.Linear(8192, n_classes)

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class Sample2(nn.Module):

    def __init__(self, n_classes=10):
        super(Sample2, self).__init__()

        self.flatten = Flatten()

        self.linear1 = nn.Linear(108, 1024)

        self.linear2 = nn.Linear(1024, n_classes)

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class Sample3(nn.Module):

    def __init__(self, n_classes=10):
        super(Sample3, self).__init__()

        self.flatten = Flatten()

        self.linear1 = nn.Linear(108, 5120)

        self.linear2 = nn.Linear(5120, n_classes)

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class Sample4(nn.Module):

    def __init__(self, n_classes=10):
        super(Sample4, self).__init__()

        self.conv = nn.Conv2d(3, 6, kernel_size = 3)

        self.relu1 = nn.ReLU(inplace = True)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(96, 5120)

        self.linear2 = nn.Linear(5120, n_classes)

        self.relu2 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x




def sample0(**kwargs):
    return Sample0(**kwargs)


def sample3(**kwargs):
    return Sample3(**kwargs)

def sample4(**kwargs):
    return Sample4(**kwargs)



def groundtruth(**kwargs):
    return GroundTruth0(**kwargs)




