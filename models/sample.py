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



class GroundTruth5(nn.Module):

    def __init__(self, n_classes=10):
        super(GroundTruth5, self).__init__()

        self.flatten = Flatten()

        self.linear1 = nn.Linear(108, 48)

        self.linear2 = nn.Linear(48, n_classes)

        self.relu1 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


class GroundTruth5_(nn.Module):

    def __init__(self, n_classes=10):
        super(GroundTruth5_, self).__init__()

        self.flatten = Flatten()

        self.linear1 = nn.Linear(108, 48)

        self.linear2 = nn.Linear(48, n_classes)

        self.relu1 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x



class GroundTruth6(nn.Module):

    def __init__(self, n_classes=10):
        super(GroundTruth6, self).__init__()

        self.conv = nn.Conv2d(3, 3, kernel_size = 3)

        self.relu1 = nn.ReLU(inplace = True)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(48, 64)

        self.linear2 = nn.Linear(64, n_classes)

        self.relu2 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x



class GroundTruth7(nn.Module):

    def __init__(self, n_classes=1):
        super(GroundTruth7, self).__init__()

        self.flatten = Flatten()

        self.linear1 = nn.Linear(108, 48)

        self.linear2 = nn.Linear(48, n_classes)

        self.relu1 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x



class GroundTruth8(nn.Module):

    def __init__(self, n_classes=10):
        super(GroundTruth8, self).__init__()

        self.conv = nn.Conv2d(3, 2, kernel_size = 3)

        self.relu1 = nn.ReLU(inplace = True)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(32, 10)

        self.linear2 = nn.Linear(10, n_classes)

        self.relu2 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


class GroundTruth9(nn.Module):

    def __init__(self, n_classes=10):
        super(GroundTruth9, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size = 3)

        self.relu1 = nn.ReLU(inplace = True)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(96, 128)

        self.relu2 = nn.ReLU(inplace = True)

        self.linear2 = nn.Linear(128, n_classes)



    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


class GroundTruth10(nn.Module):

    def __init__(self, n_classes=10):
        super(GroundTruth10, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, kernel_size = 3)

        self.relu1 = nn.ReLU(inplace = True)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(588, 96)

        self.relu2 = nn.ReLU(inplace = True)

        self.linear2 = nn.Linear(96, n_classes)



    def forward(self, x):
        x = self.conv1(x)
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



class Sample5(nn.Module):

    def __init__(self, n_classes=10):
        super(Sample5, self).__init__()

        self.flatten = Flatten()

        self.linear1 = nn.Linear(108, 2400)

        self.linear2 = nn.Linear(2400, n_classes)

        self.relu1 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x



class Sample6(nn.Module):

    def __init__(self, n_classes=10):
        super(Sample6, self).__init__()

        self.conv = nn.Conv2d(3, 6, kernel_size = 3)

        self.relu1 = nn.ReLU(inplace = True)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(96, 1960)

        self.linear2 = nn.Linear(1960, n_classes)

        self.relu2 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


class Sample7(nn.Module):

    def __init__(self, n_classes=1):
        super(Sample7, self).__init__()

        self.flatten = Flatten()

        self.linear1 = nn.Linear(108, 2400)

        self.linear2 = nn.Linear(2400, n_classes)

        self.relu1 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


class Test7(nn.Module):

    def __init__(self, n_classes=1):
        super(Test7, self).__init__()

        self.flatten = Flatten()

        self.linear1 = nn.Linear(108, 45)

        self.linear2 = nn.Linear(45, n_classes)

        self.relu1 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x




class Sample8(nn.Module):

    def __init__(self, n_classes=10):
        super(Sample8, self).__init__()

        self.conv = nn.Conv2d(3, 6, kernel_size = 3)

        self.relu1 = nn.ReLU(inplace = True)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(96, 2400)

        self.linear2 = nn.Linear(2400, n_classes)

        self.relu2 = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


class Sample9(nn.Module):

    def __init__(self, n_classes=10):
        super(Sample9, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size = 3)

        self.relu1 = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(6, 6, kernel_size = 3)

        self.relu2 = nn.ReLU(inplace = True)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(24, 512)

        self.relu3 = nn.ReLU(inplace = True)

        self.linear2 = nn.Linear(512, 512)

        self.relu4 = nn.ReLU(inplace = True)

        self.linear3 = nn.Linear(512, n_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.relu4(x)
        x = self.linear3(x)
        return x



class Sample10(nn.Module):

    def __init__(self, n_classes=10):
        super(Sample10, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size = 3)

        self.relu1 = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(6, 6, kernel_size = 3)

        self.relu2 = nn.ReLU(inplace = True)

        self.flatten = Flatten()

        self.linear1 = nn.Linear(864, 512)

        self.relu3 = nn.ReLU(inplace = True)

        self.linear2 = nn.Linear(512, 512)

        self.relu4 = nn.ReLU(inplace = True)

        self.linear3 = nn.Linear(512, n_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.relu4(x)
        x = self.linear3(x)
        return x





def sample0(**kwargs):
    return Sample0(**kwargs)


def sample3(**kwargs):
    return Sample3(**kwargs)

def sample4(**kwargs):
    return Sample4(**kwargs)

def sample5(**kwargs):
    return Sample5(**kwargs)

def sample6(**kwargs):
    return Sample6(**kwargs)

def sample7(**kwargs):
    return Sample7(**kwargs)

def sample8(**kwargs):
    return Sample8(**kwargs)

def sample9(**kwargs):
    return Sample9(**kwargs)

def sample10(**kwargs):
    return Sample10(**kwargs)



def groundtruth(**kwargs):
    return GroundTruth0(**kwargs)




