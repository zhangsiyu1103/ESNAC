import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class IMAGENET(object):

    def __init__(self, batch_size=128, val_batch_size=512, num_workers=32):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)),
        ])

        train_dataset = torchvision.datasets.ImageFolder(root='/workspace/szhang/imagenet/ILSVRC/Data/CLS-LOC/train',
            transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder(root='/workspace/szhang/imagenet/ILSVRC/Data/CLS-LOC/val',
            transform=test_transform)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=val_batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True)

class IMAGENETVal(object):

    def __init__(self, batch_size=128, val_batch_size=512, num_workers=32, val_size = 5000):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)),
        ])

        train_dataset = torchvision.datasets.ImageFolder(root='/workspace/szhang/imagenet/ILSVRC/Data/CLS-LOC/train',
            transform=train_transform)

        total_size = len(train_dataset)
        indices = list(range(total_size))
        train_size = total_size - val_size
        train_sampler = SubsetRandomSampler(indices[:train_size])
        val_sampler = SubsetRandomSampler(indices[train_size:])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        self.val_loader = DataLoader(train_dataset, batch_size=val_batch_size,
            sampler=val_sampler, num_workers=num_workers, pin_memory=True)


