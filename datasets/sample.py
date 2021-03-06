import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TrainDataset(Dataset):
    def __init__(self):
        self.samples = torch.load("./datasets/data/sample/groundtruth10/train.pt")

    def __len__(self):
        return len(self.samples["image"])

    def __getitem__(self, idx):
        sample = (self.samples["image"][idx], self.samples["category"][idx])
        return sample


class TestDataset(Dataset):
    def __init__(self):
        self.samples = torch.load("./datasets/data/sample/groundtruth10/test.pt")

    def __len__(self):
        return len(self.samples["image"])

    def __getitem__(self, idx):
        sample = (self.samples["image"][idx], self.samples["category"][idx])
        return sample

class Artificial(object):

    def __init__(self, batch_size = 32, num_workers=4):

        train_dataset = TrainDataset()
        test_dataset = TestDataset()


        self.train_loader = DataLoader(train_dataset, batch_size = batch_size,
            shuffle = True, num_workers = num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size = batch_size,
            shuffle = True, num_workers = num_workers, pin_memory=True)


class TrainDataset1(Dataset):
    def __init__(self):
        self.samples = torch.load("./datasets/data/sample/groundtruth5/train.pt")

    def __len__(self):
        return len(self.samples["image"])

    def __getitem__(self, idx):
        sample = (self.samples["image"][idx], self.samples["category"][idx])
        return sample


class TestDataset1(Dataset):
    def __init__(self):
        self.samples = torch.load("./datasets/data/sample/groundtruth5/test.pt")

    def __len__(self):
        return len(self.samples["image"])

    def __getitem__(self, idx):
        sample = (self.samples["image"][idx], self.samples["category"][idx])
        return sample

class Artificial1(object):

    def __init__(self, batch_size = 32, num_workers=4):

        train_dataset = TrainDataset1()
        test_dataset = TestDataset1()


        self.train_loader = DataLoader(train_dataset, batch_size = batch_size,
            shuffle = True, num_workers = num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size = batch_size,
            shuffle = True, num_workers = num_workers, pin_memory=True)
