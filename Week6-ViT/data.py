from torch.utils import data
import torchvision
from torchvision import transforms
import torch

class TrainDataset(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.ConvertImageDtype(torch.float),
                                         transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                              [0.2470, 0.2435, 0.2616],
                                                              inplace=True)])
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return (self.trans(self.dataset[index][0]),
                self.dataset[index][1])
    
class TestDataset(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.ConvertImageDtype(torch.float),
                                         transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                              [0.2470, 0.2435, 0.2616],
                                                              inplace=True)])
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return (self.trans(self.dataset[index][0]),
                self.dataset[index][1])
    
cifar_train = torchvision.datasets.CIFAR10(root="../data", train=True, download=True)
cifardata = cifar_train.data / 255
mean_pic = torch.tensor(cifardata.mean(axis=(0))).permute(2, 0, 1)