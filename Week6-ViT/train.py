import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import ViT
from data import TrainDataset, TestDataset
from utils import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

cifar_train = torchvision.datasets.CIFAR10(root="../data", train=True, download=True)
cifardata = cifar_train.data / 255
mean_pic = torch.tensor(cifardata.mean(axis=(0))).permute(2, 0, 1)

train_and_valid = data.random_split(torchvision.datasets.CIFAR10(root="../data", train=True, download=True),
                                    [45000, 5000],
                                    generator=torch.Generator().manual_seed(42))
train_dataset = TrainDataset(train_and_valid[0])
valid_dataset = TestDataset(train_and_valid[1])
test_dataset = TestDataset(torchvision.datasets.CIFAR10(root="../data", train=False, download=True))

class Configuration:
    def __init__(self, version):
        ############### model hyperparameters ###############
        self.version = version
        self.N = 8
        self.patch_size = 4
        self.emb_dim = 128
        self.hidden_dim = 256
        self.num_heads = 4
        self.dropout = 0.05
        self.image_size = 32
        self.num_channels = 3
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_classes = 10
        ############### train hyperparameters ###############
        self.batch_size = 256
        self.lr = 1e-1
        self.weight_decay = 5e-4
        self.warmup_steps = 1000
        self.num_epochs = 75
        self.num_workers = 8
        #####################################################
    def __str__(self):
        return self.version

if __name__ == '__main__':
    cfg = Configuration('Version4')
    net = ViT(cfg).to(device)
    net.print_num_params()
    train_ViT(net, train_dataset, valid_dataset)