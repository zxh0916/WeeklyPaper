import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from d2l import torch as d2l
import random
import time
from modules import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cifar_train = torchvision.datasets.CIFAR10(root="../data", train=True, download=True)
cifardata = cifar_train.data / 255
mean_pic = torch.tensor(cifardata.mean(axis=(0))).permute(2, 0, 1)
channel_mean, channel_std = cifardata.mean(axis=(0, 1, 2)), cifardata.std(axis=(0, 1, 2))

def subtract_mean(pic):
    return pic-mean_pic.to(pic.device)


class TrainDataset(data.Dataset):
    def __init__(self, dataset, aug=True):
        super().__init__()
        self.dataset = dataset
        if aug:
            self.trans = transforms.Compose([transforms.ToTensor(),
                                             transforms.Lambda(subtract_mean),
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.ConvertImageDtype(torch.float)])
        else:
            self.trans = transforms.Compose([transforms.ToTensor(),
                                             transforms.Lambda(subtract_mean),
                                             transforms.ConvertImageDtype(torch.float)])
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
                                         transforms.Lambda(subtract_mean),
                                         transforms.ConvertImageDtype(torch.float)])
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return (self.trans(self.dataset[index][0]),
                self.dataset[index][1])


def evaluate_loss_acc(net, data_iter, criterion, device=device):
    """使用GPU计算模型在数据集上的精度。"""
    net.eval()  # 设置为评估模式
    loss = []
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for input, target in data_iter:
            input = input.to(device)
            target = target.to(device)
            output = net(input)
            loss.append(float(criterion(output, target).item()))
            metric.add(d2l.accuracy(output, target), target.numel())
    return sum(loss) / len(loss), metric[0] / metric[1]


def get_lr(optimizer):
    return (optimizer.state_dict()['param_groups'][0]['lr'])


def train_DenseNet(net,
                   batch_size,
                   lr,
                   num_epochs,
                   weight_decay=1e-4):
    
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    net.apply(init_weights)
    
    train_iter = data.DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=2)
    valid_iter = data.DataLoader(valid_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=2)
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=lr,
                                weight_decay=weight_decay,
                                momentum=0.9)
    def lr_lambda(epoch):
        if epoch < num_epochs * 0.5:
            return 1.
        elif epoch < num_epochs * 0.75:
            return 0.1
        else:
            return 0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=False)
    scheduler_name = str(scheduler.__class__).split('.')[-1][:-2]
    writer = SummaryWriter(f'runs/DenseNet_CIFAR_L={net.L}_k={net.k}_theta={net.theta}_block={net.block}_dropout={net.dropout}')
    criterion = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        tic = time.time()
        metric = d2l.Accumulator(3)
        net.train()
        for input, target in train_iter:
            timer.start()
            optimizer.zero_grad()
            input, target = input.to(device), target.to(device)
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * input.shape[0],
                           d2l.accuracy(output, target),
                           input.shape[0])
            timer.stop()
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        valid_loss, valid_acc = evaluate_loss_acc(net, valid_iter, criterion, device)
        writer.add_scalar('train/loss', train_loss, global_step=epoch+1)
        writer.add_scalar('train/accuracy', train_acc, global_step=epoch+1)
        writer.add_scalar('valid/loss', valid_loss, global_step=epoch+1)
        writer.add_scalar('valid/accuracy', valid_acc, global_step=epoch+1)
        writer.add_scalar('learning rate', get_lr(optimizer), global_step=epoch+1)
        scheduler.step()
        toc = time.time()
        print(f"epoch {epoch+1:3d}, train loss: {train_loss:.4f}, train accuracy: {train_acc:.4f}, valid loss: {valid_loss:.4f}, valid accuracy: {valid_acc:.4f}, time: {toc-tic:.4f}")
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'valid loss {valid_loss:.3f}, valid acc {valid_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

if __name__ == '__main__':
    train_and_valid = data.random_split(torchvision.datasets.CIFAR10(root="../data", train=True, download=True),
                                        [45000, 5000],
                                        generator=torch.Generator().manual_seed(42))

    train_dataset = TrainDataset(train_and_valid[0], aug=True)
    valid_dataset = TestDataset(train_and_valid[1])
    test_dataset = TestDataset(torchvision.datasets.CIFAR10(root="../data", train=False, download=True))
    
    options = {
        #  [L,   k,   theta, block,       num_classes, dropout]
        1: [40,  12,  1.,    BRC,         10,          0      ],
        2: [100, 12,  1.,    BRC,         10,          0      ],
        3: [100, 24,  1.,    BRC,         10,          0      ],
        4: [100, 12,  0.5,   Bottleneck,  10,          0      ],
        5: [250, 24,  0.5,   Bottleneck,  10,          0      ],
        6: [190, 40,  0.5,   Bottleneck,  10,          0      ]
    }
    # net = DenseNet(*options[5]).to(device)
    net = DenseNet(L=100, k=24, theta=0.5, block=Bottleneck, num_classes=10, dropout=0).to(device)
    
    net.print_num_params()

    train_DenseNet(net,
                   batch_size=64,
                   lr=0.1,
                   num_epochs=200,
                   weight_decay=1e-4)

    test_iter = data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loss, test_acc = evaluate_loss_acc(net, test_iter, nn.CrossEntropyLoss())
    print(test_loss, test_acc)

    torch.save(net.state_dict(), f'DenseNet_CIFAR_L={net.L}_k={net.k}_theta={net.theta}_block={net.block}_dropout={net.dropout}_acc={test_acc:.4f}.pth')
