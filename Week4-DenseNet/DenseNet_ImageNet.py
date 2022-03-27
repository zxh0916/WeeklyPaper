import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from d2l import torch as d2l
import random
import time
import pandas as pd
from PIL import Image
from modules import *
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[2]:


path = '../data/dog-breed-identification/'
train_csv = pd.read_csv(path + 'labels.csv')
label_list = sorted(train_csv['breed'].unique().tolist())
test_csv = pd.read_csv(path + 'sample_submission.csv')


# In[3]:


class TrainDataset(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        
        self.trans = transforms.Compose([transforms.RandomCrop(224),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.ColorJitter(brightness=0.2,
                                                                contrast=0.2,
                                                                saturation=0.2,
                                                                hue=0.2),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.4736, 0.4504, 0.3909],
                                                              std=[0.2655, 0.2607, 0.2650],
                                                              inplace=True)])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        resize = transforms.Resize(random.randint(256, 480))
        return self.trans(resize(image)), label


# In[4]:

def stack_crops(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

class ValidDataset(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.trans = transforms.Compose([transforms.Resize(256),
                                         transforms.TenCrop(224),
                                         transforms.Lambda(stack_crops),
                                         transforms.Normalize(mean=[0.4736, 0.4504, 0.3909],
                                                              std=[0.2655, 0.2607, 0.2650],
                                                              inplace=True)])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.trans(image), label


# In[5]:


class TrainValidDataset(data.Dataset):
    def __init__(self):
        super().__init__()
                
    def __len__(self):
        return train_csv.shape[0]
    
    def __getitem__(self, index):
        image = Image.open(path + 'train/' + train_csv['id'][index] + '.jpg')
        label = label_list.index(train_csv['breed'][index])
        return image, label


# In[6]:


train_dataset, valid_dataset = data.random_split(TrainValidDataset(),
                                                 [9200, 10222-9200])
train_dataset, valid_dataset = TrainDataset(train_dataset), ValidDataset(valid_dataset)


# In[7]:


class TestDataset(data.Dataset):
    def __init__(self, size, horizontal_flip):
        super().__init__()
        self.trans = [transforms.Resize(size),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.4736, 0.4504, 0.3909],
                                           std=[0.2655, 0.2607, 0.2650],
                                           inplace=True)]
        if horizontal_flip:
            self.trans.insert(0, transforms.RandomHorizontalFlip(p=1))
        self.trans = transforms.Compose(self.trans)
    def __len__(self):
        return test_csv.shape[0]
    
    def __getitem__(self, index):
        image = Image.open(path + 'test/' + test_csv['id'][index] + '.jpg')
        return self.trans(image)

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
            
            # output = net(input)
            bs, ncrops, c, h, w = input.size()
            output = net(input.view(-1, c, h, w))
            output = output.view(bs, ncrops, -1).mean(dim=1)
            
            loss.append(float(criterion(output, target).item()))
            metric.add(d2l.accuracy(output, target), target.numel())
    return sum(loss) / len(loss), metric[0] / metric[1]


# In[10]:


def get_lr(optimizer):
    return (optimizer.state_dict()['param_groups'][0]['lr'])


# In[11]:


def train_DenseNet(net,
                   batch_size,
                   lr,
                   num_epochs,
                   weight_decay=1e-4):
    def lr_lambda(epoch):
        if epoch < num_epochs * 0.33:
            return 1.
        elif epoch < num_epochs * 0.67:
            return 0.1
        else:
            return 0.01

    train_iter = data.DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=4)
    valid_iter = data.DataLoader(valid_dataset, batch_size=batch_size // 10, 
                                 shuffle=False, num_workers=4)
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=lr,
                                weight_decay=weight_decay,
                                momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=False)
    writer = SummaryWriter(f'runs/DenseNet_ImageNet_archi={net.archi}_k={net.k}_theta={net.theta}_dropout={net.dropout}')
    criterion = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        tic = time.time()
        metric = d2l.Accumulator(3)
        net.train()
        for i, (input, target) in enumerate(train_iter):
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


# In[12]:

if __name__ == '__main__':
    net = DenseNet_ImageNet(k=24,
                            theta=0.5,
                            block=Bottleneck,
                            archi='169',
                            num_classes=120,
                            batch_norm=True,
                            dropout=0.1).to(device)
    net.print_num_params()

    # In[13]:

    train_DenseNet(net,
                   batch_size=64,
                   lr=0.1,
                   num_epochs=240,
                   weight_decay=1e-4)
    torch.save(net.state_dict(), f'DenseNet_ImageNet_archi={net.archi}_k={net.k}_theta={net.theta}_dropout={net.dropout}.pth')