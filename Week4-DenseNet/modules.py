import torch
import torch.nn as nn
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class BRC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 batch_norm=True):

        super().__init__()
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm)

    def forward(self, x):
        if self.batch_norm:
            x = self.bn(x)
        return self.conv(self.relu(x))
    
    def __str__(self):
        return 'BasicBlock'
    
class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 k,
                 batch_norm=True):
        super().__init__()
        self.BRC1 = BRC(in_channels, 4*k, kernel_size=1, stride=1, padding=0, batch_norm=batch_norm)
        self.BRC2 = BRC(4*k, k, batch_norm=batch_norm)
        
    def forward(self, x):
        return self.BRC2(self.BRC1(x))
    
    def __str__(self):
        return 'Bottleneck'
    
class DenseBlock(nn.Module):
    def __init__(self, k_0, k, block, num_layer, batch_norm=True, dropout=0):
        super().__init__()
        self.num_layer = num_layer
        self.layers = []
        for i in range(self.num_layer):
            self.layers.append(block(k_0 + k*i, k, batch_norm=batch_norm))
        self.out_channels = k_0 + num_layer * k
        self.layers = nn.Sequential(*self.layers)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
    def forward(self, x):
        outputs = x
        for i in range(self.num_layer):
            outputs = torch.concat((outputs, self.layers[i](outputs)), dim=1)
        return self.dropout(outputs)

class DownSample(nn.Module):
    def __init__(self, in_channels, theta, batch_norm=True, dropout=0):
        super().__init__()
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, int(theta * in_channels), kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.out_channels = int(theta * in_channels)
    def forward(self, x):
        if self.batch_norm:
            x = self.bn(x)
        return self.avgpool(self.dropout(self.conv(x)))
    
class DenseNet_ImageNet(nn.Module):
    def __init__(self, k, theta, block, archi, num_classes, batch_norm=True, dropout=0):
        super().__init__()
        self.k = k
        self.theta = theta
        self.archi = archi
        self.architectures = {
            '121': [6, 12, 24, 16],
            '169': [6, 12, 32, 32],
            '201': [6, 12, 48, 32],
            '264': [6, 12, 64, 48],            
        }
        num_layers = self.architectures[self.archi]
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.in_channels = 2 * k
        self.conv = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocks = [[] for _ in range(len(num_layers))]
        for i, num_layer in enumerate(num_layers):
            self.blocks[i].append(DenseBlock(self.in_channels, k, block, num_layer, self.batch_norm, self.dropout))
            self.in_channels = self.blocks[i][-1].out_channels
            if i != len(num_layers) - 1:
                self.blocks[i].append(DownSample(self.in_channels, theta, self.batch_norm, dropout))
                self.in_channels = self.blocks[i][-1].out_channels
            self.blocks[i] = nn.Sequential(*self.blocks[i])
        self.blocks = nn.Sequential(*self.blocks)
        self.FC = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(self.in_channels, num_classes))
        
    def forward(self, x):
        out = self.conv(x)
        out = self.maxpool(out)
        out = self.blocks(out)
        out = self.FC(out)
        return out
    
    def print_num_params(self):
        """打印网络参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} trainable parameters.')


class DenseNet(nn.Module):
    def __init__(self, L, k, theta, block, num_classes, batch_norm=True, dropout=0):
        super().__init__()
        self.L, self.k, self.theta = L, k, theta
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.in_channels = 16
        self.block = str(block(1, 1))
        if block == Bottleneck and theta < 1.:
            self.in_channels = 2 * k
        num_block = 3
        num_layer = (L - 1 - num_block) // num_block
        if block == Bottleneck:
            num_layer = num_layer // 2
        self.conv = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1)
        self.blocks = [[] for _ in range(num_block)]
        for i in range(num_block):
            self.blocks[i].append(DenseBlock(self.in_channels, k, block, num_layer, self.batch_norm, self.dropout))
            self.in_channels = self.blocks[i][-1].out_channels
            if i != num_block - 1:
                self.blocks[i].append(DownSample(self.in_channels, theta, self.batch_norm, self.dropout))
                self.in_channels = self.blocks[i][-1].out_channels
            self.blocks[i] = nn.Sequential(*self.blocks[i])
        self.blocks = nn.Sequential(*self.blocks)
        self.FC = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(self.in_channels, num_classes))
        
    def forward(self, x):
        out = self.conv(x)
        out = self.blocks(out)
        out = self.FC(out)
        return out
    
    def print_num_params(self):
        """打印网络参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} trainable parameters.')
