import torch
import torch.nn as nn
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class CBR(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 batch_norm=True):
        
        super().__init__()
        self.batch_norm = batch_norm
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              bias=not self.batch_norm)
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, X):
        output = self.conv(X)
        if self.batch_norm:
            output = self.bn(output)
        output = self.relu(output)
        return output


def get_downsample(in_channels, out_channels, stride, batch_norm, option='B'):
    if option in ['A', 'B'] and in_channels == out_channels:
        downsample = None
    elif option == 'A':
        downsample = lambda x: torch.concat((x[:, :, ::stride, ::stride],
                                             torch.zeros(x.shape[0],
                                                         out_channels - x.shape[1],
                                                         x.shape[2] // stride,
                                                         x.shape[3] // stride,
                                                         device=device)), dim=1)
    elif option in ['B', 'C']:
        downsample = [nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=stride,
                                padding=0,
                                bias=not batch_norm)]
        if batch_norm:
            downsample.append(nn.BatchNorm2d(out_channels))
        downsample = nn.Sequential(*downsample)
    return downsample


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, batch_norm, option='B', plain=False):
        super().__init__()
        self.plain = plain
        self.conv1 = CBR(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, batch_norm=batch_norm)
        self.conv2 = [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)]
        if batch_norm:
            self.conv2.append(nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Sequential(*self.conv2)
        self.relu = nn.ReLU(inplace=True)
        if not self.plain:
            self.downsample = get_downsample(in_channels, out_channels, stride, batch_norm, option=option)
        
    def forward(self, X):
        identity = X
        output = self.conv1(X)
        output = self.conv2(output)
        if not self.plain:
            if self.downsample is not None:
                identity = self.downsample(identity)
            output += identity
        output = self.relu(output)
        return output       
       

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    def __init__(self, in_channels, out_channels, stride, batch_norm, option='B', plain=False):
        super().__init__()
        self.plain = plain
        bottleneck = out_channels // 4
        self.conv1 = CBR(in_channels, bottleneck, kernel_size=1, stride=1, padding=0, batch_norm=batch_norm)
        self.conv2 = CBR(bottleneck, bottleneck, kernel_size=3, stride=stride, padding=1, batch_norm=batch_norm)
        self.conv3 = [nn.Conv2d(bottleneck, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
        if batch_norm:
            self.conv3.append(nn.BatchNorm2d(out_channels))
        self.conv3 = nn.Sequential(*self.conv3)
        self.relu = nn.ReLU(inplace=True)
        if not self.plain:
            self.downsample = get_downsample(in_channels, out_channels, stride, batch_norm, option=option)

    def forward(self, X):
        identity = X
        output = self.conv1(X)
        output = self.conv2(output)
        output = self.conv3(output)
        if not self.plain:
            if self.downsample is not None:
                identity = self.downsample(identity)
            output += identity
        output = self.relu(output)
        return output
    
    
class ResNet_ImageNet(nn.Module):
    def __init__(self, architecture, num_classes, option='B', batch_norm=True, dropout=0, plain=False):
        super().__init__()
        self.in_channels = 64
        self.architecture = architecture
        self.option = option
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.plain = plain
        self.architectures = {
            '18':  [BasicBlock, [2, 2,  2, 2], [ 64, 128,  256,  512]],
            '34':  [BasicBlock, [3, 4,  6, 3], [ 64, 128,  256,  512]],
            '50':  [Bottleneck, [3, 4,  6, 3], [256, 512, 1024, 2048]],
            '101': [Bottleneck, [3, 4, 23, 3], [256, 512, 1024, 2048]],
            '152': [Bottleneck, [3, 8, 36, 3], [256, 512, 1024, 2048]]
        }
        self.conv1 = CBR(3, self.in_channels, kernel_size=7, stride=2, padding=3, batch_norm=batch_norm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        archi = self.architectures[self.architecture]
        self.layers = []
        self.layers.append(self._make_layer(archi[0],
                                            archi[1][0],
                                            self.in_channels,
                                            archi[2][0],
                                            1,
                                            option,
                                            batch_norm,
                                            plain))
        self.in_channels = archi[2][0]
        for i in range(1, len(archi[1])):
            self.layers.append(self._make_layer(archi[0],
                                                archi[1][i],
                                                self.in_channels,
                                                archi[2][i],
                                                2,
                                                option,
                                                batch_norm,
                                                plain))
            self.in_channels = archi[2][i]
        self.layers = nn.Sequential(*self.layers)
        self.dp = nn.Dropout2d(p=dropout)
        self.FC = nn.Sequential(nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
                                nn.Conv2d(self.in_channels, num_classes, kernel_size=1, stride=1, padding=0),
                                nn.Flatten(2, 3))
    
    def print_num_params(self):
        """打印网络参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} trainable parameters.')

    def _make_layer(self,
                    block,
                    num_blocks,
                    in_channels,
                    out_channels,
                    stride,
                    option,
                    batch_norm,
                    plain):
        blocks = []
        blocks.append(block(in_channels, out_channels, stride, batch_norm, option, plain))
        for _ in range(1, num_blocks):
            blocks.append(block(out_channels, out_channels, 1, batch_norm, option, plain))
        return nn.Sequential(*blocks)
    
    def forward(self, X):
        output = self.conv1(X)
        output = self.maxpool1(output)
        output = self.layers(output)
        output = self.dp(output)
        output = self.FC(output)
        output = output.mean(dim=2)
        return output
    

class ResNet_CIFAR(nn.Module):
    def __init__(self, n, option='A', batch_norm=True, dropout=0, plain=False):
        super().__init__()
        self.n = n
        self.option = option
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.plain = plain
        self.conv1 = CBR(3, 16, kernel_size=3, stride=1, padding=1, batch_norm=batch_norm)
        
        self.stage1 = nn.Sequential(*[BasicBlock(16, 16, 1, batch_norm, option, plain) for _ in range(n)])
        
        self.stage2 = [BasicBlock(16, 32, 2, batch_norm, option, plain)]
        for _ in range(n-1):
            self.stage2.append(BasicBlock(32, 32, 1, batch_norm, option, plain))
        self.stage2 = nn.Sequential(*self.stage2)
        
        self.stage3 = [BasicBlock(32, 64, 2, batch_norm, option, plain)]
        for _ in range(n-1):
            self.stage3.append(BasicBlock(64, 64, 1, batch_norm, option, plain))
        self.stage3 = nn.Sequential(*self.stage3)
        self.dp = nn.Dropout2d(p=dropout)
        self.FC = nn.Sequential(nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                                nn.Flatten(1, 3),
                                nn.Linear(64, 10))
    
    def print_num_params(self):
        """打印网络参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} trainable parameters.')
        
    def forward(self, X):
        output = self.conv1(X)
        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.dp(output)
        output = self.FC(output)
        return output
    
    
    
if __name__ == '__main__':
    test_net1 = ResNet_ImageNet(architecture='50',
                                num_classes=1000,
                                option='B',
                                batch_norm=True).to(device)
    test_net2 = ResNet_CIFAR(n=3, option='A', batch_norm=True).to(device)
    a = torch.zeros(3, 3, 224, 224, device=device)
    print(test_net1(a).shape)
    b = torch.zeros(4, 3, 32, 32, device=device)
    print(test_net2(b).shape)