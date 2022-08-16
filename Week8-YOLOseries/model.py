import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms as T
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL
import random
from time import time
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import *

class Backbone(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        # YOLO的backbone需要输出三张特征图
        # 三张特征图的下采样率分别为8、16和32
        # module_dict中每个值都是一个包含四个元素的元组
        # 其中第一个元素是使用torchvision的API取backbone的函数
        # 后三个元素分别是输出为上述三种特征图的网络的三个部分的模块列表
        module_dict = {
            'resnet18': (models.resnet18,
                         ['conv1', 'bn1', 'relu', 'maxpool',
                          'layer1', 'layer2'],
                         ['layer3'], ['layer4']),
            'resnet34': (models.resnet34,
                         ['conv1', 'bn1', 'relu', 'maxpool',
                          'layer1', 'layer2'],
                         ['layer3'], ['layer4']),
            'resnet50': (models.resnet50,
                         ['conv1', 'bn1', 'relu', 'maxpool',
                          'layer1', 'layer2'],
                         ['layer3'], ['layer4']),
            'resnet101': (models.resnet101,
                          ['conv1', 'bn1', 'relu', 'maxpool',
                           'layer1', 'layer2'],
                          ['layer3'], ['layer4']),
            'resnet152': (models.resnet152,
                          ['conv1', 'bn1', 'relu', 'maxpool',
                           'layer1', 'layer2'],
                          ['layer3'], ['layer4']),
            'densenet121': (models.densenet121,
                            ['conv0', 'norm0', 'relu0', 'pool0',
                             'denseblock1', 'transition1', 'denseblock2'],
                            ['transition2', 'denseblock3'],
                            ['transition3', 'denseblock4', 'norm5']),
            'densenet161': (models.densenet161,
                            ['conv0', 'norm0', 'relu0', 'pool0',
                             'denseblock1', 'transition1', 'denseblock2'],
                            ['transition2', 'denseblock3'],
                            ['transition3', 'denseblock4', 'norm5']),
            'densenet169': (models.densenet169,
                            ['conv0', 'norm0', 'relu0', 'pool0',
                             'denseblock1', 'transition1', 'denseblock2'],
                            ['transition2', 'denseblock3'],
                            ['transition3', 'denseblock4', 'norm5']),
            'densenet201': (models.densenet201,
                            ['conv0', 'norm0', 'relu0', 'pool0',
                             'denseblock1', 'transition1', 'denseblock2'],
                            ['transition2', 'denseblock3'],
                            ['transition3', 'denseblock4', 'norm5']),
            'mobilenet_v3_small': (models.mobilenet_v3_small,
                                   ['0', '1', '2', '3'],
                                   ['4', '5', '6', '7', '8'],
                                   ['9', '10', '11', '12']),
            'mobilenet_v3_large': (models.mobilenet_v3_large,
                                   ['0', '1', '2', '3', '4', '5', '6'],
                                   ['7', '8', '9', '10', '11', '12'],
                                   ['13', '14', '15', '16'])
        }
        assert backbone_name in list(module_dict.keys())
        raw_backbone = module_dict[backbone_name][0](pretrained=True)._modules
        if backbone_name[:6] != 'resnet':
            raw_backbone = raw_backbone['features']._modules
        self.backbone_ds8  = nn.Sequential(*[raw_backbone[key] for key in module_dict[backbone_name][1]])
        self.backbone_ds16 = nn.Sequential(*[raw_backbone[key] for key in module_dict[backbone_name][2]])
        self.backbone_ds32 = nn.Sequential(*[raw_backbone[key] for key in module_dict[backbone_name][3]])
        
    def forward(self, input):
        """用网络的三个部分依次计算下采样率为8、16和32的特诊图"""
        fmap_s8 = self.backbone_ds8(input)
        fmap_s16 = self.backbone_ds16(fmap_s8)
        fmap_s32 = self.backbone_ds32(fmap_s16)
        return fmap_s8, fmap_s16, fmap_s32
    
class Focus(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        assert input.shape[-2] % 2 == 0 and input.shape[-1] % 2 == 0
        reshaped_fmap = torch.cat([input[:, :, i::2, j::2] for i in (0, 1) for j in (0, 1)], dim=1)
        return reshaped_fmap

class SPP(nn.Module):
    """空间金字塔池化层"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = F.max_pool2d(x, kernel_size=5,  stride=1, padding=2)
        y2 = F.max_pool2d(x, kernel_size=9,  stride=1, padding=4)
        y3 = F.max_pool2d(x, kernel_size=13, stride=1, padding=6)
        return torch.cat([x, y1, y2, y3], dim=1)
    
class CBL(nn.Sequential):
    """网络基本组成模块"""
    def __init__(self, in_channels, out_channels=None, k=3, s=1, p=1):
        if out_channels is None:
            out_channels = in_channels
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())
        
class FPNBlock(nn.Module):
    """
    将语义信息较为丰富的小尺寸特征图
    和空间信息较为丰富的大尺寸特征图融合的网络结构，
    有利于提升小尺寸物体的检测质量。
    """
    def __init__(self, small_in_channels, big_in_channels, hidden_layers=5, out_channels=256):
        super().__init__()
        # 对小尺寸特征图进行上采样
        self.small_branch = nn.Sequential(
            CBL(small_in_channels, out_channels),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())
        # 对大尺寸特征图的通道数进行变换
        self.big_branch = CBL(big_in_channels, out_channels, k=1, s=1, p=0)
        # 融合拼接后的特征图
        self.merge = [CBL(2 * out_channels, out_channels)]
        for i in range(hidden_layers - 1):
            self.merge.append(CBL(out_channels))
        self.merge = nn.Sequential(*self.merge)
    def forward(self, small, big):
        return self.merge(torch.cat([self.small_branch(small), self.big_branch(big)], dim=1))
    
class PANBlock(nn.Module):
    """
    将浅层较为丰富的几何信息再次传递给深层，
    进一步增强了网络输出的预测框的精确度。
    """
    def __init__(self, small_in_channels, big_in_channels, hidden_layers=5, out_channels=256):
        super().__init__()
        # 对大尺寸特征图进行下采样
        self.big_branch = CBL(big_in_channels, out_channels, s=2)
        # 对小尺寸特征图的通道数进行变换
        self.small_branch = CBL(small_in_channels, out_channels, k=1, s=1, p=0)
        # 融合拼接后的特征图
        self.merge = [CBL(2 * out_channels, out_channels)]
        for i in range(hidden_layers - 1):
            self.merge.append(CBL(out_channels))
        self.merge = nn.Sequential(*self.merge)
    def forward(self, small, big):
        return self.merge(torch.cat([self.small_branch(small), self.big_branch(big)], dim=1))
    
class Neck(nn.Module):
    """
    YOLOv4网络结构的Neck部分，
    将backbone输出的特征图使用PAN结构进行融合后
    送至Head进行预测
    """
    def __init__(self, ds8_outchannels, ds16_outchannels, ds32_outchannels, hidden_layers=5, out_channels=256):
        super().__init__()
        self.trans_3_4 = nn.Sequential(
            CBL(ds32_outchannels, out_channels), CBL(out_channels), CBL(out_channels),
            SPP(),
            CBL(4 * out_channels, out_channels), CBL(out_channels), CBL(out_channels))
        self.trans_42_5 = FPNBlock(out_channels, ds16_outchannels, hidden_layers, out_channels)
        self.trans_51_6 = FPNBlock(out_channels, ds8_outchannels, hidden_layers, out_channels)
        self.trans_56_7 = PANBlock(out_channels, out_channels, hidden_layers, out_channels)
        self.trans_47_8 = PANBlock(out_channels, out_channels, hidden_layers, out_channels)
    
    def forward(self, input):
        fmap_1, fmap_2, fmap_3 = input
        fmap_4 = self.trans_3_4(fmap_3)
        fmap_5 = self.trans_42_5(fmap_4, fmap_2)
        fmap_6 = self.trans_51_6(fmap_5, fmap_1)
        fmap_7 = self.trans_56_7(fmap_5, fmap_6)
        fmap_8 = self.trans_47_8(fmap_4, fmap_7)
        return fmap_6, fmap_7, fmap_8

class Head(nn.Module):
    """YOLO网络结构中的检测头，三个检测头参数各不相同"""
    def __init__(self, in_channels, num_classes, num_anchors, hidden_layers):
        super().__init__()
        out_channels = num_anchors * (5 + num_classes)
        self.head_big, self.head_mid, self.head_sml = [], [], []
        for i in range(hidden_layers):
            self.head_big.append(CBL(in_channels))
            self.head_mid.append(CBL(in_channels))
            self.head_sml.append(CBL(in_channels))
        self.head_big.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
        self.head_mid.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
        self.head_sml.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
        self.head_big = nn.Sequential(*self.head_big)
        self.head_mid = nn.Sequential(*self.head_mid)
        self.head_sml = nn.Sequential(*self.head_sml)
    def forward(self, input):
        fmap_big, fmap_mid, fmap_sml = input
        return self.head_big(fmap_big), self.head_mid(fmap_mid), self.head_sml(fmap_sml)
    
class Yolo(nn.Module):
    """一个简单的YOLO目标检测模型"""
    def __init__(self, backbone, anchors, num_classes, hidden_channels, neck_hidden_layers, head_hidden_layers):
        super().__init__()
        self.num_classes = num_classes # 类别数
        # 将所有锚框三等分，分别分配给三个检测头
        self.num_anchors = len(anchors)
        self.anchor_wh = [anchors[0:len(anchors)//3], anchors[len(anchors)//3:-len(anchors)//3], anchors[-len(anchors)//3:]]
        self.backbone = Backbone(backbone)
        fmap_s8, fmap_s16, fmap_s32 = self.backbone(torch.zeros(1, 3, 64, 64))
        self.neck = Neck(fmap_s8.shape[1], fmap_s16.shape[1], fmap_s32.shape[1], neck_hidden_layers, hidden_channels)
        self.head = Head(hidden_channels, num_classes, len(anchors) // 3, head_hidden_layers)
        init_weight(self.neck)
        init_weight(self.head)
        
    def forward(self, input):
        """前向传播"""
        out_big, out_mid, out_sml = self.head(self.neck(self.backbone(input))) # 获取网络输出
        objectness, shift, class_conf, anchors_cxcywh = [], [], [], []
        # 从每个检测头的输出中分别提取物体评分输出、边界框预测输出和类别概率预测输出
        for i, out in enumerate((out_big, out_mid, out_sml)):
            out = out.permute(0, 2, 3, 1)
            n, h, w, c = out.shape
            out = out.reshape(n, h, w, self.num_anchors//3, self.num_classes + 5)
            objectness.append(out[:, :, :, :, 0])
            shift.append(out[:, :, :, :, 1:5])
            class_conf.append(out[:, :, :, :, -self.num_classes:])
            # 三个检测头所对应原图上方形区域的边长分别为8、16和32
            anchors_cxcywh.append(self.generate_anchor((h, w), self.anchor_wh[i], downsample_rate=8*2**i))
        return objectness, shift, class_conf, anchors_cxcywh
        
    def generate_anchor(self, fmap_size, anchor_wh, downsample_rate):
        num_anchors = len(anchor_wh)
        img_h, img_w = fmap_size[-1] * downsample_rate, fmap_size[-2] * downsample_rate
        # 此处输入的anchor_wh均为锚框的高宽相对于原图高宽的比例，故需与原图高宽相乘
        # 锚框的高宽与锚框中心点的位置无关
        anchor_wh = torch.tensor([(round(w*img_w), round(h*img_h)) for (w, h) in anchor_wh],
                                 device=device).reshape(1, 1, num_anchors, 2)
        # 锚框中心点以对应特征图的下采样率为步长均匀分布在整张图片上
        cx = torch.arange(0, fmap_size[-1], 1, device=device).reshape(1, fmap_size[-1], 1, 1) \
           * downsample_rate + downsample_rate // 2
        cy = torch.arange(0, fmap_size[-2], 1, device=device).reshape(fmap_size[-2], 1, 1, 1) \
           * downsample_rate + downsample_rate // 2
        # 将锚框的高宽和中心点坐标拼接起来，形成cxcywh格式
        anchor_cxcywh = torch.cat([cx.expand(fmap_size[-2], -1, num_anchors, -1),
                                   cy.expand(-1, fmap_size[-1], num_anchors, -1),
                                   anchor_wh.expand(fmap_size[-2], fmap_size[-1], -1, -1)], dim=-1)
        return anchor_cxcywh
    
    def get_prediction(self, input, iou_thres=0.4, conf_thres=0.5):
        """端到端地获取网络的预测输出"""
        if input.dim() == 3:
            input = input.unsqueeze(0)
        preds = []
        # 前向传播
        with torch.no_grad():
            objectness, shift, class_conf, anchors_cxcywh = self.forward(input)
        n = input.shape[0]
        # 记录网络各个检测头输出的预测框数量并累加
        num_preds = [0] + [objectness[i].shape[1] * objectness[i].shape[2] * objectness[i].shape[3] for i in range(3)]
        num_preds_accu = [sum(num_preds[:i+1]) for i in range(len(num_preds))]
        # 将三个检测头的输出拼接起来
        objectness = torch.cat([obj.reshape(n, -1).unsqueeze(-1) for obj in objectness], dim=1)
        shift = torch.cat([sft.reshape(n, -1, 4) for sft in shift], dim=1)
        class_conf = torch.cat([cls_conf.reshape(n, -1, self.num_classes) for cls_conf in class_conf], dim=1)
        class_conf = torch.sigmoid(class_conf) * torch.sigmoid(objectness) # 置信度等于物体评分与类别概率最大值之乘积
        anchors_cxcywh = torch.cat([anchor.reshape(-1, 4) for anchor in anchors_cxcywh], dim=0)
        max_conf, max_idx = class_conf.max(dim=-1)
        for i in range(n): # 遍历小批量中所有样本
            mask = max_conf[i] >= conf_thres # 筛选置信度大于阈值的预测结果
            pred_xyxy = []
            for j in range(3): # 遍历3个检测头的预测结果
                idx = torch.arange(0, sum(num_preds), 1, device=device)
                head_mask = (idx >= num_preds_accu[j]) & (idx < num_preds_accu[j+1]) & mask
                # 用网络的边界框预测输出对锚框进行修正并转换为xyxy格式
                pred_xyxy.append(cxcywh2xyxy(refine_box(anchors_cxcywh[head_mask], shift[i, head_mask], 8*2**j)))
            pred_xyxy = torch.cat(pred_xyxy, dim=0) # 拼接三个检测头的输出
            # 逐类别非极大值抑制
            remains = batched_nms(pred_xyxy.float(), max_conf[i, mask], max_idx[i, mask], iou_thres)
            pred_xyxy = pred_xyxy[remains]
            remains = torch.where(mask)[0][remains]
            pred_conf, pred_idx = max_conf[i, remains], max_idx[i, remains]
            # 每条预测结果为一个6维向量：物体类别、置信度和xyxy坐标
            pred = torch.cat([pred_idx[:, None], pred_conf[:, None], pred_xyxy], dim=-1)
            preds.append(pred)
        return preds
    
def init_weight(module):
    """递归初始化模型参数"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Sequential, nn.ModuleList)):
        for m in module:
            init_weight(m)