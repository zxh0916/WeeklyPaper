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
import os
from time import time
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PascalVOC(torch.utils.data.Dataset):
    """PASCAL VOC 2007 + 2012 数据集"""
    def __init__(self, train=True, image_sizes=None, ratio=1.0):
        super().__init__()
        self.train = train
        # PASCAL VOC 2007
        self.data07 = torchvision.datasets.VOCDetection(root='../data',
                                                        year='2007',
                                                        image_set='train' if train else 'val',
                                                        download=False)
        # PASCAL VOC 2012
        self.data12 = torchvision.datasets.VOCDetection(root='../data',
                                                        year='2012',
                                                        image_set='train' if train else 'val',
                                                        download=False)
        # 设定要用多少比例的数据，方便使用少量数据调试代码
        if ratio != 1.:
            size07, size12 = int(len(self.data07) * ratio), int(len(self.data12) * ratio)
            self.data07, _ = torch.utils.data.random_split(self.data07, [size07, len(self.data07)-size07])
            self.data12, _ = torch.utils.data.random_split(self.data12, [size12, len(self.data12)-size12])
        # 类型转换、色彩扰动和归一化
        self.trans_train = T.Compose([T.ToTensor(),
                                      T.ColorJitter(brightness=0.2,
                                                    contrast=0.2,
                                                    saturation=0.2,
                                                    hue=0.1),
                                      T.Normalize(mean=[0.4541, 0.4336, 0.4016],
                                                   std=[0.2396, 0.2349, 0.2390],)])
        self.trans_valid = T.Compose([T.ToTensor(),
                                      T.Normalize(mean=[0.4541, 0.4336, 0.4016],
                                                   std=[0.2396, 0.2349, 0.2390],)])
        # 标签列表
        self.cls_labels = ['person',
                           'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
                           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
        # YOLOv2(http://arxiv.org/abs/1612.08242)中提到，为了获得缩放不变性，
        # 训练时每10个step，在{320, 352, ..., 608}中随机挑选一个数作为训练图片的尺寸。
        if image_sizes is not None:
            self.img_sizes = image_sizes
        else:
            self.img_sizes = [i * 32 + 320 for i in range(10)]
        self.current_shape = None
        self.random_size()
        assert self.current_shape is not None
        
    def __len__(self):
        return len(self.data07) + len(self.data12)
    
    def random_size(self):
        """从尺寸集合中随机挑选一个图片尺寸"""
        if self.train:
            self.current_shape = self.img_sizes[random.randint(0, len(self.img_sizes) - 1)]
        else:
            self.current_shape = 416
        return self.current_shape
    
    def Resize(self, image, box_coords, size):
        """调整图片和其对应的真实边界框的尺寸"""
        if isinstance(size, (int, float)):
            size = (int(size), int(size))
        h, w = image.size[1], image.size[0]
        resize_ratio = (size[0] / w, size[1] / h)
        image = T.Resize(size)(image)
        box_coords[:, 0::2] = (box_coords[:, 0::2] * resize_ratio[0]).int()
        box_coords[:, 1::2] = (box_coords[:, 1::2] * resize_ratio[1]).int()
        return image, box_coords
    
    def __getitem__(self, index):
        # 判断是使用07年的数据还是12年的数据
        data = self.data07 if index < len(self.data07) else self.data12
        index = index if index < len(self.data07) else index - len(self.data07)
        image = data[index][0]
        box_labels, box_coords = self.get_label_list(data[index][1])
        if self.train:
            image, box_coords = self.Resize(image, box_coords, self.current_shape)
            image, box_coords = self.RandomHorizontalFlip(image, box_coords)
            image = self.trans_train(image)
        else:
            image, box_coords = self.Resize(image, box_coords, 416)
            image = self.trans_valid(image)
        return image, torch.cat((torch.zeros_like(box_labels, dtype=int),
                                 box_labels, box_coords), dim=1)
    
    def get_label_list(self, label):
        """获取图片中物体的类别和真实边界框的xyxy坐标"""
        obj_list = label['annotation']['object']
        box_labels = [self.cls_labels.index(obj['name'] if type(obj['name']) == str else obj['name'][0]) for obj in obj_list]
        box_coords = []
        for obj in obj_list:
            coord = []
            for k in ['xmin', 'ymin', 'xmax', 'ymax']:
                v = obj['bndbox'][k]
                coord.append(int(v if type(v) == str else v[0]))
            box_coords.append(coord)
        return (torch.tensor(box_labels)[:, None], torch.tensor(box_coords))

    def RandomHorizontalFlip(self, image, box_coords):
        """随机水平翻转"""
        if random.random() > 0.5:
            w = image.size[0]
            image = T.RandomHorizontalFlip(p=1)(image)
            x1, x2 = box_coords[:, 0], box_coords[:, 2]
            box_coords[:, 0], box_coords[:, 2] = w - x2, w - x1
        return image, box_coords
    
    def collate(self, batch):
        """将一个批量的数据整合成两个张量"""
        image, labels = zip(*batch)
        image = torch.stack(image, 0)
        for i, label in enumerate(labels):
            label[:, 0] = i
        # 第一个返回值是图片，形状为 [batch_size, C, H, W]
        # 第二个返回值是标签，形状为 [batch_size, 6]
        # 其中每行的第一个数为这行标签对应的图片样本下标，
        # 第二个数为这行标签所对应的物体的类别编号，
        # 后四个数为真实边界框的xyxy坐标。
        return image, torch.cat(labels, 0)
    
class HuaweiDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, size=640):
        super().__init__()
        self.train = train
        self.class_labels = ['red_stop', 'yellow_back', 'green_go',
                             'pedestrian_crossing', 'speed_limited', 'speed_unlimited']
        self.trans_train = T.Compose([T.ToTensor(),
                                      T.ColorJitter(brightness=0.2,
                                                    contrast=0.2,
                                                    saturation=0.2,
                                                    hue=0.1),
                                      T.Normalize(mean=[0.3898, 0.4011, 0.3812],
                                                   std=[0.2562, 0.2588, 0.2561],)])
        self.trans_valid = T.Compose([T.ToTensor(),
                                      T.Normalize(mean=[0.3898, 0.4011, 0.3812],
                                                   std=[0.2562, 0.2588, 0.2561],)])
        self.size = size
        self.path_list = []
        subset = 'train' if train else 'val'
        self.image_path = f'./hw_data/images/{subset}/'
        self.label_path = f'./hw_data/labels/{subset}/'
        for file_path in os.listdir(self.image_path):
            if file_path[-3:] == 'jpg':
                self.path_list.append(file_path)
        
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        img = cv2.imread(self.image_path + self.path_list[index])[:, :, ::-1] # BGR->RGB
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        label, box = self.get_gt(self.path_list[index], h, w)
        label = label[:, None]
        if self.train:
            img = self.trans_train(img)
            img, box = self.RandomHorizontalFlip(img, box)
        else:
            img = self.trans_valid(img)
        img, box = self.Resize(img, box, self.size)
        return img, torch.cat((torch.zeros_like(label, dtype=int), label, box), dim=1)
        
    def get_gt(self, img_path, h, w):
        label_path = self.label_path + img_path.split('.')[0] + '.txt'
        with open(label_path, 'r') as label_file:
            labels = label_file.readlines()
        labels = [line[:-1].split(' ') for line in labels]
        labels = [([int(label[0])] + list(map(float, label[1:]))) for label in labels]
        for label in labels:
            label[1] = int(label[1] * w)
            label[2] = int(label[2] * h)
            label[3] = int(label[3] * w)
            label[4] = int(label[4] * h)
        labels = [(label[0], label[1:]) for label in labels]
        labels = list(zip(*labels))
        label, box = torch.tensor(labels[0]), cxcywh2xyxy(torch.tensor(labels[1]))
        return label, box
        
    def RandomHorizontalFlip(self, image, box_coords):
        if random.random() > 0.5:
            w = image.shape[-1]
            image = T.RandomHorizontalFlip(p=1)(image)
            x1, x2 = box_coords[:, 0], box_coords[:, 2]
            box_coords[:, 0], box_coords[:, 2] = w - x2, w - x1
        return image, box_coords
    
    def Resize(self, image, box_coords, size):
        if isinstance(size, (int, float)):
            size = (int(size), int(size))
        h, w = image.shape[-2:]
        resize_ratio = (size[0] / w, size[1] / h)
        image = T.Resize(size)(image)
        box_coords[:, 0::2] = (box_coords[:, 0::2] * resize_ratio[0]).int()
        box_coords[:, 1::2] = (box_coords[:, 1::2] * resize_ratio[1]).int()
        return image, box_coords
    
    def collate(self, batch):
        image, labels = zip(*batch)
        image = torch.stack(image, 0)
        for i, label in enumerate(labels):
            label[:, 0] = i
        return image, torch.cat(labels, 0)