import torch
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
torch.manual_seed(seed)
random.seed(seed)

from utils import train_yolo
from model import Yolo

class Configuration:
    def __init__(self):
        self.version = 'version 10'
        self.backbone = 'resnet50'
        self.num_classes = 20
        self.neck_hidden_layers = 1
        self.head_hidden_layers = 4
        self.hidden_channels = 128
        self.neg_thres = 0.3
        
        self.data_ratio = 1.0
        self.anchors = [(0.07, 0.14), (0.1, 0.1), (0.14, 0.07),
                        (0.274, 0.548), (0.387, 0.387), (0.548, 0.274),
                        (0.5, 1.0), (0.8, 0.8), (1.0, 0.5)]
        self.image_sizes = [i * 32 + 320 for i in range(10)]
        self.obj_pos_weight = 10.
        self.obj_pos_ratio = 0.05
        self.obj_gain = 1.
        self.cls_gain = 3.
        self.reg_gain = 1.
        
        self.lr = 5e-3
        self.warmup_steps = 0.1
        self.lr_decay_power = 0.5
        self.batch_size = 8
        self.num_epochs = 70
        self.weight_decay = 0
        self.num_workers = 8
        
if __name__ == '__main__':
    cfg = Configuration()
    yolo = Yolo(cfg.backbone,
                cfg.anchors,
                cfg.num_classes,
                cfg.hidden_channels,
                cfg.neck_hidden_layers,
                cfg.head_hidden_layers).to(device)
    train_yolo(yolo, cfg)