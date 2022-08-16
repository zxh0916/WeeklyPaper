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

from data import PascalVOC

def inv_sigmoid(x):
    return -torch.log(torch.pow(torch.clamp(x, 1e-6, 1.-1e-6), -1) - 1)

def refine_box(box_cxcywh, shift, downsample_rate=32):
    """由锚框坐标和网络输出计算预测框"""
    box = box_cxcywh.to(shift.device)
    # 上图中默认方形区域的边长为1，而实际上原图上方形区域的边长为该特征图的下采样率
    # 故需将坐标计算出来之后乘一个下采样率
    p_cx = downsample_rate * (torch.sigmoid(shift[:, 0]) + (box[:, 0] / downsample_rate).floor())
    p_cy = downsample_rate * (torch.sigmoid(shift[:, 1]) + (box[:, 1] / downsample_rate).floor())
    p_w = box[:, 2] * torch.exp(shift[:, 2])
    p_h = box[:, 3] * torch.exp(shift[:, 3])
    return torch.stack([p_cx, p_cy, p_w, p_h], dim=1)

def coord_to_shift(src_cxcywh, tgt_cxcywh, downsample_rate=32):
    """由锚框和预测框反算出期望的网络输出"""
    assert src_cxcywh.shape == tgt_cxcywh.shape
    t_x = inv_sigmoid(tgt_cxcywh[:, 0] / downsample_rate - (tgt_cxcywh[:, 0] / downsample_rate).floor())
    t_y = inv_sigmoid(tgt_cxcywh[:, 1] / downsample_rate - (tgt_cxcywh[:, 1] / downsample_rate).floor())
    t_w = torch.log(tgt_cxcywh[:, 2] / src_cxcywh[:, 2])
    t_h = torch.log(tgt_cxcywh[:, 3] / src_cxcywh[:, 3])
    return torch.stack([t_x, t_y, t_w, t_h], dim=1)

# 边界框格式转换
def cxcywh2xyxy(boxes_cxcywh):
    dim = boxes_cxcywh.dim()
    if dim == 1:
        boxes_cxcywh = boxes_cxcywh.unsqueeze(0)
    boxes_xyxy = torchvision.ops.box_convert(boxes_cxcywh, 'cxcywh', 'xyxy').int()
    if dim == 1:
        boxes_xyxy = boxes_xyxy.squeeze(0)
    return boxes_xyxy
def xyxy2cxcywh(boxes_xyxy):
    dim = boxes_xyxy.dim()
    if dim == 1:
        boxes_xyxy = boxes_xyxy.unsqueeze(0)
    boxes_cxcywh = torchvision.ops.box_convert(boxes_xyxy, 'xyxy', 'cxcywh').int()
    if dim == 1:
        boxes_cxcywh = boxes_cxcywh.squeeze(0)
    return boxes_cxcywh

# 锁定/解锁模型参数
def freeze(module):
    for param in module.parameters():
        param.requires_grad_(False)
def unfreeze(module):
    for param in module.parameters():
        param.requires_grad_(True)

# 逐类别非极大值抑制
def batched_nms(boxes, scores, idxs, iou_threshold):
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep_indices = torchvision.ops.nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]
            
def show_boxes(image, box1=None, box2=None, display=True, scale=2.0):
    """把框画在图片上"""
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.clone()
        image *= torch.tensor([0.2396, 0.2349, 0.2390], device=image.device).reshape(3, 1, 1)
        image += torch.tensor([0.4541, 0.4336, 0.4016], device=image.device).reshape(3, 1, 1)
        image = T.Resize(int(scale * min(image.shape[-1], image.shape[-2])))(image)
        image = T.ToPILImage()(image)
    image = np.array(image)
    if box2 is not None:
        box2 = (box2 * scale).int()
        for box in box2:
            cv2.rectangle(image,
                          (box[0].item(), box[1].item()),
                          (box[2].item(), box[3].item()),
                          (0, 255, 0), int(2*scale))
    if box1 is not None:
        box1 = (box1 * scale).int()
        for box in box1:
            cv2.rectangle(image,
                          (box[0].item(), box[1].item()),
                          (box[2].item(), box[3].item()),
                          (255, 0, 0), int(1*scale))
            cv2.circle(image,
                       ((box[0].item()+box[2].item())//2,
                        (box[1].item()+box[3].item())//2),
                       int(1*scale), (128, 128, 255), -1)
    if display:
        plt.figure(figsize=(10, 10), dpi=int(60*scale))
        plt.imshow(image)
    return image

def show_predictions(net,
                     data,
                     conf_thres,
                     iou_thres,
                     display=True,
                     scale=2.0):
    """
    给定模型和数据，应用前向传播，得到预测框，并将预测框、对应类别和置信度
    和真实边界框一同显示在图片上。
    """
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    if images.dim() == 3:
        images = images.unsqueeze(0)
    net.eval()
    n = images.shape[0]
    with torch.no_grad():
        preds = net.get_prediction(images,
                                   iou_thres,
                                   conf_thres)
    label_text = ['person',
                  'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                  'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
                  'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
    results = []
    for i in range(n):
        pred = preds[i]
        pred[:, 2::2] = torch.clamp(pred[:, 2::2], 0, images[i].shape[-1])
        pred[:, 3::2] = torch.clamp(pred[:, 3::2], 0, images[i].shape[-2])
        label = labels[labels[:, 0]==i][:, 2:]
        if pred.shape[0] != 0:
            image = show_boxes(images[i], pred[:, 2:].int(), label, display=False, scale=scale)
        else:
            image = show_boxes(images[i], None, label, display=False, scale=scale)
        for j in range(pred.shape[0]):
            category, confidence = int(pred[j, 0]), pred[j, 1].item()
            text_pos = pred[j, 2:4] * scale
            text_pos[1] -= scale * 2
            text_pos = text_pos.int().cpu().numpy()
            cv2.putText(image, f'{label_text[category]} {confidence:.2f}',
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * scale,
                        (255, 0, 0),
                        max(1, round(scale)))
        results.append(image)
    if display:
        for i in range(n):
            plt.figure(figsize=(10, 10), dpi=int(60*scale))
            plt.imshow(results[i])
    return results

def train_yolo_one_step(net, data, criterion, optimizer):
    """训练一步"""
    image, labels = data
    image, labels = image.to(device), labels.to(device)
    preds = net(image)
    loss_obj, loss_cls, loss_reg, loss = criterion(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_obj, loss_cls, loss_reg, loss.item()

def warmup_lr_ratio(warmup_steps, cur_step, power=1.):
    if cur_step == 0:
        return 0
    lr_ratio = min(cur_step ** -power,
                   (cur_step * warmup_steps ** -(1.+power))) * warmup_steps ** power
    return lr_ratio

def get_lr(optimizer):
    return (optimizer.state_dict()['param_groups'][0]['lr'])

def train_yolo(net, cfg):
    voc_train = PascalVOC(train=True, image_sizes=cfg.image_sizes, ratio=cfg.data_ratio)
    # 创建dataloader时需手动指定整合batch中数据的函数
    # 否则会因为各个样本的标签张量形状不同而报错
    dataloader = torch.utils.data.DataLoader(voc_train,
                                             batch_size=cfg.batch_size,
                                             collate_fn=voc_train.collate,
                                             shuffle=True,
                                             num_workers=cfg.num_workers)
    num_batches = len(dataloader)
    criterion = ComputeLoss(cfg.obj_pos_weight,
                            cfg.num_classes,
                            cfg.obj_gain,
                            cfg.cls_gain,
                            cfg.reg_gain,
                            cfg.neg_thres,
                            cfg.obj_pos_ratio)
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=cfg.lr,
                                weight_decay=cfg.weight_decay,
                                momentum=0.9)
    # 迭代步小于指定步数时，学习率线性增加
    # 超过指定步数后呈指数衰减，衰减速度由cfg.lr_decay_power控制
    warmup_lr = lambda cur_step: warmup_lr_ratio(int(cfg.warmup_steps*cfg.num_epochs*num_batches),
                                                 cur_step, cfg.lr_decay_power)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    writer = SummaryWriter(log_dir=f'runs/{cfg.version}')
    net.train()
    global_step = 0
    for epoch in range(1, cfg.num_epochs+1):
        epoch_loss = []
        pbar = tqdm(enumerate(dataloader), total=num_batches)
        for i, data in pbar:
            pbar.set_description(f"epoch {epoch:3d}")
            loss_obj, loss_cls, loss_reg, loss = train_yolo_one_step(net, data, criterion, optimizer)
            scheduler.step()
            global_step += 1
            pbar.set_postfix(obj=f"{loss_obj:.4f}", cls=f"{loss_cls:.4f}", reg=f"{loss_reg:.4f}", loss=f"{loss:.4f}")
            # 每10个迭代步，随机改变一次训练图片的尺寸
            if global_step % 10 == 0:
                voc_train.random_size()
                writer.add_scalars('train/loss', {'reg': loss_reg, 
                                                  'cls': loss_cls,
                                                  'obj': loss_obj,
                                                  'weighted sum': loss}, global_step=global_step)
                writer.add_scalar('train/lr', get_lr(optimizer), global_step=global_step)
            epoch_loss.append(loss)
            if global_step % (num_batches // 5) == 0:
                net.eval()
                with torch.no_grad():
                    data = (data[0][0][None, :], data[1][data[1][:, 0]==0])
                    infer_result = show_predictions(net,
                                                    data,
                                                    conf_thres=0.5,
                                                    iou_thres=0.2,
                                                    display=False,
                                                    scale=2.0)
                net.train()
                writer.add_image('train/images_with_predictions',
                                 infer_result[0],
                                 global_step=global_step,
                                 dataformats='HWC')
        print(f'epoch {epoch:4d}, loss={sum(epoch_loss) / len(epoch_loss):8.4f}')
        torch.save(net.backbone.state_dict(), f'models/{cfg.version}_backbone.pth')
        torch.save(net.neck.state_dict(), f'models/{cfg.version}_neck.pth')
        torch.save(net.head.state_dict(), f'models/{cfg.version}_head.pth')
        
class ComputeLoss:
    def __init__(self,
                 obj_pos_weight,
                 num_classes,
                 obj_gain,
                 cls_gain,
                 reg_gain,
                 neg_thres,
                 obj_pos_ratio):
        self.num_classes = num_classes
        if isinstance(obj_pos_weight, (int, float)):
            obj_pos_weight = torch.tensor(obj_pos_weight, device=device).float()
        # 物体评分和类别概率使用二分类交叉熵作为损失函数
        self.criterion_obj = nn.BCEWithLogitsLoss(pos_weight=obj_pos_weight)
        self.criterion_cls = nn.BCEWithLogitsLoss()
        # 边界框回归使用CIOU_Loss作为损失函数
        self.criterion_reg = torchvision.ops.complete_box_iou_loss
        
        self.obj_gain = obj_gain
        self.cls_gain = cls_gain
        self.reg_gain = reg_gain
        self.neg_thres = neg_thres # 某锚框与所有真实边界框的最大值小于该阈值才被归为负样本
        self.obj_pos_ratio = obj_pos_ratio # 物体评分的训练中正样本的比例
        
    def __call__(self, preds, labels):
        """计算多任务损失函数"""
        (reg_outputs, reg_targets), (obj_outputs, obj_targets), (cls_outputs, cls_targets) = \
            self.build_target(preds, labels)
        loss_obj = self.criterion_obj(obj_outputs, obj_targets)
        loss_cls = self.criterion_cls(cls_outputs, F.one_hot(cls_targets, self.num_classes).float())
        loss_reg = self.criterion_reg(reg_outputs, reg_targets, reduction='mean')
        loss = self.obj_gain * loss_obj + \
               self.cls_gain * loss_cls + \
               self.reg_gain * loss_reg
        return loss_obj.item(), loss_cls.item(), loss_reg.item(), loss # 总loss需要计算反向传播，故不取.item()
        
    def build_target(self, preds, labels):
        """根据网络输出和标签整理出用于计算损失的数据"""
        objectness, shift, class_conf, anchors_cxcywh = preds
        n = objectness[0].shape[0]
        # 记录哪些锚框是正/负样本
        pos_table = [torch.zeros_like(objectness[i], device=labels.device, dtype=bool) for i in range(3)]
        neg_table = [torch.ones_like(objectness[i], device=labels.device, dtype=bool) for i in range(3)]
        obj_outputs, obj_targets = [], []
        reg_outputs, reg_targets = [], []
        cls_outputs, cls_targets = [], []
        responsible_anchors = []
        # 负样本
        for i in range(n):
            for j in range(3):                
                gt_xyxy = labels[labels[:, 0]==i][:, 2:]
                h, w, c = anchors_cxcywh[j].shape[:3]
                anchors_xyxy = cxcywh2xyxy(anchors_cxcywh[j].reshape(-1, 4))
                # 对每个锚框分别计算其与所有真实边界框的IOU的最大值
                anchor_gt_iou = torchvision.ops.box_iou(anchors_xyxy, gt_xyxy).reshape(h, w, c, gt_xyxy.shape[0])
                max_values, _ = anchor_gt_iou.max(dim=-1)
                # 若锚框与所有真实边界框的IOU都小于给定阈值，则将其标记为负样本
                neg_table[j][i] = max_values < self.neg_thres
        # 正样本，遍历所有真实边界框
        for label in labels:
            if label.dim() == 2:
                label = label.squeeze(0)
            sample_idx, category, gt_xyxy = label[0], label[1].reshape(-1), label[None, 2:]
            gt_cxcywh = xyxy2cxcywh(gt_xyxy)
            corresponding_anchors_cxcywh = []
            for j in range(3):
                # 当前真实边界框的中心点所在的网格坐标
                gt_cx = int((gt_cxcywh[0, 0] / (8*2**j)).floor().item())
                gt_cy = int((gt_cxcywh[0, 1] / (8*2**j)).floor().item())
                # 当前真实边界框的中心点所在网格上的锚框
                corresponding_anchors_cxcywh.append(anchors_cxcywh[j][gt_cy, gt_cx])
            corresponding_anchors_cxcywh = torch.cat(corresponding_anchors_cxcywh, dim=0)
            corresponding_anchors_xyxy = cxcywh2xyxy(corresponding_anchors_cxcywh)
            
            # 找出与真实边界框IOU最高的锚框
            gt_anchor_iou = torchvision.ops.box_iou(gt_xyxy,
                                                    corresponding_anchors_xyxy).squeeze(0)
            idx = int(gt_anchor_iou.argmax())
            # 第几个检测头的第几个anchor
            head_idx, anchor_idx = idx//(anchors_cxcywh[0].shape[2]), idx%(anchors_cxcywh[0].shape[2])
            gt_cx = int((gt_cxcywh[0, 0] / (8*2**head_idx)).floor().item())
            gt_cy = int((gt_cxcywh[0, 1] / (8*2**head_idx)).floor().item())
            # 如果与当前真实边界框最大的锚框已经与其他真实边界框匹配，那么就选IOU次大的
            while pos_table[head_idx][sample_idx, gt_cy, gt_cx, anchor_idx]:
                gt_anchor_iou[idx] = -1.
                idx = int(gt_anchor_iou.argmax())
                head_idx, anchor_idx = idx//(anchors_cxcywh[0].shape[2]), idx%(anchors_cxcywh[0].shape[2])
                gt_cx = int((gt_cxcywh[0, 0] / (8*2**head_idx)).floor().item())
                gt_cy = int((gt_cxcywh[0, 1] / (8*2**head_idx)).floor().item())
                if gt_anchor_iou.max() < 0:
                    break
            # 如果一个真实边界框中心点所在网格中所有的锚框都已经与其他真实边界框对应
            # 那么就忽略这个真实边界框，不参与反向传播
            if gt_anchor_iou.max() < 0:
                continue
            responsible_anchor = corresponding_anchors_cxcywh[None, idx]
            reg_target = gt_xyxy.float()
            # 用网络的输出修正与该真实边界框匹配的锚框
            reg_output = torchvision.ops.box_convert(
                refine_box(responsible_anchor,
                           shift[head_idx][sample_idx, gt_cy, gt_cx, anchor_idx][None, :],
                           downsample_rate=8*2**head_idx),
                'cxcywh', 'xyxy')
            cls_output = class_conf[head_idx][sample_idx, gt_cy, gt_cx, anchor_idx]
            obj_output = objectness[head_idx][sample_idx, gt_cy, gt_cx, anchor_idx].reshape(-1)
            reg_outputs.append(reg_output)
            reg_targets.append(reg_target)
            obj_outputs.append(obj_output)
            obj_targets.append(torch.ones_like(obj_output))
            cls_outputs.append(cls_output)
            cls_targets.append(category)
            responsible_anchors.append(responsible_anchor)
            # 把被选中的锚框在正负样本table中标记出来
            pos_table[head_idx][sample_idx, gt_cy, gt_cx, anchor_idx] = True
            neg_table[head_idx][sample_idx, gt_cy, gt_cx, anchor_idx] = False
        
        # 用正样本比例计算出负样本数量
        num_pos_samples = sum([pos_table[i].sum() for i in range(3)])
        num_neg_samples = int(((1-self.obj_pos_ratio) / self.obj_pos_ratio) * num_pos_samples)
        obj_output = torch.cat([objectness[i][neg_table[i]] for i in range(3)])
        # 在所有负样本中随机采样
        mask = torch.rand_like(obj_output) < float(num_neg_samples/obj_output.shape[0])
        obj_output = obj_output[mask]
        obj_outputs.append(obj_output)
        obj_targets.append(torch.zeros_like(obj_output))
        # 确保所有锚框要么是正样本要么是负样本
        assert all(((pos_table[i] & neg_table[i]).sum().item() == 0 for i in range(3)))
        reg_outputs, reg_targets = torch.cat(reg_outputs, dim=0), torch.cat(reg_targets, dim=0)
        obj_outputs, obj_targets = torch.cat(obj_outputs, dim=0), torch.cat(obj_targets, dim=0)
        cls_outputs, cls_targets = torch.stack(cls_outputs, dim=0), torch.cat(cls_targets, dim=0)
        num_pos_samples = cls_targets.shape[0]
        obj_outputs = obj_outputs[:int(num_pos_samples / self.obj_pos_ratio)]
        obj_targets = obj_targets[:int(num_pos_samples / self.obj_pos_ratio)]
        # 返回物体评分、边界框预测和类别概率预测三个部分计算损失所用的数据
        return (reg_outputs, reg_targets), (obj_outputs, obj_targets),\
               (cls_outputs, cls_targets)