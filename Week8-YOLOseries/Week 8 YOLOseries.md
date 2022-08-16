# Week 8 YOLO系列

## 前言

​		目标检测是深度学习应用于计算机视觉领域最成功的方向之一。目标检测算法可以分为双阶段检测算法和单阶段检测算法。双阶段检测算法先获取各类别通用的候选框以覆盖图像中可能存在的物体，再对候选框进行修正从而得到预测框并进行类别预测；而单阶段检测算法**直接输出预测框与各个预测框对应的类别。**

​		如果读者已经学习过双阶段检测算法，那么大致可以从如下两个角度来理解双阶段算法和单阶段算法的区别：

-   单阶段算法相当于把类别预测提前到了双阶段算法的第一个阶段来进行；
-   单阶段算法相当于将双阶段算法中第一个阶段中对锚框（若有）进行修正的过程删去。

​		在介绍YOLO系列算法之前，我们先介绍一下目标检测算法的组成部分：

1.   Backbone，通常为一个用于图像分类的CNN的一部分，负责从图片中提取特征；
2.   Neck，用于对Backbone提取出来的特征图进行处理，输出处理过后的特征图；
3.   Head，使用Neck输出的特征图进行边界框的回归和类别的预测。

​		YOLO系列算法作为知名度最高，应用最广的单阶段检测算法，其仅用一个阶段完成目标检测任务的思路绝对是值得我们学习的。下面，我们先总结YOLO的核心思路，同时也是其快速且有效的原因：***YOLO按Backbone输出的特征图尺寸将图片分为许多个等大的正方形区域，特征图上各个位置只负责预测中心点落在其对应方形区域之内的物体。***

![YOLO系列的核心思想](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolo%E6%A0%B8%E5%BF%83%E6%80%9D%E6%83%B3.jpg)

​		YOLO系列的作者在YOLOv1开始使用这个思路。笔者是这么理解这个做法的：特征图上各个值的感受野的中心应当正好落在其在原图上对应的方形区域内。在特征图的感受野大小尚未覆盖整张图片时，**仅让物体边界框中心点所在位置对应的特征图的像素负责预测该物体**，可以在最大程度上保证较大的物体也能被包含在感受野之内。若特征图上其他位置也参与预测该物体，则在感受野不够大或物体较大时，则有更高的概率物体会超出感受野范围，后果就是模型没有办法精准的预测物体的边界框甚至类别。

​		读者看到这里一定心生疑问：谈了半天**到底什么是“负责”预测某物体？**别急，我们将在具体结构的讲解中说明不同版本的YOLO算法是如何**将特征图上的值和原图中的某个物体关联起来**的。

## YOLO v1——类别无关的预测框

### 一、网络结构

​		YOLOv1的网络结构如下图所示。

![YOLOv1 网络架构](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov1-archi.jpg)

![YOLOv1 推理流程](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov1-infer.jpg)

### 二、模型输出

​		YOLOv1以一张图片作为输入，经过输个卷积、池化、激活层之后，将图片变换为一个$S\times S\times 1024$的特征图，随后经过两个全连接层，输出一个$S\times S\times (B\times5+C)$维的向量，再将这个向量重排列成$S\times S\times (B\times5+C)$的张量，其中$S$为输出特征图的长宽，$B$为特征图上每个位置上输出的预测框数量，$C$为数据集的物体类别总数。YOLOv1中，作者选定$S=7$，$B=2$，$C=20$。

​		YOLOv1之所以能将传统目标检测的两个阶段合二为一，与其解读网络输出的方式密切相关。我们提取出这张特征图某个位置上全部通道的30个数值，看看在YOLOv1的定义下，这30个数值如何完成图片中物体的分类和边界框的回归。

![YOLOv1对网络输出的解释](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov1-output.jpg)

​		我们先来说明YOLOv1解码网络边界框回归输出的方式，如下图：

![预测框解码方式](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov1-box.jpg)

​		其中$t_*$为网络的输出，$p_*$为解码后的预测框的cxcywh坐标。至于各个预测框的物体评分(obj)，我们认为其输出的是该预测框与其想要预测的物体的真实边界框的IOU。

### 三、模型训练

​		正如上文介绍的，若前向传播后模型输出了高宽为$S$的特征图，则将原图分为$S^2$个正方形区域，每个正方形区域预测$B$个预测框和$C$个类别概率。

​		给定一张训练图片和其对应的数个真实边界框，对于每个真实边界框，假设其落在了序号为$(i, j)$的方形区域内。接下来，分别计算真实边界框和$(i, j)$处$B$个预测框的IOU，**将这个真实边界框，和与之IOU最大的预测框（序号为$k$）匹配起来**。这样的匹配策略会使$B$个预测框越来越特化，在训练的过程中分工逐渐趋于明确，初始微偏窄长的预测框会越来越窄长，初始微偏矮胖的预测框会越来越偏矮胖。

​		这样，YOLOv1的优化目标为：

1.   $(i, j)$处输出的分类输出在该真实边界框对应类别上的类别概率应当尽可能高；
2.   该真实边界框和位置$(i, j)$上的第$k$个预测框应当尽可能重合；
3.   位置$(i, j)$上的第$k$个预测框的**物体评分（$obj_k$）应当与该真实边界框和预测框的IOU尽可能接近。**

​		由此，我们可以写出YOLOv1的损失函数：

![YOLOv1的损失函数](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov1-loss.jpg)

​		其中，$\mathbb {1}_{i}^{obj}$代表下标为$i$的方形区域内是否包含了某物体的真实边界框的中心点，$\mathbb {1}_{ij}^{obj}$代表下标为$i$的方形区域内的第$j$个预测框是否与某真实边界框相匹配。由上式可以看出，对于物体评分，无论某个预测框是否与一个真实边界框相匹配，其对应的物体评分都参与物体评分损失的计算。但对于预测框坐标来说，仅**与真实边界框相匹配的预测框**的四个坐标才参与回归损失的计算。相似的，仅**包含了某真实边界框中心点的方形区域**的分类输出才参与类别损失的计算。

​		由上文的匹配策略可知，真实边界框和预测框是一一对应的，因此在物体评分obj的优化中，绝大部分样本都为负样本，为了避免负样本主导了物体评分损失的梯度，导致网络将所有预测框的物体评分都压低至0，作者在物体评分负样本项前添加了一小于1的增益$\lambda_{noobj}=0.5$来平衡正负样本的影响。

​		值得一提的是，相比于用物体评分直接预测该预测框是否负责预测某物体（第三行$\hat{C_i}=1$，第四行$\hat{C_i}=0$），这样调整优化目标**使得网络试图预测其输出的预测框和真实边界框的IOU**，虽然会让网络在预测框回归上表现较好之后才开始提升置信度，但却能让网络学会评估自己输出的预测框的质量，而非简单地判断此区域是否包含了某物体的中心点。

### 四、推理流程

​		给定一张图片，网络完成前向传播后，将每个位置输出的概率密度最高值对应的类别当做预测框的类别，**将概率密度最高值与$B$个物体评分相乘之后当做预测框的置信度**，最后将这$S\times S\times B$个预测框送入NMS消除冗余后输出。

![YOLOv1推理流程](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov1-obj,cls.reg.jpg)

​		从推理流程可以看出，这样设置网络输出的一大弊端就是网络无法预测中心点在同一个方形区域内的两个不同类别的物体。

## YOLO v2——Focus结构，BN，先验框和逐框分类

### 一、网络结构

​		YOLOv2的网络结构相比于YOLOv1来说有如下几个变化：

1.   在网络中间加入了批归一化层(Batch Normalization)以加速收敛；
2.   抛弃了YOLOv1的全连接层，从Neck的输出到Head的输出之间的通道维度的变化使用1x1卷积来实现。这一方面带来了更小的参数数量，另一方面导致了有限的感受野；
3.   引入了Focus结构作为Neck以提升在小目标上的边界框预测质量；具体来说，这个模块以Backbone的倒数第二个阶段的特征图（下采样率为最后一个阶段的二分之一）为输入，逐通道将其分为数个2x2的小方格后，将每个小方格中左上、右上、左下和右下的值分别**在宽高上拼接起来**之后再**在通道维拼接起来**，形成一个宽高为原来一半，但通道数是原来的四倍的一个特征图。

![Focus结构](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-focus.jpg)

代码实现如下：

```python
class Focus(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        assert input.shape[-2] % 2 == 0 and input.shape[-1] % 2 == 0
        return torch.cat([input[:, :, i::2, j::2] for i in (0, 1) for j in (0, 1)],
                         dim=1)
```

​		随后，**将这个特征图和Backbone最后一个阶段输出的特征图在通道维拼接后，当做Head的输入进行预测。**这样做能够使Head在进行预测时同时获取到语义信息较为丰富的最后一层特征和几何信息相对丰富的倒数第二层特征，对于网络预测较小尺寸的物体有所帮助。

### 二、模型输出

​		与YOLOv1为每个方形区域预测类别不同，YOLOv2**对每个预测框都单独预测类别**，这使得YOLOv2可以预测中心点距离较近的异类物体。仍以$S$代表特征图高宽，$B$代表单个区域上的预测框个数，$C$代表数据集物体类别总数，则YOLOv2输出的特征图的通道数为$B\times (5+C)$(图源https://blog.csdn.net/litt1e/article/details/88852745)：

![YOLOv2对网络输出的解释](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov2-output.jpg)

​		YOLOv2在模型输出上与YOLOv1的另一个不同点是其使用了在数据及上聚类而来的先验框来帮助预测物体的边界框。具体来说，作者将数据集中所有真实边界框的高宽收集起来后，进行了簇数$k=5$，距离度量$d(\mathrm{box_1}, \mathrm{box_2})=1-\mathrm{IOU}(\mathrm{box_1}, \mathrm{box_2})$的K-means聚类算法，最后选用5个聚类中心的宽高作为5个先验框，并按下图利用网络输出完成对先验框的修正（$\mathrm b_*$代表预测框坐标，$\mathrm p_*$代表先验框宽高）：

![预测框解码方式](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov2-box.jpg)

​		代码实现如下：

```python
def refine_box(box_cxcywh, shift, downsample_rate=32):
    box = box_cxcywh.to(shift.device)
    b_cx = downsample_rate * (torch.sigmoid(shift[:, 0]) + (box[:, 0] / downsample_rate).floor())
    b_cy = downsample_rate * (torch.sigmoid(shift[:, 1]) + (box[:, 1] / downsample_rate).floor())
    b_w = box[:, 2] * torch.exp(shift[:, 2])
    b_h = box[:, 3] * torch.exp(shift[:, 3])
    return torch.stack([b_cx, b_cy, b_w, b_h], dim=1)
```

​		值得注意的一点是，如果我们看原文中的消融实验部分，可以看到"Anchor Box"一栏是没有被勾选的。

![消融实验](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov2-ablation.jpg)

​		但通过上文所述的YOLOv2的预测框解码方式可以看出，若网络输出$t_x=t_y=0$，则预测框的中心点就位于其所在方形区域的正中心，因此我们也可以将这些先验框看做是预先配置在每个方形区域中间的"锚框"。

### 三、模型训练

​		YOLOv2和YOLOv1的训练方法仅在预测框和真实边界框的匹配策略上有一点不同：在匹配时，计算的是某真实边界框和其对应的$B$个**锚框**的IOU，而不是和$B$个**预测框**的IOU。

​		另外，由于每个预测框都单独预测类别，因此计算分类损失时需要遍历所有与真实边界框相匹配的预测框，而不是遍历所有$S\times S$个方形区域。

### 四、总结

​		YOLOv2相比于YOLOv1，主要有以下几个改进：

| YOLOv1                           | 而YOLOv2                                                     |
| -------------------------------- | ------------------------------------------------------------ |
| 使用了全连接层，参数量巨大       | 摒弃了全连接层，全部使用卷积层，参数量小，但感受野有限       |
| 直接输出预测框坐标               | 基于先验框预测边界框，网络易于学习                           |
| 对每个方形区域仅输出一组类别概率 | 为所有方形区域内的每个预测框都单独输出类别概率，可以检测中心点较近的异类物体 |
| 仅用Backbone最后一张特征图做预测 | 使用Backbone倒数第一、第二阶段特征图用Focus结构处理后共同预测，改善了小目标上的检测效果 |
| 没有批归一化层                   | 加入了批归一化层，加速收敛                                   |

## YOLO v3——残差连接，FPN特征金字塔结构，多尺度预测

### 一、网络结构

​		YOLOv3的网络结构如下图所示。图源：https://blog.csdn.net/leviopku/article/details/82660381

![YOLOv3 网络架构](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov3-archi.jpg)

​		YOLOv2虽然在YOLOv1的基础上做了几点改动，但总体上网络结构仍与图片分类所用的CNN类似。但YOLOv3的网络结构相比于前两个版本来说更加复杂，下面我们分别从Backbone、Neck和Head三个部分分别阐述网络架构的升级点。

1.   Backbone方面，YOLOv3采用了ResNet所提出的残差结构，这使得网络能够堆叠更多的卷积层而无需担心模型退化的问题。
2.   Neck方面，YOLOv3采用了http://arxiv.org/abs/1612.03144所提出的特征金字塔(Feature Pyramid Network, FPN)结构，这种结构将网络深层语义信息较为丰富的特征图与浅层空间信息较为完整的特征图进行融合。**通过将丰富的语义信息自顶向下地带回较早阶段的特征图，这样的结构使得模型能够在精确预测不同大小的物体边界框的同时不损失分类精度。**
3.   Head方面，不同于前两个版本，由于YOLOv3的Neck输出三张不同大小的特征图，因此需要有三个参数各不相同的Head在三个尺寸的特征图上分别进行检测。

### 二、模型输出

​		YOLOv3与YOLOv2定义模型输出的方式极为相似，通道数都为$B\times(5+C)$。唯一的不同点是，**YOLOv3的前向传播得到的是三张不同大小的特征图**，尺寸较大的特征图由于元素较多，因此对原图的方形区域划分也较为精细，再加上较大的特征图本就保留了较多空间几何信息，因此被用来检测尺寸较小的物体；同理，中等尺寸的特征图被用来检测中等大小的物体，较小的特征图被用来检测尺寸较大的物体。

​		YOLOv3和YOLOv2相同，都使用预先在数据集上聚类得到的簇中心点作为先验框，但YOLOv3中使用了9个先验框，而不是YOLOv2中的5个。**三个较小的先验框被分给较大的特征图，三个中等大小的先验框被分配给中等大小的特征图，尺寸较小的特征图则以三个最大的先验框为基础进行预测。**因此，YOLOv3中每个检测头Head的$B=3$。

![特征图尺寸和先验框大小的关系](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov3-fmap-and-prior.jpg)

### 三、模型训练

​		YOLOv3的训练方法与YOLOv2有如下不同：

1.   分类损失采用了二分类交叉熵（逻辑回归）而非多分类交叉熵或均方误差；
2.   预测框中心点cx、cy的损失函数采用了二分类交叉熵；
3.   由于有多尺度特征图，**每个真实边界框需要在三种粒度的网格共$3\times B$个锚框里匹配与之IOU最高的锚框；**

### 四、总结

​		YOLOv3相比于YOLOv2，主要有以下几个改进：

| YOLOv2                         | 而YOLOv3                                                     |
| ------------------------------ | ------------------------------------------------------------ |
| 使用传统单前向通路网络结构     | 使用了残差结构，能够使用更深的Backbone网络                   |
| 使用Focus结构融合多尺度特征图  | 使用特征金字塔FPN结构融合多尺度特征图，保证了小目标的分类准确度 |
| 检测头仅使用一张特征图进行预测 | FPN输出三张特征图，三个检测头分别在三张特征图上预测多种尺寸的物体 |
| 使用均方误差计算分类损失       | 使用二分类交叉熵计算分类损失                                 |

## YOLO v4——CSP结构，PAN结构，和一堆其他Trick的堆砌

### 一、网络结构

​		YOLOv4的网络结构如下图所示。

![YOLOv4 网络架构](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov4-archi.jpg)

1.   Backbone方面，YOLOv4加入了CSP结构，在不减小模型容量的前提下减小了模型的参数数量和计算复杂度；激活函数由Leaky ReLU替换为Mish，采用DropBlock作为正则化方法，略微提高了模型精度；

![CSP结构](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov4-CSPblock.jpg)

![Dropblock](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov4-dropblock.jpg)

2.   Neck方面，一方面使用了空间金字塔池化(Spatial Pyramid Pooling, SPP)结构来扩大感受野，另一方面采用了http://arxiv.org/abs/1803.01534中提出的PAN结构，又**将浅层较为丰富的几何信息传递给深层，进一步增强了网络输出的预测框的精确度。**SPP代码如下：

```python
class SPP(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = F.max_pool2d(x, kernel_size=5,  stride=1, padding=2)
        y2 = F.max_pool2d(x, kernel_size=9,  stride=1, padding=4)
        y3 = F.max_pool2d(x, kernel_size=13, stride=1, padding=6)
        return torch.cat([x, y1, y2, y3], dim=1)
```

3.   Head方面，YOLOv4与YOLOv3并无变化。

### 二、模型输出

​		和YOLOv3完全一致。

### 三、模型训练

​		将预测框的损失函数替换为了CIOU_Loss。各类IOU_Loss的讲解详见https://zhuanlan.zhihu.com/p/94799295。

### 四、总结

​		YOLOv4并非前三个版本的原作者所作，并且通常在很长一段时间内被认为是简单的将tricks堆砌起来的产物。作者将YOLOv4相较于YOLOv3新增的trick分为两大类：免费包(Bag of Freebies, BoF)，指仅增加训练成本，不增加推理成本却能提升模型性能的trick；特价包(Bag of Specials, BoS)，指仅小幅度增加推理成本，却能大幅提升模型性能的trick。

| BoF                            | 链接                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| CutMix数据增强、Mosaic数据增强 | https://zhuanlan.zhihu.com/p/405639109                       |
| DropBlock正则化                | https://zhuanlan.zhihu.com/p/469849057                       |
| Label Smoothing类别标签平滑    | https://www.cnblogs.com/irvingluo/p/13873699.html            |
| Mish激活函数                   | https://pytorch.org/docs/stable/generated/torch.nn.Mish.html#torch.nn.Mish |
| CIOU-Loss                      | https://zhuanlan.zhihu.com/p/94799295                        |
| CmBN跨小批量批归一化           | https://blog.csdn.net/qq_38253797/article/details/116847588，https://blog.csdn.net/qq_35447659/article/details/107797737 |
| SAT自对抗训练                  | https://arxiv.org/pdf/1703.08603.pdf                         |
| 单个真实边界框匹配多个锚框     |                                                              |
| 余弦退火学习率                 | https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html |
| 随机选择训练图片的尺寸         |                                                              |

| BoS                         | 链接                                                         |
| --------------------------- | ------------------------------------------------------------ |
| Mish激活函数                | https://pytorch.org/docs/stable/generated/torch.nn.Mish.html |
| CSP跨阶段部分链接           | https://blog.csdn.net/qq_44666320/article/details/108188558  |
| SPP空间金字塔池化层         | http://arxiv.org/abs/1406.4729                               |
| SAM空间注意力机制           | http://arxiv.org/abs/1904.05873                              |
| PAN路径融合Neck             | http://arxiv.org/abs/1803.01534                              |
| NMS使用DIOU计算候选框相似度 |                                                              |

## YOLO v5——最实用的目标检测算法

​		YOLOv5网络结构图：

![YOLOv5 网络架构](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov5-archi.jpg)

​		作为目前最方便应用的目标检测算法，YOLOv5相对于v4的改进较少，详见这篇文章：https://blog.csdn.net/nan355655600/article/details/107852353

## YOLO算法总结

​		YOLO系列作为现阶段知名度最高，应用最广的目标检测算法，具有思路简单、结构清晰的特点，每一代相对于上一代的改进点都有明确的动机。笔者总结了一下各代相对于上一代的改进点，供读者参考。

![增量式学习YOLO](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolo-increment.jpg)

## 自己动手实现一个简单的YOLO算法！

​		笔者同样作为一个初学者在学习YOLO系列算法时也遇到了一些坎坷和困惑，其中绝大部分的困惑都由论文的描述无法为实现算法提供精确的指导而造成。在这部分中，我们动手实现一个类似YOLOv4的简化版YOLO算法，完成在Pascal VOC数据集上的目标检测任务，以求用代码来帮助各位看官更细节地理解算法。

### 导入所需的包

```python
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
%matplotlib inline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 设置随机种子
seed = 42 # 宇宙的答案
torch.manual_seed(seed)
random.seed(seed)
```

### 定义数据集

```python
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
        # 缩放图片
        image = T.Resize(size)(image)
        # 缩放边界框
        box_coords[:, 0::2] = (box_coords[:, 0::2] * resize_ratio[0]).int()
        box_coords[:, 1::2] = (box_coords[:, 1::2] * resize_ratio[1]).int()
        return image, box_coords
    
    def __getitem__(self, index):
        """判断是使用07年的数据还是12年的数据"""
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
        """
        获取图片中物体的类别和真实边界框的xyxy坐标，
        这部分代码不是很理解的话可以将原始数据集的标签打印出来对照查看。
        """
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
            # 翻转图片
            image = T.RandomHorizontalFlip(p=1)(image)
            x1, x2 = box_coords[:, 0], box_coords[:, 2]
            # 翻转边界框
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
```

### 定义预测框的修正方式

![预测框解码方式](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov2-box.jpg)

```python
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
```

### 定义网络结构

#### Backbone部分

```python
class Backbone(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        # YOLO的backbone需要输出三张特征图
        # 三张特征图的下采样率分别为8、16和32
        # module_dict中每个值都是一个包含四个元素的元组
        # 其中第一个元素是使用torchvision的API取backbone模型的函数
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
        # 根据模块名获取backbone的三个部分
        self.backbone_ds8  = nn.Sequential(*[raw_backbone[key] for key in module_dict[backbone_name][1]])
        self.backbone_ds16 = nn.Sequential(*[raw_backbone[key] for key in module_dict[backbone_name][2]])
        self.backbone_ds32 = nn.Sequential(*[raw_backbone[key] for key in module_dict[backbone_name][3]])
        
    def forward(self, input):
        """用网络的三个部分依次计算下采样率为8、16和32的特诊图"""
        fmap_s8 = self.backbone_ds8(input)
        fmap_s16 = self.backbone_ds16(fmap_s8)
        fmap_s32 = self.backbone_ds32(fmap_s16)
        return fmap_s8, fmap_s16, fmap_s32
```

#### Neck部分

```python
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
```

![YOLOv4 网络架构](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5C%E5%9B%BE%E5%BA%8Adetection-yolov4-archi.jpg)

```python
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
```

#### 检测头部分

```python
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
```

#### 网络整体

```python
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
```

### 定义损失函数

​		`ComputeLoss`类中，我们不仅需要定义损失函数的计算方式，还需要定义将真实边界框与锚框匹配起来的机制。

```python
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
```

### 训练网络

#### 单步训练

```python
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
```

#### 学习率调整

```python
def warmup_lr_ratio(warmup_steps, cur_step, power=1.):
    if cur_step == 0:
        return 0
    lr_ratio = min(cur_step ** -power,
                   (cur_step * warmup_steps ** -(1.+power))) * warmup_steps ** power
    return lr_ratio
```

#### 训练代码

```python
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
```

### 超参数设置

```python
class Configuration:
    def __init__(self):
        self.version = 'version 10'
        self.backbone = 'resnet50'
        self.num_classes = 20
        self.neck_hidden_layers = 2
        self.head_hidden_layers = 2
        self.hidden_channels = 256
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
        
        self.lr = 1e-2
        self.warmup_steps = 0.1
        self.lr_decay_power = 0.5
        self.batch_size = 16
        self.num_epochs = 100
        self.weight_decay = 0
        self.num_workers = 8
```

### 其他实用函数

```python
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

def init_weight(module):
    """递归初始化模型参数"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Sequential, nn.ModuleList)):
        for m in module:
            init_weight(m)
            
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

def get_lr(optimizer):
    return (optimizer.state_dict()['param_groups'][0]['lr'])
```

