# Week 9 Swin Transformer

原文地址：[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

官方源代码地址：https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

上期回顾：[Week 8 YOLO系列](../Week8-YOLOseries/Week 8 YOLOseries.md)

## 简介

​		基于自注意力机制的Transformer模型在自然语言处理领域的成功引起了计算机视觉研究者的注意。近年来，有越来越多的研究者正试图将Transformer应用于视觉领域。但Transformer终究还是为了解决NLP领域的问题而设计的，将其应用到视觉领域会遇到两个需要解决的问题：

1.   在NLP领域，具有完整语义信息的一个个体通常仅为一个单词或几个词元的组合体，尺度较小且较为固定，而视觉领域中，**一个完整的带有独立语义信息的物体在图片中可大可小，尺寸变化较大**；
2.   NLP领域中，一个句子或一个段落被分割成token后得到的序列长度仍然在几十至几百不等，而视觉任务所处理的图片数据通常拥有成千上万甚至百万个像素，简单地将图像中的像素展平成一个向量会导致**巨大的序列长度**，这对于计算复杂度与序列长度的平方成正比的Transformer来说是不可接受的。

​		2021年，Vision Transformer横空出世，力求**对原始Transformer结构做最少的改动**而后将其应用于视觉领域，并取得了极佳的效果。ViT通过**将图片分割成16x16的图片块**，并在图片块之间做自注意力计算的方式大大缩短了序列长度，使得Transformer应用于视觉任务成为可能。

![Vision Transformer模型架构](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5Cpicturesvit-architecture.png)

​		但ViT并非十全十美。首先，由于完全采用原始Transformer架构，几乎不含任何与视觉任务相关的先验信息，因此**需要在大量数据的训练下才能表现出良好的性能**。其次，ViT通过分割图片块的方式缩短了序列长度，但自注意力操作仍需在整张图片上的所有图片块之间进行，**计算复杂度仍与图片面积的平方成正比**。最后，ViT论文中仅仅对Transformer应用于图像分类任务进行了尝试，这仅需单一尺寸的图片特征即可完成。而许多其他的视觉任务（尤其是密集预测性任务，如目标检测和语义、实例分割）都**需要模型输出多尺寸的特征**，这恰是ViT所不能提供的。

​		为了获得一个能够作为**视觉任务通用骨干网络**的基于Transformer架构的视觉模型，本文的作者提出了"使用移动窗口注意力的层级式Vision Transformer"，简称Swin Transformer。通过基于ViT的一系列改进，Swin做到了**正比于图像面积的计算复杂度**和**多尺寸特征输出**，为Transformer在视觉领域应用于更大的图片和更多的任务铺平了道路。

## 模型架构和方法

​		在这一节中，我们将通过模型的演化动机和Swin Transformer解决的问题来引出并认识Swin的结构。先上一张总体结构图：

![Swin Transformer](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5C%E5%9B%BE%E5%BA%8A%5CWeeklyPaper_imagesswin-archi.png)

### Patch Partition+Linear Embedding

​		与ViT中相同，Swin将图片均分为数个高宽为$p$个像素的小patch，随后将一个patch内的像素值拉直成一个向量，最后将这个向量经过一个**在所有patch之间共享的线性层**（参数矩阵$\mathbf E\in\mathbb R^{3p^2\times C}$）变换为一个$C$维的向量送入Transformer。

​		这两步操作在代码中被等效为一个卷积核大小为4，步长为4，输出通道数为$C$的卷积层。

### 究竟是CNN出了轨 还是ViT劈了腿

#### 无限感受野还是空间局部性？这是一个问题

​		让我们回忆一下Transformer的优势：它其中的所有模块平等地对待序列中的每一个元素，元素之间并没有顺序关系，序列中的**每个元素都能和另外所有元素建立关联**，这赋予了Transformer极强的全局建模能力。移植到视觉领域上之后，Transformer继承了这一特性，但这也带来了极大的计算复杂度。ViT将图片分割成图片块，将图片块内所有像素拼接过后经过投影变为一个向量，当做序列中的一个元素处理，从而将序列长度减小了${patch\_size^2}$倍，但由于各个图片块仍需与所有图片块建立关联，因此计算复杂度仍与图片面积的平方成正比，但好处是序列中的每个元素在各层均有覆盖全图的感受野。

​		空间局部性和平移等变性是CNN特有的的先验知识。ViT的作者通过实验证明了在模型和数据集的体量都足够大的情况下，大规模预训练后的Transformer的性能要优于自带先验知识的CNN。但Transformer强大的建模能力是以计算复杂度为代价的。从ViT论文中这张图可以看到，在ViT较浅的层中，大部分与某个元素关联性较大的元素与该元素在空间上的距离（用平均注意力距离来衡量，即用注意力权重加权的空间距离之和）都较小。

![ViT和混合模型各层的平均注意力距离](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5Cpicturesvit-attention-weight.png)

​		空间局部性是指，图片中两个有关联的物体，或同一物体的两部分，空间距离大概率较小。在较浅的层中，网络仍处在对图片的低级特征做处理、整合的阶段，因此作者认为，**在网络浅层保证能够建立起图片的任意两个区域的联系，一方面能使网络拥有无限的感受野，但另一方面实际上是对计算资源的浪费。**为了降低计算复杂度，我们可以将空间局部性适当引入Vision Transformer，使得较浅的层中各个元素仅与其临近的元素做自注意力操作，模仿CNN在网络深度增加的过程中逐渐扩大感受野直至感受野覆盖全图。

#### 窗口注意力（Window Multihead Self-Attention，W-MSA）

​		一个直观的想法是：模仿CNN，以每个用作query的元素为中心划定出一个小区域，作为query的元素仅与该小区域内的元素计算注意力分数，也即仅将CNN中的卷积操作替换成范围和卷积核大小相等的注意力机制。但这样做的缺点较为明显：每个query元素计算注意力分数的元素范围都不一样，导致计算复杂度较高，而且感受野扩大较慢。

​		Swin Transformer采用了**窗口注意力**，如下图：

![Swin Transformer对比ViT](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5Cpicturesswin-compare-to-vit.png)

​		图中，黑色方框分割出的$p\times p$个像素的小区域为基本计算单元，ViT将图片分割成$p=16$的图片块后作为序列中的元素，而Swin则将图片分为更多更小的$p=4$的小块。红色方框代表自注意力的计算范围，共有$\frac{H}{M}\times\frac{W}{M}$个，其中$H,W$代表图片高宽，每个红色方框内可容纳$M^2$个$p=4$的图片patch（图中$M=4$）。从图中可以看到，ViT每一层自注意力的计算范围都为整张图片，而**Swin每一层自注意力的计算范围仅为该patch所在的$M\times M$的红色窗口内部，即仅计算一个窗口内部所有元素互相之间的注意力分数，处于不同窗口的图片patch并无关联。**这样，经过一次自注意力操作，该窗口内所有元素的感受野就都为其所在的红色窗口了。

##### 窗口注意力如何解决计算复杂度问题

​		自注意力机制的计算复杂度与参与计算自注意力的元素的数量的平方成正比。下面我们看看Swin的这一点改动是如何有效降低计算复杂度的。

​		多头自注意力的计算公式如下：

![多头注意力的计算公式](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5Cpicturestransformer-multi-head-attention-equation)

​		计算流程如下：

![多头自注意力的计算流程](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5CWeeklyPaper%5Cpicturestransformer-multi-head-attention-visualization.jpg)

​		若不计Softmax函数的计算复杂度，那么上图中第一步的计算复杂度为$3nd^2$，第二步的计算复杂度为$n^2d$， 第三步的计算复杂度为$n^2d+nd^2$，其中$n$为参与自注意力计算的序列长度，在ViT中$n=(H/p)\times (W/p)$，故整个多头注意力的计算复杂度为$4nd^2+2n^2d=4(HW)d^2/p^2+2(HW)^2d/p^4$，其中$n$为参与自注意力计算的序列长度，$d$为序列中每个元素特征向量的长度。

​		在将图片分割成数个$n=M\times M$的窗口之后，窗口内的元素数量固定，且窗口数量$(H/pM)\times(W/pM)$与图片面积成正比。因此，窗口注意力的计算复杂度为$(H/pM)\times(W/pM)\times (4M^2d^2+2M^4d)$$=4(HW)d^2/p^2+2(HW)M^2d/p^2$。

​		对比$\Omega(MSA)=4(HW)d^2/p^2+2(HW)^2d/p^4$和$\Omega(WMSA)=4(HW)d^2/p^2+2(HW)M^2d/p^2$可知，前者与图片面积的平方成正比，后者仅与图片面积的一次方成正比，Swin就是这样通过窗口分治注意力实现了计算复杂度的控制。

### 下采样方法Patch Merging——Focus结构+1x1卷积

​		解决了计算复杂度的问题，还有一个问题需要解决：Transformer中并没有下采样层，较多较小的patch构成的特征无法下采样成尺寸更小的特征图，模型无法输出多尺度特征，也就无法满足许多算法对多尺度特征的需要。为此，作者将Focus结构作为下采样手段，**将每4个相邻的patch特征在高宽维度割开后在通道维拼接，得到一张高宽为原来的一半，通道数是原来的4倍的特征图**。为了与CNN中下采样层"高宽减半，通道数加倍"的惯例保持一致，作者设定特征图经过Focus操作后还要**经过一个将通道数减半的1x1卷积层处理**，最后将下采样后的特征图送入下一阶段继续计算。作者将这个下采样操作称为Patch Merging。

![Focus结构](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:\图床\WeeklyPaper_imagesdetection-focus.jpg)

​		经过简单的思考不难发现，Focus+1x1卷积操作可以用一个卷积核大小为2，步长为2的卷积层代替，但作者没有这样做的原因是在Focus结构之后，1x1卷积之前，还有一个LayerNorm操作。

​		这样，通过在Transformer块之间插入这样的下采样块，就能使得**网络由浅到深输出的特征尺寸逐渐缩小**，使得将多尺度特征送入下游检测器成为了可能。

### 移动窗口注意力(**S**hifted **Win**dow Multihead Self-Attention, SW-MSA)

​		若Swin Transformer对原始Vision Transformer的修改就到此为止了，这多少有些矫枉过正了：当前配置下，下采样层来临之前，**每个阶段内各个patch特征的感受野都仅为其所在的窗口**，当且仅当下采样过后，感受野才会扩大。这就将**图片不同区域之间的关联性完全局限在窗口之内**了，各个窗口之间毫无关联，虽然解决了序列长度过长的问题，但却埋没了Transformer的优点——强大的全局建模能力。

​		为了解决这一问题，作者团队提出了移动窗口注意力，这也是该模型"Swin"的名称来源：**S**hift **Win**dow。具体来说，在一个使用窗口注意力的Transformer块之后，紧接着再放置另一个Transformer块。后者与使用窗口注意力的前者唯一的区别就是：每个计算自注意力的窗口在高、宽两个维度上各偏移了$\lfloor\frac{M}{2}\rfloor$个patch，见下图：

![移动窗口注意力](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5C%E5%9B%BE%E5%BA%8A%5CWeeklyPaper_imagesswin-shifted-window.jpg)

​		让我们用上图追踪一下移动窗口注意力是如何扩大感受野的：假设`Layer l`即为图片的第一个Transformer块，经过`Layer l`的处理后，蓝色的所有patch的特征的感受野即为蓝色区域，绿色的所有patch的特征的感受野即为绿色区域，以此类推。假设就这样堆叠`Layer l`，则每一个Transformer块输出的特征的感受野都还仅仅局限在其所在的窗口之中。

​		但若在`Layer l`之后增加一个`Layer l+1`，将窗口向左上方位移$\lfloor\frac{M}{2}\rfloor$个patch后，右下方四个蓝色的patch与左下方四个绿色patch、右上方四个紫色patch、左上方四个黄色patch的特征进行自注意力操作，也就能够获取到这些patch的特征所包含的信息。而这些patch由于`Layer l`已经包含了与其颜色相同的所有patch的信息，蓝色区域右下角四个patch**吞并了与其一同计算自注意力的patch的感受野**，因此蓝色区域右下角四个patch的感受野就已经是整个4个窗口了。

​		这样，仅仅是将窗口的划分做了一个简单的偏移，就构造出了移动窗口注意力。将窗口注意力和偏移后的移动窗口注意力交替使用，就能有效地迅速扩大各个patch的特征的感受野，增强窗口之间的信息交换，建立起图片各个位置的信息的联系，发挥出Transformer强大的全局建模能力。

### 相对位置编码

​		Transformer平等地对待输入序列中的每个元素，因此需要使用额外的手段来为模型注入每个patch的位置信息。

​		与Transformer、ViT和BERT采用的绝对位置编码方式不同，Swin采用了可学习的相对位置编码来为Transformer提供位置信息。具体来说，单个窗口内$M^2$个patch中，任意两个patch在行维度有$-M+1, -M+2,...,-1,0,1,...,M-2,M-1$共$2M-1$种行相对位置关系，同理在列维度也应有$2M-1$种列相对位置关系：

![相对位置坐标举例](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5C%E5%9B%BE%E5%BA%8A%5CWeeklyPaper_imagesswin-relative-coordinate.jpg)

​		相对位置编码为**每个Transformer块的每个自注意力头**定义一个维度为$[2M-1,2M-1]$的参数矩阵，称为相对位置偏置，形成一个形状为$[num\_heads,2M-1,2M-1]$的张量，在某个patch作为query时，将这个注意力头**所有patch与所有patch的相对位置**（$M^2$个patch共$M^4$个相对位置，拆分为行、列后共$(2M-1)^2$种行列相对位置组合）对应的相对位置张量的值（维度为$[M^2,M^2]$）与所有patch之间计算出的注意力分数（维度也为$[M^2,M^2]$）相加后再用Softmax归一化成注意力权重（注意力分数和注意力权重见[这篇文章](https://mp.weixin.qq.com/s/CGBNz-gv3Jrh1To-HWjw3Q)）。

​		这样的位置编码方式仅对patch之间的相对位置进行建模，而对某个patch在窗口内的绝对位置不敏感，某个patch需要参考其它所有patch的信息才能推断出自身处于其所在得窗口内的哪个位置。

​		具体为什么作者摒弃了绝对位置编码，而采用了更为复杂的相对位置编码来编码位置信息，相信读者看完下一节的讲述自然会得到答案。

### 循环偏移批量化+掩码，加速移动窗口注意力计算

![移动窗口注意力](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5C%E5%9B%BE%E5%BA%8A%5CWeeklyPaper_imagesswin-shifted-window.jpg)

​		从上图中不难看出，移动窗口注意力在偏移之后，会使得窗口注意力的各个窗口大小不一，且窗口数量由$\frac{H}{pM}\times\frac{H}{pM}$变为了$(\frac{H}{pM}+1)\times(\frac{H}{pM}+1)$。在计算注意力时，将窗口维分离出去，也当做一个"批量"对所有窗口进行并行计算能够缩短计算时间。但**窗口大小不一时，就无法将所有窗口拼接成一个张量**。解决这个问题的其中一个思路就是将不完整的窗口在缺失的位置补充0至尺寸与完整窗口相同，并在计算自注意力时用遮罩(mask)忽略掉补0的位置：

![填补不完整的窗口](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5C%E5%9B%BE%E5%BA%8A%5CWeeklyPaper_imagesswin-zero-padded-incomplete-windows.jpg)

​		但这样做的缺点是显而易见的：**窗口数量的增加浪费了计算资源**。上图中，窗口注意力的窗口数量为4，移动窗口注意力的窗口数量为9，计算量增加了225%。**计算量增加的比例随着网络深度的增加，也即窗口数量的减少而上升。**为了解决批量计算问题的同时不增加额外的计算量，作者设计了一种巧妙的循环位移方式，**将残缺的窗口配对，拼凑成完整的窗口后和本就完整的窗口一同计算**：

![循环位移拼凑完整窗口加速SW-MSA计算](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5C%E5%9B%BE%E5%BA%8A%5CWeeklyPaper_imagesswin-cyclic-shift.png)

​		循环位移及其逆过程可以分别用一行代码实现：

```python
# cyclic shift
shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
# reverse cyclic shift
x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
```

​		但这样会带来一个问题：本不相邻甚至相隔甚远的图片patch特征出现在了一个窗口中，作为同一个序列的成员参与自注意力的计算。从图片的角度上来说，相隔较远的图片块之间建立联系本是一件好事，增强图片不同区域间的联系也是视觉领域引入Transformer的初衷。

​		但是从相对位置编码的角度上来说，**本不完整的窗口中的各个patch都是由图片的不同位置拼凑而来，相对位置错乱，加入相对位置偏置时会误导模型，使得模型误以为这些实际上来自四面八方的patch是相邻的。**因此，拼凑而来的窗口中，原本来自不同窗口的patch之间不应当计算自注意力。实现时，只需要将来自不同窗口的patch之间计算出的注意力分数设为绝对值较大的负值（-100）即可。

​		在模型中，窗口注意力和移动窗口注意力需要交替使用，因此在模型的各个阶段中，Transformer块都为偶数，两两一对。

### 相对位置编码和循环位移的因与果

​		由于窗口划分偏移后会产生许多不完整的窗口，且图片靠左、靠上的特征在循环位移后会出现在新的窗口的右侧、下侧，因此绝对位置编码在Swin中并不适用，只能用相对位置编码。

​		由于相对位置编码编码的是同一个窗口中不同patch的相对位置，而循环位移会将本不相邻甚至相隔甚远的图片patch特征放入一个窗口之中，因此需要用遮罩来确保原本不来自于同一个窗口中的patch之间不计算自注意力。

![消融实验](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5C%E5%9B%BE%E5%BA%8A%5CWeeklyPaper_imagesswin-ablation-study.png)

​		上图的消融实验结果显示了在[**窗口注意力+移动窗口注意力**/仅窗口注意力]和[无位置编码/绝对位置编码/**相对位置编码**/绝对+相对位置编码]组合下的模型表现，可以看到相对位置编码和窗口注意力+移动窗口注意力的组合性能最优，与理论分析的结果相符。

### 总体结构

![Swin Transformer](https://cdn.jsdelivr.net/gh/zxh0916/WeeklyPaper_images/G:%5C%E5%9B%BE%E5%BA%8A%5CWeeklyPaper_imagesswin-archi.png)

​		图片输入进模型后，首先被打成patch，进入一对W-MSA+SW-MSA，随后重复"Patch Merging下采样+N*(W-MSA+SW-MSA)"的循环，最后根据任务需要提取出最后一层的特征或多级特征送入下游网络进行推理。**每一级网络都包含偶数个Swin Transformer块**，这是由于每个使用W-MSA的Transformer块后都会紧跟着一个使用SW-MSA的Transformer块，二者交替叠加。

## 总结

​		Swin Transformer主要干了两件事：

1.   窗口注意力、移动窗口注意力，将计算复杂度降为与图片面积的一次方成正比，同时性能不降反升；
2.   加入了下采样层，构造出了层级式的网络结构以输出多尺度特征，使得Swin可以作为视觉通用骨干网络。

​		这两点修改，都是为了**将Transformer更好地适配于视觉任务**，为模型大一统做出了不可磨灭的贡献。但对Transformer面向视觉任务进行针对性地修改之后，势必会增加将其应用在多模态领域的困难程度。作者也提到，希望之后的工作能够致力于将移动窗口注意力应用于NLP领域。因为这篇博客的作者我自己对多模态一窍不通，就不再胡言乱语了。