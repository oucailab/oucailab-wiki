# 实验4：MobileNet & ShuffleNet

> 学习要求：
>
> - CNN的基本结构：卷积、池化、全连接
> - 典型的⽹络结构：AlexNet、VGG、GoogleNet、ResNet
>
> **截止时间：11月2日（星期六） 22:00** 
>
> 大家务必注意时间，超出时间要扣分！

<br>

# 1、论⽂阅读与视频学习

## 1.1 MobileNet V1 & V2 

- MobileNet_V1_V2⽹络讲解 https://www.bilibili.com/video/BV1yE411p7L7/
- Pytorch搭建MobileNetV2⽹络 （有余⼒的同学可学习代码）https://www.bilibili.com/video/BV1qE411T7qZ/

## 1.2 MobileNet V3 

- MobileNet_V3⽹络讲解 https://www.bilibili.com/video/BV1GK4y1p7uE/
- Pytorch搭建MobileNetV3⽹络 （有余⼒的同学可学习代码） https://www.bilibili.com/video/BV1zT4y1P7pd/

## 1.3 ShuffleNet

- ShuffleNet ⽹络讲解 （ShuffleNetV1掌握就可以了，V2稍微了解下就好） https://www.bilibili.com/video/BV15y4y1Y7SY/
- 弄清楚 Channel 的 shuffle 是如何⽤代码实现的

## 1.4 SENet & CBAM

阅读Momenta公司的《ImageNet2017冠军模型-SENet详解》（[访问链接](https://cloud.tencent.com/developer/article/1052599)），掌握SENet的基本原理。同时，学习SENet的示例代码（[实验指导链接](https://gitee.com/gaopursuit/ouc-dl/blob/master/lab/week04_SENet_CIFAR10.ipynb)），数据加载和训练的代码可以参考ResNet（[访问链接](https://gitee.com/gaopursuit/ouc-dl/blob/master/lab/week04_Resnet_CIFAR10.ipynb)）

# 2、代码作业

阅读论⽂《HybridSN: Exploring 3-D–2-DCNN Feature Hierarchy for Hyperspectral Image Classification》，思考3D卷积和2D卷积的区别。并阅读HybridSN的代码 ([实验指导链接](https://gitee.com/gaopursuit/ouc-dl/blob/master/lab/week04_HybridSN_GRSL2021.ipynb)) 

把代码敲⼊ Colab 运⾏，⽹络部分需要⾃⼰完成。 

# 3、思考题

● 训练HybridSN，然后多测试⼏次，会发现每次分类的结果都不⼀样，请思考为什么？ 

● 如果想要进⼀步提升⾼光谱图像的分类性能，可以如何改进？ 

● depth-wise conv 和 分组卷积有什么区别与联系？ 

● SENet 的注意⼒是不是可以加在空间位置上？ 

● 在 ShuffleNet 中，通道的 shuffle 如何⽤代码实现？


## 5、评分规则

本次实验满分10分，按时提交并且内容符合要求8分，发布在个人博客并提供链接加0.5分，使用markdown加0.5分，内容质量高加0.5到1分。错过时间提交，扣1分。
