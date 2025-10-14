# 实验3：卷积神经网络

> 学习要求：
>
> - CNN的基本结构：卷积、池化、全连接
> - 典型的⽹络结构：AlexNet、VGG、GoogleNet、ResNet
>
> **截止时间：10月18日（星期六） 22:00** 
>
> 大家务必注意时间，超出时间要扣分！

<br>

## 引言

我们会有4~5周的时间学习深度学习的入门知识，为后面的小组项目打好基础。

本周视频为“深度学习概述”，下载链接为：https://www.jianguoyun.com/p/Da26fK0QrKKIBhi_0okGIAA



<br>




## 1、实验说明

实验报告推荐采用 Typora 软件编写，完成以后可以直接导出为PDF，也方便直接粘帖在 CSDN 等博客发布。实验报告模板如下：

```markdown
<center>姓名：XXXX  学号：XXXX</center>

| 姓名和学号？         | XXXX，200023230                     |
| ------------------- | ------------------------------      |
| 本实验属于哪门课程？  | 中国海洋大学25秋《软件工程原理与实践》 |
| 实验名称？           | 实验2：深度学习基础                  |
| 博客链接：           |  选做                               |

## 一、实验内容

XXXXXXXXXX

## 二、问题总结与体会

描述实验过程中所遇到的问题，以及是如何解决的。有哪些收获和体会，对于课程的安排有哪些建议。
```

<br>

## 2、视频学习

学习视频：卷积神经网络，主要内容包括：

- CNN的基本结构：卷积、池化、全连接
- 典型的⽹络结构：AlexNet、VGG、GoogleNet、ResNet

<br>

## 3、代码练习

###  实验3：MNIST数据集分类

构建简单的CNN对 mnist 数据集进⾏分类。同时，还会在实验中学习池化与卷积操作的基本作⽤， 实验指导链接：[实验3： 使用CNN对MINIST数据集分类](https://oucaigroup.feishu.cn/wiki/ZZHiwlpZJiLuNCkpL5ScHeCTnqg)

要求：把代码输入 colab，在线运行观察效果。

### 实验4：CIFAR10 数据集分类

使⽤ CNN 对 CIFAR10 数据集进⾏分类， 实验指导链接：[实验4：使用LeNet对CIFAR10数据分类](https://oucaigroup.feishu.cn/wiki/TWFdw1ecwiBILikSJrecvMX4nOf)

要求：把代码输入 colab，在线运行观察效果。

### 实验5：VGG16对CIFAR10分类

- 使⽤ VGG16 对 CIFAR10 分类，实验指导链接：[实验5：使用VGG对CIFAR10分类](https://oucaigroup.feishu.cn/wiki/UJ1xwmjVziixTckP1zFcgNgcnqc)

要求：把代码输入 colab，在线运行观察效果。

<br>



## 4、雨课堂提交实验报告：

完成⼀个实验报告，内容包括两部分：

【第⼀部分：代码练习】在⾕歌 Colab 上完成 pytorch 代码练习，关键步骤截图，并附⼀些自己的想法和解读。

【第⼆部分：问题总结】思考下⾯的问题：

- dataloader ⾥⾯ shuffle 取不同值有什么区别？
- transform ⾥，取了不同值，这个有什么区别？
- epoch 和 batch 的区别？
- 1x1的卷积和 FC 有什么区别？主要起什么作⽤？
- residual leanring 为什么能够提升准确率？
- 代码练习⼆⾥，⽹络和1989年 Lecun 提出的 LeNet 有什么区别？
- 代码练习⼆⾥，卷积以后feature map 尺⼨会变⼩，如何应⽤ Residual Learning?
- 有什么⽅法可以进⼀步提升准确率？

如果还有其它问题，可以总结一下，写在文档里，下周一起讨论。



## 5、评分规则

本次实验满分10分，按时提交并且内容符合要求8分，发布在个人博客并提供链接加0.5分，使用markdown加0.5分，内容质量高加0.5到1分。错过时间提交，扣1分。
