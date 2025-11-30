# 第1周：深度学习基础

> 学习要求：
>
> - 深度学习的入门知识
> - pytorch 基础练习，螺旋数据分类代码练习 
>

<br>

## 引言

本周学习视频为“01-深度学习概述”，下载链接为：https://www.jianguoyun.com/p/Dde3HS8QrKKIBhi2xpEGIAA



<br>


## 1、视频学习

学习视频：深度学习基础，主要内容包括：

- 浅层神经⽹络：⽣物神经元到单层感知器，多层感知器，反向传播和梯度消失
- 神经⽹络到深度学习：逐层预训练，⾃编码器和受限玻尔兹曼机

<br>

## 2、代码练习

学习完成以后进行代码练习，代码练习需要使⽤⾕歌的 Colab，它是⼀个 Jupyter 笔记本环境，已经默认安装好 pytorch，不需要进⾏任何设置就可以使⽤，并且完全在云端运⾏。使⽤⽅法可以参考 Rogan 的博客：https://www.cnblogs.com/lfri/p/10471852.html 

国内⽬前⽆法访问 colab，可以安装 Ghelper: http://googlehelper.net/

### **练习1：pytorch 基础练习**

基础练习部分包括 pytorch 基础操作，实验指导在第4.1节

**要求：** 把代码输⼊ colab，在线运⾏观察效果。

### **练习2：螺旋数据分类**

⽤神经⽹络实现简单数据分类，实验指导在第4.2节。

运行代码会发现少了一个图片，原作者移动位置了，新的位置在： https://raw.githubusercontent.com/Atcold/pytorch-Deep-Learning/master/res/ziegler.png

**要求：** 把代码输⼊ colab，在线运⾏观察效果

<br>



## 3、博客作业

完成⼀个博客，内容包括两部分：

【第⼀部分：代码练习】在⾕歌 Colab 上完成 pytorch 代码练习中的 3.1 pytorch基础练习、3.2螺旋数据分类，关键步骤截图，并附⼀些自己的想法和解读。

【第⼆部分：问题总结】思考下⾯的问题：

1、AlexNet有哪些特点？为什么可以比LeNet取得更好的性能？ 

2、激活函数有哪些作⽤？ 

3、梯度消失现象是什么？

4、神经网络是更宽好还是更深好？

5、为什么要使⽤Softmax? 

6、SGD 和 Adam 哪个更有效？

如果还有其它问题，可以总结⼀下，写在周打卡里，下周⼀起讨论。

<br>

## 4、实验环节

### 4.1 pytorch 基础练习

#### 什么是 PyTorch ?

PyTorch是一个python库，它主要提供了两个高级功能：

- GPU加速的张量计算
- 构建在反向自动求导系统上的深度神经网络

####  1. 定义数据

一般定义数据使用torch.Tensor ， tensor的意思是张量，是数字各种形式的总称

```Python
import torch

# 可以是一个数
x = torch.tensor(666)
print(x)
```

输出：tensor(666)

<br>

```Python
# 可以是一维数组（向量）
x = torch.tensor([1,2,3,4,5,6])
print(x)
```

输出：tensor([1, 2, 3, 4, 5, 6])

<br>

```Python
# 可以是二维数组（矩阵）
x = torch.ones(2,3)
print(x)
```

输出：tensor([[1., 1., 1.],

​                    [1., 1., 1.]])

<br>

```Python
# 可以是任意维度的数组（张量）
x = torch.ones(2,3,4)
print(x)
```

输出：

tensor([[[1., 1., 1., 1.],

​         [1., 1., 1., 1.],

​         [1., 1., 1., 1.]],

​        [[1., 1., 1., 1.],

​         [1., 1., 1., 1.],

​         [1., 1., 1., 1.]]])

Tensor支持各种各样类型的数据，包括：

torch.float32, torch.float64, torch.float16, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64 。这里不过多描述。

创建Tensor有多种方法，包括：ones, zeros, eye, arange, linspace, rand, randn, normal, uniform, randperm, 使用的时候可以在线搜，下面主要通过代码展示。

```Python
# 创建一个空张量
x = torch.empty(5,3)
print(x)
```

输出：

tensor([[1.4178e-36, 0.0000e+00, 4.4842e-44],

​        [0.0000e+00,        nan, 0.0000e+00],

​        [1.0979e-05, 4.2008e-05, 2.1296e+23],

​        [1.0386e+21, 4.4160e-05, 1.0742e-05],

​        [2.6963e+23, 4.2421e-08, 3.4548e-09]])

<br>

```Python
# 创建一个随机初始化的张量
x = torch.rand(5,3)
print(x)
```

输出：

tensor([[0.3077, 0.0347, 0.3033],

​        [0.9099, 0.2716, 0.4310],

​        [0.8286, 0.3317, 0.0536],

​        [0.9529, 0.4905, 0.1403],

​        [0.6899, 0.8349, 0.4015]])

<br>

```Python
# 创建一个全0的张量，里面的数据类型为 long
x = torch.zeros(5,3,dtype=torch.long)
print(x)
```

输出：

tensor([[0, 0, 0],

​        [0, 0, 0],

​        [0, 0, 0],

​        [0, 0, 0],

​        [0, 0, 0]])

<br>

```Python
# 基于现有的tensor，创建一个新tensor，
# 从而可以利用原有的tensor的dtype，device，size之类的属性信息
y = x.new_ones(5,3)   #tensor new_* 方法，利用原来tensor的dtype，device
print(y)
```

输出：

tensor([[1, 1, 1],

​        [1, 1, 1],

​        [1, 1, 1],

​        [1, 1, 1],

​        [1, 1, 1]])

<br>

```Python
z = torch.randn_like(x, dtype=torch.float)    
# 利用原来的tensor的大小，但是重新定义了dtype
print(z)
```

输出：

tensor([[ 1.4363, -2.1019,  0.4444],

​        [-0.4706,  0.7441, -0.4631],

​        [-1.3860, -1.8919,  1.8794],

​        [ 1.8617,  0.6469,  0.5235],

​        [-0.1271, -1.0755,  0.0359]])

<br>



#### 2. 定义操作

凡是用Tensor进行各种运算的，都是Function

最终，还是需要用Tensor来进行计算的，计算无非是

- 基本运算，加减乘除，求幂求余
- 布尔运算，大于小于，最大最小
- 线性运算，矩阵乘法，求模，求行列式

基本运算包括： abs/sqrt/div/exp/fmod/pow ，及一些三角函数 cos/ sin/ asin/ atan2/ cosh，及 ceil/round/floor/trunc 等具体在使用的时候可以百度一下

布尔运算包括： gt/lt/ge/le/eq/ne，topk, sort, max/min

线性计算包括： trace, diag, mm/bmm，t，dot/cross，inverse，svd 等

不再多说，需要使用的时候百度一下即可。下面用具体的代码案例来学习。

```python
# 创建一个 2x4 的tensor
m = torch.Tensor([[2, 5, 3, 7],
                  [4, 2, 1, 9]])

print(m.size(0), m.size(1), m.size(), sep=' -- ')
```

2 -- 4 -- torch.Size([2, 4])

<br>

```python
# 返回 m 中元素的数量
print(m.numel())
```

8

<br>

```Plain
# 返回 第0行，第2列的数
print(m[0][2])
```

tensor(3.)

<br>

```Plain
# 返回 第1列的全部元素
print(m[:, 1])
```

tensor([5., 2.])

<br>

```Plain
# 返回 第0行的全部元素
print(m[0, :])
```

tensor([2., 5., 3., 7.])

<br>

```Plain
# Create tensor of numbers from 1 to 5
# 注意这里结果是1到4，没有5
v = torch.arange(1, 5)
print(v)
```

tensor([1, 2, 3, 4])

<br>



```Plain
# Scalar product
m @ v
```

tensor([49., 47.])

<br>

```Plain
# Calculated by 1*2 + 2*5 + 3*3 + 4*7
m[[0], :] @ v
```

tensor([49.])

<br>

```Plain
# Add a random tensor of size 2x4 to m
m + torch.rand(2, 4)
```

tensor([[2.2495, 5.7699, 3.3819, 7.0271],

​        [4.4853, 2.1948, 1.8039, 9.2615]])

<br>

```Plain
# 转置，由 2x4 变为 4x2
print(m.t())

# 使用 transpose 也可以达到相同的效果，具体使用方法可以百度
print(m.transpose(0, 1))
```

tensor([[2., 4.],

​        [5., 2.],

​        [3., 1.],

​        [7., 9.]])

tensor([[2., 4.],

​        [5., 2.],

​        [3., 1.],

​        [7., 9.]])

<br>

```Plain
# returns a 1D tensor of steps equally spaced points between start=3, end=8 and steps=20
torch.linspace(3, 8, 20)
```

tensor([3.0000, 3.2632, 3.5263, 3.7895, 4.0526, 4.3158, 4.5789, 4.8421, 5.1053,

​        5.3684, 5.6316, 5.8947, 6.1579, 6.4211, 6.6842, 6.9474, 7.2105, 7.4737,

​        7.7368, 8.0000])

<br>

```Plain
from matplotlib import pyplot as plt

# matlabplotlib 只能显示numpy类型的数据，下面展示了转换数据类型，然后显示
# 注意 randn 是生成均值为 0， 方差为 1 的随机数
# 下面是生成 1000 个随机数，并按照 100 个 bin 统计直方图
plt.hist(torch.randn(1000).numpy(), 100);
```

<p align=center><img src =https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/ScreenShot_2025-11-30_193541_304.jpg width=30%></p>

```Plain
# 当数据非常非常多的时候，正态分布会体现的非常明显
plt.hist(torch.randn(10**6).numpy(), 100);
```

1000000

<br>

<p align=center><img src=https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/ScreenShot_2025-11-30_193650_719.jpg width=30%></p>

<br>

```Plain
# 创建两个 1x4 的tensor
a = torch.Tensor([[1, 2, 3, 4]])
b = torch.Tensor([[5, 6, 7, 8]])

# 在 0 方向拼接 （即在 Y 方各上拼接）, 会得到 2x4 的矩阵
print( torch.cat((a,b), 0))
```

tensor([[1., 2., 3., 4.],

​        [5., 6., 7., 8.]])

<br>

```Plain
# 在 1 方向拼接 （即在 X 方各上拼接）, 会得到 1x8 的矩阵
print( torch.cat((a,b), 1))
```

tensor([[1., 2., 3., 4., 5., 6., 7., 8.]])

<br>

**One more thing ~**

其实基本操作还有非常非常多，详细可以查阅官方文档。
