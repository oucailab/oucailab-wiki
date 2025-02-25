# 2. 云端上的深度学习

2023年以来，以DeepSeek、ChatGPT、Sora等为代表的预训练大模型持续取得突破，大模型的性能和效率不断提升，预训练成本不断下降，激发了人工智能赋能产业转型的巨大潜力。以DeepSeek为代表的开源大模型正逐渐渗透到诸多行业，引发新一代人工智能技术发展新浪潮。

在这个浪潮下，我们跃跃欲试。开始开发复杂模型时，尝试在本地计算机上训练模型通常不是一个可行的选择，因为我们本地的显存都比较受限制，而且也不是所有同学所在的实验室都有显卡资源。因此，比较推荐的方法是**在线上租服务器来训练**。

本节课将学习如何**利用`AutoDL`构建解决方案**，具体来说我们将探索和使用容器实例。这些课程将以现场演示/代码演练的形式进行。我们将首先完成 `AutoDL` 设置，在那里我们将配置并连接到实例，并介绍一些工具，这些工具可以帮助改善开发体验。

下一步将是调整现有代码，以便它可以与 `GPU` 一起使用，从而大大加快计算过程（例如训练模型）。同时也将讨论几种进一步加快模型训练速度的方法。最后我们将对 `CheXzero` 代码库进行一些实际操作：**把代码添加到我们的实例中，并确保我们可以运行模型训练过程。**



## 1、使用 AutoDL 进行深度学习

### 第一步：AutoDL 快速开始

AutoDL是国内知名的云GPU服务平台，我们首先登录网站注册。

![20250225005836](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225005836.jpg)

### 第二步：创建实例

注册后进入控制台，在我的实例菜单下，点击租用新实例。

![20250225010041](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225010041.jpg)

在租用实例页面：选择**计费方式**，选择合适的主机，选择要创建实例中的GPU数量，选择镜像（内置了不同的深度学习框架），最后创建即可。

![20250225010103](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225010103.jpg)

创建完成后等待开机，今后主要用到的操作入口见截图中。

![20250225010136](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225010136.jpg)

### 第三步：上传数据

开机后在这个正在运行中的实例上找到快捷工具：`JupyterLab`，点击打开，在下面的截图中找到上传按钮，即可上传数据。

![20250225010148](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225010148.jpg)

### 第四步：终端训练

在打开的`JupyterLab`页面中打开终端。

![20250225010201](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225010201.jpg)

在终端中执行Python命令等完成训练。

![20250225010227](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225010227.jpg)

### 第五步：转换为使用GPU

你可以使用以下命令检查远程实例的主机名以及是否加载了GPU：

```
nvidia-smi
```



这个命令非常有用，因为它可以显示你的GPU实际上是否被利用，这有助于调试机器学习程序。

#### 环境设置



`“conda list env”`命令列出了实例上可用的所有`conda`环境。此时我们将利用我们AMI附带的`“pytorch”`环境，使用以下命令在终端中激活预装环境：

```
source activate pytorch
```



我们使用“source activate”而不是“`conda activate`”，因为在实例首次连接时需要初始化`conda`。因此，以下命令产生相同的效果：

```
conda init
conda activate pytorch
```



#### 代码设置



我们将使用一些可以在此处找到的入门代码，在VS Code终端上将存储库克隆到实例上。

具体来说，我们将处理`main.py`文件，该文件训练一个模型。我们想比较在不同情况下运行一个epoch所需的时间。因此，第一步是修改代码以包含计时。我们将导入time，使用time.time()函数来监控每个epoch所需的时间，然后打印结果，这将在主函数内的循环中进行。

```
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    train(args, model, train_loader, optimizer, epoch)
    t_diff = time.time() - t0
    print(f"Elapsed time is {t_diff}")
    test_loss, test_acc = test(model, test_loader)
    scheduler.step()
```



#### 添加 `Wandb` 日志记录



正如我们在前几讲中所做的那样，我们还将`Weights and Biases`合并到代码库中，以保持良好的实践。`conda`环境中尚未包含该库，因此我们必须在终端中键入“`conda install wandb`”来安装它。

在`main.py`中，我们将导入`wandb`，使用配置中的训练和测试参数初始化`wandb`，然后记录相关信息。

```
wandb.init(config={"train_args": train_kwargs, "test_args": test_kwargs})
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    train(args, model, train_loader, optimizer, epoch)
    t_diff = time.time() - t0
    print(f"Elapsed time is {t_diff}")
    test_loss, test_acc = test(model, test_loader)
    scheduler.step()
    wandb.log({"test_loss": test_loss, "test_acc": test_acc, "time_taken": t_diff}, step=epoch)
```



我们还需要确保在循环结束时添加一个日志记录结束命令。

```
wandb.finish()
```

一旦添加了这个，我们可以运行带有wandb日志记录的代码。此时，你还可以在代码库的开头添加一个登录语句，或提前使用`wandb-CLI`进行登录。

#### GPU 调整



如果我们现在运行代码，训练时间会相当慢，在终端中运行 `nvidia-smi` 可以揭示一些潜在的问题，比如GPU资源利用率不足、内存耗尽或者其他进程正在竞争GPU资源等，通过分析 `nvidia-smi` 的输出，可以更好地优化你的训练过程，提高效率和性能。

尽管我们的实例拥有GPU，但我们仍然没有使用它们，代码本身必须设置为利用GPU。为此，我们将调整`main.py`文件中的现有入门代码。

附：如果我们在没有GPU的实例（如`t2.micro`实例类型）上运行`nvidia-smi`命令，它会抛出一个错误，第一件事是检查CUDA是否可用。

在解析器和参数创建并配置后，我们将在主函数中执行此检查。如果cuda可用，需要相应地定义设备，设置工作线程数和`shuffle`值，并更新训练和测试参数。

```
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
    cuda_kwargs = {'num_workers': 1, 'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
else:
    device = torch.device("cpu")
```



接下来，我们必须将模型和数据（包括训练和测试数据）加载到设备上。这可以通过“`.to(device)`”函数来完成。我们可以在定义模型时将其移动到设备上。

```
model = Net().to(device)
```

在处理训练和测试函数的循环时，可以将训练和测试数据移动到设备上。我们还需要更新函数以接收设备作为输入，并在主函数的循环中调用函数时包含设备。

```
def train(args, model, train_loader, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        ……
def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            ……
def main():
    ……
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train(args, model, train_loader, optimizer, epoch, device)
        t_diff = time.time() - t0
        print(f"Elapsed time is {t_diff}")
        test_loss, test_acc = test(model, test_loader, device)
        scheduler.step()
        wandb.log({"test_loss": test_loss, "test_acc": test_acc, "time_taken": t_diff}, step=epoch)
    ……
```



现在我们使用了GPU，代码运行速度比以前快得多，每个epoch应该大约需要10秒钟来运行。

#### 提高速度



此时，我们应该思考这个问题：如何使其更快？我们将讨论几种可能的想法。

**想法1：更多的GPU**

如果我们使用了所有的GPU容量，那么选择一个具有更多GPU的新实例类型可能是有益的。然而，这里并不是这种情况。

**想法2：增加批处理大小以使用更多的GPU**

也许我们应该尝试将批处理大小从默认的64增加到128。理由是我们只使用了约20%的GPU存储，因此我们可以在每步中使用更多的存储。我们可以通过运行以下命令来尝试这一点：

```
python main_working.py -batch-size=128
```



事实证明，这种变化可能不会对训练速度产生显著影响（可能对较小/较慢的GPU实例如P2s产生更大影响）。因为这里的模型非常小，瓶颈不在于模型计算，而在于内存瓶颈，这引出了我们的最后一个想法。

**想法3：更改工作线程数**

早些时候，当我们建立cuda参数时，我们将工作线程数设置为1，我们可以尝试增加这个数量。

如果我们将工作线程数设置得过高，如100可能无法工作。相反，我们应该尝试为此实例推荐的最佳数量，即16。

**增加工作线程数的理由是，更多的工作线程将帮助你更快地加载数据，即在GPU完成一个小批次的前向和后向传递后，工作线程已经准备好下一个批次，而不是在上一个批次完全完成后才开始加载数据。**

我们可以看到，这对数据加载器有影响，因为工作线程数作为参数传递给它。

事实证明，这种变化在缩小内存瓶颈和减少训练时间方面是有效的。此外，还可以通过多线程训练过程来优化数据加载。

### 跑一个实际的代码库



现在我们将过渡到处理一个代码库 [CDLab](./HowToRunCDLab)，我们的目标是让`run_train.py`文件成功运行。我们可以继续使用之前的实例，但我们将有一个新的环境代码库等。
现在我们将过渡到处理一个代码库CDLab，我们的目标是让run_train.py文件成功运行。我们可以继续使用之前的实例，但我们将有一个新的环境代码库等。

练习:

将这个代码库参考这个 [具体的教程](./HowToRunCDLab)（github上的附带文件）

在租的服务器上跑通训练和测试流程。

## 2、使用谷歌 Colab 进行深度学习

### 2.1 Colab 是什么

Colab = Colaboratory（即合作实验室），是谷歌提供的一个在线工作平台，用户可以直接通过浏览器执行python代码并与他人分享合作。Colab的主要功能当然不止于此，它还为我们提供免费的GPU。熟悉深度学习的同学们都知道：CPU计算力高但核数量少，善于处理线性序列，而GPU计算力低但核数量多，善于处理并行计算。在深度学习中使用GPU进行计算的速度要远快于CPU，因此有高算力的GPU是深度学习的重要保证。由于不是所有GPU都支持深度计算（大部分的Macbook自带的显卡都不支持），同时显卡配置的高低也决定了计算力的大小，因此Colab最大的优势在于我们可以“借用”谷歌免费提供的GPU来进行深度学习。

综上：Colab = "python版"Google doc + 免费GPU

有一些相关概念：

**Jupyter Notebook：** 在Colab中，python代码的执行是基于.ipynb文件，也就是Jupyter Notebook格式的python文件。这种笔记本文件与普通.py文件的区别是可以分块执行代码并立刻得到输出，同时也可以很方便地添加注释，这种互动式操作十分适合一些轻量的任务。

**代码执行程序：** 代码执行程序就是Colab在云端的"服务器"。简单来说，我们先在笔记本写好需要运行的代码，连接到代码执行程序，然后Colab会在云端执行代码，最后把结果传回浏览器。

**实例空间：** 连接到代码执行程序后，Colab需要为其分配实例空间(Instance)，可以简单理解为运行笔记本而创建的"虚拟机"，其中包含了执行ipynb文件时的默认配置、环境变量、自带的库等等。

**会话：** 当笔记本连接到代码执行程序并分配到实例空间后，就成为了一个会话(Session)，用户能开启的回话数量是有限的。



### 2.2 工作流程

**2.2.1 新建笔记本**

首先我们需要创建一个谷歌账户，申请谷歌账户需要能接受短信的手机号码。目前不能通过中国手机来创建账户，但是账号在创建后可以改绑中国手机。如何注册大家可以自己想办法。

Colab一般配合Google Drive使用（下文会提到这一点）。因此如有必要，我建议拓展谷歌云端硬盘的储存空间，个人认为性价比较高的是基本版或标准版。在购买完额外的空间后，头像外部会出现一个四色光环，就像作者一样。

![](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225012654.jpg)


直接在浏览器中输入 （https://colab.research.google.com），进入Colab的页面后点击新建笔记本即可。使用这种方法新建的笔记本时，会在云端硬盘的根目录自动创建一个叫Colab Notebook的文件夹，新创建的笔记本就保存在这个文件夹中。

![20250225012750](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225012750.jpg)

**2.2.2 载入笔记本**

可以打开云端硬盘中的已经存在的笔记本，还可以从Github中导入笔记本。如果关联了Github账户，可以选择一个账户中的Project，如果其中有ipynb文件就可以在Colab中打开。注意：关联Github不是把Github中的项目文件夹加载到实例空间！

![20250225012919](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225012919.jpg)

![20250225012937](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225012937.jpg)

**2.2.3 笔记本界面**

![20250225013144](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225013144.jpg)



**标题**：笔记本的名称

**代码块**：分块执行的代码

**文件浏览**：Colab为笔记本分配的实例空间

**代码执行程序**：用于执行笔记本程序的服务器

**代码段**：常用的代码段，比如装载云端硬盘

**命令面板**：常用的命令，比如查找/替换

**终端**：文件浏览下的终端（非常卡，不建议使用）

连接代码执行程序

点击连接按钮即可在5s左右的时间内连接到代码执行程序并分配实例空间，此时可以看到消耗的RAM和磁盘

**RAM**：虚拟机运行内存，更大内存意味着更大的算力（之后会在Colab Pro中介绍）

**磁盘**：虚拟机文件的储存空间，要注意的是购买GoogleDrive的存储空间不能增加实例空间内可用的磁盘空间

![](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225013225.jpg)

在打开笔记本后，我们默认的文件路径是*"/content"*，这个路径也是执行笔记本时的路径，同时我们一般把用到的各种文件也保存在这个路径下。在点击*".."*后即可返回查看根目录*"/"*（如下图），可以看到根目录中保存的是一些虚拟机的环境变量和预装的库等等。不要随意修改根目录中的内容，以避免运行出错，我们所有的操作都应在*"/content"*中进行。

![](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2025/20250225013301.jpg)

**2.2.4 执行代码块**

notebook文件通过的代码块来执行代码，同时支持通过"!<command>"的方式来执行UNIX终端命令（比如"!ls"可以查看当前目录下的文件）。Colab已经预装了大多数常见的深度学习库，比如pytorch，tensorflow等等，如果有需要额外安装的库可以通过"!pip3 install <package>"命令来安装。下面是一些常见的命令。

