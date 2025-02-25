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



