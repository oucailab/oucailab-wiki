## 实验5：高校新闻网

> 本实验来自于
>   
>  **截止时间：9月1日（星期二） 22:00** 
>
> 大家务必注意时间，超出时间要扣分！

中国海洋大学新闻网的界面如下，提供最新新闻资讯和个性化的收藏功能。



### 一、实验介绍

本实验主要介绍的ArkTS程序编译后在鸿蒙系统安装运行。通过本实验，您将能够掌握在ArkTS程序的编译，熟悉在鸿蒙系统的安装和运行的查看。本实验需要用到一台安装有Windows10 64位或Windows11 64位的主机，要求内存为16GB及以上，推荐为32GB，硬盘为100GB及以上，分辨率：1280*800像素及以上。



### 二、开发环境搭建

#### 2.1 安装DevEco Studio

- 步骤 1  进入下载页面https://developer.huawei.com/consumer/cn/download/，选择最新版本下载。
- 步骤 2  参考页面进行安装：https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V5/ide-software-install-V5

<br>

#### 2.2 创建模拟器

整体步骤参考链接：https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V5/ide-emulator-create-V5

步骤1：点击菜单栏的Tools > Device Manager，点击右下角的Edit设置模拟器实例的存储路径Local Emulator Location，Mac默认存储在~/.Huawei/Emulator/deployed下，Windows默认存储在C:\Users\xxx\AppData\Local\Huawei\Emulator\deployed下。

<p align=center><img src=https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_233252_769.jpg width=600px></p>

步骤2：   在Local Emulator页签中，单击右下角的New Emulator按钮，创建一个模拟器。在模拟器配置界面，可以选择一个默认的设备模板，首次使用时会提示“Download the system image first”，请点击设备右侧的下载模拟器镜像，您也可以在该界面更新或删除不同设备的模拟器镜像。单击Edit可以设置镜像文件的存储路径。Mac默认存储在~/Library/Huawei/Sdk下，Windows默认存储在C:\Users\xxx\AppData\Local\Huawei\Sdk下。

<p align=center><img src=https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_233355_893.jpg width=600px></p>

步骤3：单击Next，核实确定需要创建的模拟器的名称，内存和存储空间，然后单击Finish创建模拟器。

<p align=center><img src=https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_233441_442.jpg width=600px></p>

步骤4：在设备管理页面，启动模拟器。

<p align=center><img src=https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_233526_834.jpg width=600px></p>

步骤5：单击DevEco Studio的Run > Run'模块名称'

<p align=center><img src=https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_233558_064.jpg width=600px></p>

步骤6：DevEco Studio会启动应用/服务的编译构建与推包，完成后应用/服务即可运行在模拟器上。

<p align=center><img src=https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_233636_523.jpg width=300px></p>

至此，完成了DevEcoStudio及模拟器的安装。

#### 2.3 开发及运行环境验收

DevEco Studio安装完成并创建模拟器后，可以通过运行Hello World工程来验证环境设置是否正确。接下来以创建一个支持Phone设备的工程为例进行介绍。

- 步骤 1   打开DevEco Studio，在欢迎页单击Create Project，创建一个新工程。
- 步骤 2   根据工程创建向导，选择创建Application或Atomic Service。选择Empty Ability模板，然后单击Next。

<p align=center><img src=https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_234100_996.jpg width=600px></p>

- 步骤 3  填写工程相关信息，单击Finish。关于各个参数的详细介绍，请参考创建一个新的工程。

<p align=center><img src=https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_234111_182.jpg width=600px></p>

**Project name**：工程的名称，可以自定义，由大小写字母、数字和下划线组成。

**Bundle name**：标识应用的包名，用于标识应用的唯一性。

**说明**

应用包名要求：1. 必须为以点号（.）分隔的字符串，且至少包含三段，每段中仅允许使用英文字母、数字、下划线（_），如“com.example.myapplication ”。2. 首段以英文字母开头，非首段以数字或英文字母开头，每一段以数字或者英文字母结尾，如“com.01example.myapplication”。3. 不允许多个点号（.）连续出现，如“com.example..myapplication ”。4. 长度为7~128个字符。

**Save location**：工程文件本地存储路径，由大小写字母、数字和下划线等组成，不能包含中文字符。

**Compatible SDK**：兼容的最低API Version。

**Module name**： 模块的名称。

**Device type****：**该工程模板支持的设备类型。

注意，工程创建完成后，DevEco Studio会自动进行工程的同步。

- 步骤 4  单击DevEco Studio的Run > Run'模块名称'。

<p align=center><img src=https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_234330_082.jpg width=600px></p>

- 步骤5 DevEco Studio会启动应用/服务的编译构建与推包，完成后应用/服务即可运行在模拟器上。

<p align=center><img src=https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_234406_965.jpg width=300px></p>


### 三、实现简易计算器

教程链接：https://developer.huawei.com/consumer/cn/doc/architecture-guides/calculator-0000002298744774

按照步骤实现简易计算器，并进行个性化创新开发。


### 四、实验总结

期待大家发挥创意，做出带有个人想法与风格的计算器作品。




