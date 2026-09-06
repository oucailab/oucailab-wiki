## 实验5：高校新闻网

> 本实验来自于
>   
>  **截止时间：9月1日（星期二） 22:00** 
>
> 大家务必注意时间，超出时间要扣分！

中国海洋大学新闻网的界面如下，提供最新新闻资讯和个性化的收藏功能。



### 一、实验介绍

本实验主要介绍的ArkTS程序编译后在鸿蒙系统安装运行。通过本实验，您将能够掌握在ArkTS程序的编译，熟悉在鸿蒙系统的安装和运行的查看。本实验需要用到一台安装有Windows10 64位或Windows11 64位的主机，要求内存为16GB及以上，推荐为32GB，硬盘为100GB及以上，分辨率：1280*800像素及以上。



### 2、开发环境搭建

#### 2.1 安装DevEco Studio

- 步骤 1  进入下载页面https://developer.huawei.com/consumer/cn/download/，选择最新版本下载。
- 步骤 2  参考页面进行安装：https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V5/ide-software-install-V5

<br>

#### 2.2 创建模拟器

整体步骤参考链接：https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V5/ide-emulator-create-V5

步骤1：点击菜单栏的Tools > Device Manager，点击右下角的Edit设置模拟器实例的存储路径Local Emulator Location，Mac默认存储在~/.Huawei/Emulator/deployed下，Windows默认存储在C:\Users\xxx\AppData\Local\Huawei\Emulator\deployed下。

![ScreenShot_2026-09-06_233252_769](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_233252_769.jpg)

步骤2：   在Local Emulator页签中，单击右下角的New Emulator按钮，创建一个模拟器。在模拟器配置界面，可以选择一个默认的设备模板，首次使用时会提示“Download the system image first”，请点击设备右侧的下载模拟器镜像，您也可以在该界面更新或删除不同设备的模拟器镜像。单击Edit可以设置镜像文件的存储路径。Mac默认存储在~/Library/Huawei/Sdk下，Windows默认存储在C:\Users\xxx\AppData\Local\Huawei\Sdk下。

![ScreenShot_2026-09-06_233355_893](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_233355_893.jpg)

步骤3：单击Next，核实确定需要创建的模拟器的名称，内存和存储空间，然后单击Finish创建模拟器。

![ScreenShot_2026-09-06_233441_442](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_233441_442.jpg)

步骤4：在设备管理页面，启动模拟器。

![ScreenShot_2026-09-06_233526_834](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_233526_834.jpg)

步骤5：单击DevEco Studio的Run > Run'模块名称'

![ScreenShot_2026-09-06_233558_064](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_233558_064.jpg)

步骤6：DevEco Studio会启动应用/服务的编译构建与推包，完成后应用/服务即可运行在模拟器上。

![ScreenShot_2026-09-06_233636_523](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-09-06_233636_523.jpg)

至此，完成了DevEcoStudio及模拟器的安装。


### 五、实验总结

期待看到大家完成海大主题的新闻网，期待大家发挥创意，做出带有个人想法与风格的作品。




