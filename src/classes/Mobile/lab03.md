## 实验3：高校新闻网

```
本实验来自于周文洁老师的《微信小程序开发实战》第十四章。
在学习了小程序的基础知识和各类 API 以后，尝试独立动手创建一个小程序前端综合设计实例。
我们将从零开始详解如何模仿网易新闻实现一个基于模拟数据的简易高校新闻小程序。

学习目标： 1、综合所学知识创建完整的前端新闻小程序项目；能够在开发过程中熟练掌握真机预览、调试等操作。

注意事项：
1、提供 common.js、图片文件、以及 index 页面的代码，其它部分的代码大家自己完成，下载地址：https://gaopursuit.oss-cn-beijing.aliyuncs.com/2022/demo4_file.zip
2、detail 页面的 wxss 文件中， .poster image 中的 width 设置为 100% 的话，图片总是无法显示。修改为 700rpx 就可以显示，我不清楚什么原因，大家如果知道为什么可以告诉我。
```

中国海洋大学新闻网的界面如下，提供最新新闻资讯和个性化的收藏功能。

![](https://gaopursuit.oss-cn-beijing.aliyuncs.com/img/2026/ScreenShot_2026-08-30_211045_064.jpg)

### 1、功能需求介绍

本项目一共需要3个页面，即首页、新闻页和个人中心页，其中首页和个人中心页需要以tabBar的形式展示，可以点击tab图标互相切换。

- 首页功能需求如下：

（1）首页需要包含幻灯片播放效果和新闻列表；

（2）幻灯片至少要有3幅图片自动播放；

（3）点击新闻列表可以打开新闻全文。

- 新闻页功能需求如下：

（1）阅读新闻全文的页面需要显示新闻标题、图片、正文和日期；

（2）允许点击按钮将当前阅读的新闻添加到本地收藏夹中；

（3）已经收藏过的新闻也可以点击按钮取消收藏。

- 个人中心页功能需求如下：

（1）未登录状态下显示登录按钮，用户点击以后可以显示微信头像和昵称。

（2）登录后读取当前用户的收藏夹，展示收藏的新闻列表。

（3）收藏夹中的新闻可以直接点击查看内容。

（4）未登录状态下收藏夹显示为空。

<br>

### 2、创建项目

- 新建项目，选择空白文件夹，创建小程序，选择不使用云服务，模板选择JS基础。
- 在app.json中的pages属性下添加pages/detail/detail和pages/my/my，保存后pages文件夹下自动生成detail页面和my页面。
- 删除 index.wxml和index.wxss全部代码。
- 删除index.js中全部代码，输入关键词page，找到Page选项回车自动补全函数。
- 删除app.wxss中全部代码。
- 删除app.js中全部代码，输入关键词app，找到App选项回车自动补全函数。
- 下载压缩包https://gaopursuit.oss-cn-beijing.aliyuncs.com/2022/demo4_file.zip
- 解压，将images和utils文件夹放在根目录下，分别用于存储图片素材和JS工具文件。

<br>

### 3、视图设计

#### 3.1 导航栏设计

在**app.json**中修改window属性配置导航栏效果，修改了导航栏的颜色，标题以及标题颜色。

```css
    "window": {
        "navigationBarBackgroundColor": "#328EEB",
        "navigationBarTitleText": "我的新闻网",
        "navigationBarTextStyle":"white"
    }
```
<br>

#### 3.2 tabBar设计

在**app.json**中启用`tabBar`，同时引用`images`文件夹中的图片素材，在`tabBar`可以点击`首页`和`我的`图标，切换至对应的页面，同时改变显示的图标。

```
 "tabBar": {
    "color": "#000000",
    "selectedColor": "#328EEB",
    "list": [
      {
        "pagePath": "pages/index/index",
        "text": "首页",
        "iconPath": "images/index1.png",
        "selectedIconPath": "images/index2.png"
      },
      {
        "pagePath": "pages/my/my",
        "text": "我的",
        "iconPath": "images/my1.png",
        "selectedIconPath": "images/my2.png"
      }
    ]
  }
```

<br>

### 3.3 首页设计

首页包含两部分内容，分别是幻灯片滚动和新闻列表，使用`<swiper>`组件和`<view>`容器。

1. 轮播组件 (swiper):
```
- 使用 `<swiper>` 标签创建一个轮播组件。
- `indicator-dots="true"` 表示显示轮播指示点。
- `autoplay="true"` 表示自动播放。
- `interval="5000"` 设置自动播放的间隔时间为 5000 毫秒（5 秒）。
- `duration="500"` 设置滑动动画的持续时间为 500 毫秒。
- 使用 `<view wx:for="{{swiperImg}}" wx:key="swiper{{index}}">` 循环遍历 `swiperImg` 数组，生成多个 `<swiper-item>`。
- 每个 `<swiper-item>` 包含一个 `<image>` 标签，用于显示图片。
```
<br>

2. 新闻列表 (news-list):
```
- 使用 `<view class="news-list">` 创建一个新闻列表的容器。
- 使用 `<view class="news-item" wx:for="{{newsList}}" wx:key="{{item.id}}">` 循环遍历 `newsList` 数组，生成多个新闻项。
- 每个新闻项包含一个 `<image>` 标签，用于显示新闻的海报图片。
- 每个新闻项还包含一个 `<text>` 标签，显示新闻标题和添加日期，并绑定点击事件 `bindtap='goToDetail'`，点击时会调用 `goToDetail` 方法，并传递新闻项的 `id`。
```
<br>

注意，`swiperImg` 和 `newsList` 是数据源，应该在页面的 JavaScript 文件中定义和赋值。

**index.wxml**文件代码： 

```
<!-- 幻灯片 -->
<swiper indicator-dots="true" autoplay="true" interval="5000" duration="500">
  <view wx:for="{{swiperImg}}" wx:key="swiper{{index}}">
    <swiper-item>
      <image src="{{item.src}}"></image>
    </swiper-item>
  </view>
</swiper>
<!-- 新闻列表 -->
<view class="news-list">
  <view class="news-item" wx:for="{{newsList}}" wx:key="{{item.id}}" >
    <image src="{{item.poster}}" ></image>
    <text bindtap = 'goToDetail' data-id="{{item.id}}">{{item.title}}————{{item.add_date}}</text>
  </view>
</view>
```



     **index.wxss**文件代码：

```css
/* 幻灯片部分 */
swiper{
  height: 400rpx;
  width:100%;
}
swiper image{
  height: 100%;
  width:100%;
}
```

后续在个人中心页也会用到新闻列表，所以将这部分样式写在公共样式表**app.wxss**中重复利用，作为全局样式。

**app.wxss**文件代码：

```css
/* 新闻列表 */
.news-list{
  min-height: 600rpx;
  padding: 15rpx;
}
.news-item{
  display: flex;
  flex-direction: row;
  border-bottom:1rpx solid black;
}
.news-item image{
  height: 150rpx;
  width: 230rpx;
  margin: 10rpx;
}
.news-item text{
  width:100%;
  line-height: 60rpx;
  font-size:40rpx;
}
```
<br>

### 3.4 个人中心页设计

个人中心页包含两部分，登录页面和收藏列表，使用<view>容器。



这段代码是一个微信小程序的页面布局，包含登录页面和收藏列表。以下是详细描述：

1. **登录页面 (myLogin)**:

   - 使用 `<view class="myLogin">` 创建一个登录页面的容器。

   - 使用 `<block wx:if="{{isLogin}}">`判断用户是否已登录。
   - 如果已登录，显示用户头像和昵称。
   - 如果未登录，显示登录按钮，并绑定点击事件 `bindtap="getUserInfo"`，点击时会调用 `getUserInfo` 方法。

2. **收藏列表 (myFavorite)**:

   - 使用 `<view class="myFavorite">` 创建一个收藏列表的容器。

   - `<text>我的收藏（{{number}}）</text>` 显示收藏的数量。
   - 使用` <view class="news-list">`创建一个新闻列表的容器。
   - 使用 for循环`newsList` 数组，生成多个新闻项。
   - 每个新闻项包含一个 `<image>` 标签，用于显示新闻的海报图片。
   - 每个新闻项还包含一个 `<text>` 标签，显示新闻标题和添加日期，并绑定点击事件 `bindtap='goToDetail'`，点击时会调用 `goToDetail` 方法，并传递新闻项的 `id`。

注意，`isLogin`、`src`、`nickName`、`number` 和 `newsList` 是数据源，应该在页面的 JavaScript 文件中定义和赋值。                        

**my.wxml**文件代码：

```html
<!-- 登陆页面 -->
<view class="myLogin">
  <block wx:if="{{isLogin}}">
    <image src="{{src}}"></image>
    <text>{{nickName}}</text>
  </block>
  <button wx:else bindtap="getUserInfo" >未登录，点此登录</button>
</view>
<!-- 收藏列表 -->
<view class="myFavorite"> 
  <text>我的收藏（{{number}}）</text>
  <view class="news-list">
    <view class="news-item" wx:for="{{newsList}}" wx:key="{{item.id}}">
      <image src="{{item.poster}}"></image>
      <text bindtap = 'goToDetail' data-id="{{item.id}}">{{item.title}}————{{item.add_date}}</text>
    </view>
  </view>
</view>
```

**my.wxss**文件代码：

```css
/* 登陆页面 */
.myLogin{
  height: 400rpx;
  background-color: #328EEB;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-around;
}
.myLogin image{
  height: 200rpx;
  width:200rpx;
  border-radius: 50%;
}
.myLogin text{
  color: white;
}
/* 收藏列表 */
.myFavorite{
  padding: 20rpx;
}
```
<br>


