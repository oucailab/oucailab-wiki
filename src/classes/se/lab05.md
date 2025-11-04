# 实验5：ViT & Swin Transformer

> 学习要求：
>
> - Transformer的基本原理
> - 典型基于Transformer的网络结构
>   
> **截止时间：11月11日（星期二） 22:00** 
>
> 大家务必注意时间，超出时间要扣分！

<br>

## 1、视频学习

### 1.1 Vision Transformer (ViT) 

- ViT 网络详解，包括 Embedding, Transformer Encoder, MLP head 【[视频链接](https://www.bilibili.com/video/BV1Jh411Y7WQ)】
- 使用 pytorch 实现 ViT 【[视频链接](https://www.bilibili.com/video/BV1AL411W7dT)】

### 1.2 Swin Transformer 

- Swin Transformer网络详解【[视频链接](https://www.bilibili.com/video/BV1pL4y1v7jC)】
- 使用 pytorch 实现 Swin Transformer，了解即可，不需要掌握【[视频链接](https://www.bilibili.com/video/BV1yg411K7Yc)】

### 1.3 视觉Transformer综述

华为韩凯的综述，内容非常好，了解即可【[视频链接](https://www.bilibili.com/video/BV1cr4y1m7tS)】

## 2、思考题

- 在ViT中要降低 Attention的计算量，有哪些方法？（提示：Swin的 Window attention，PVT的attention）
- Swin体现了一种什么思路？对后来工作有哪些启发？（提示：先局部再整体）
- 有些网络将CNN和Transformer结合，为什么一般把 CNN block放在面前，Transformer block放在后面？
- 阅读并了解Restormer，思考：Transformer的基本结构为 attention+ FFN，这个工作分别做了哪些改进？


## 3、评分规则

本次实验满分10分，按时提交并且内容符合要求8分，发布在个人博客并提供链接加0.5分，使用markdown加0.5分，内容质量高加0.5到1分。错过时间提交，扣1分。
