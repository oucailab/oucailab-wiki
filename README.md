# oucailab-wiki



OUCAILAB，基于 VitePress 构建的现代化文档网站。


## 快速开始

### 环境要求

- [Node.js](https://nodejs.org/) >= 18
- [npm](https://www.npmjs.com/)

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/oucailab/oucailab-wiki.git
cd oucailab-wiki
```

2. 安装依赖：
```bash
npm install
```

### 开发与部署

开发模式：
```bash
npm run dev
```

构建生产版本：
```bash
npm run build
```

预览构建结果：
```bash
npm run preview
```

## 项目结构

```
ITStudio-Wiki/
├── .vitepress/          # VitePress 配置
│   ├── config.mts       # 站点配置文件
│   └── theme/           # 主题相关文件
├── src/                 # 文档源文件
│   ├── public/          # 静态资源文件
│   └── index.md        # 首页
└── package.json        # 项目配置文件
```

## 参与贡献

我们欢迎所有形式的贡献，无论是新功能、文档改进还是问题反馈。

1. Fork 本仓库
2. 创建你的特性分支：`git checkout -b feature/YourFeature`
3. 提交你的改动：`git commit -m 'Add some feature'`
4. 推送到分支：`git push origin feature/YourFeature`
5. 提交 Pull Request

## 文档编写指南

1. 所有文档都使用 Markdown 格式
2. 文档应放在适当的目录结构中
3. 图片等静态资源请放在 `src/public` 目录下
4. 新增文档需要在 `.vitepress/config.mts` 中添加相应的导航配置，如果是在已经创建的目录中新建文档，则无需修改导航配置。
5. 文档有且只有一个一级标题。

## 许可证

本项目采用 Apache2 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情