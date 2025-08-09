import { defineConfig } from "vitepress";
import fs from "fs";
import path from "path";

const BASE_PATH = path.join(__dirname, "../src");

function getMarkdownFiles(dirPath: string): { text: string; link: string }[] {
  if (!fs.existsSync(dirPath)) return [];
  return fs
    .readdirSync(dirPath)
    .filter(
      (file) =>
        file.endsWith(".md") &&
        !file.startsWith(".") &&
        !file.startsWith("_") &&
        file !== "index.md"
    )
    .sort()
    .map((file) => ({
      text:
        file.replace(/\.md$/, "").charAt(0).toUpperCase() +
        file.replace(/\.md$/, "").slice(1),
      link:
        "/" +
        path
          .relative(BASE_PATH, path.join(dirPath, file))
          .replace(/\.md$/, "")
          .replace(/\\/g, "/"),
    }));
}

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "OUC AI Lab",
  description: "MAKE OUCAILAB GREAT AGAIN",
  lang: "zh-CN",
  srcDir: "src",
  lastUpdated: true,
  ignoreDeadLinks: true,
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config

    logo: "/logo.png",
    siteTitle: "OUCAILAB",

    // Navbar
    nav: [{ text: "Home", link: "/" }],

    // Sidebar
    sidebarMenuLabel: "菜单",
    sidebar: [
      {
        text: "Welcome",
        items: [
          { text: "OUC AI Lab", link: "/index" },
          { text: "Junyu Dong", link: "/dongjy" },
          { text: "Feng Gao", link: "/fenggao" },
          { text: "Lin Qi", link: "/linqi" },
          { text: "Yanhai Gan", link: "/ganyh" },
        ],
      },
      {  text: "📚 Publications", link: "/resources/papers"},
    ],

    // Social links
    socialLinks: [
      { icon: "github", link: "https://github.com/oucailab" },
    ],

    // Footer
    footer: {
      copyright: "Copyright © 2025  OUC AI Lab",
    },

    // Last updated
    lastUpdated: {
      text: "上次更新",
    },

    // Enable search
    search: {
      provider: "local",
      options: {
        translations: {
          button: {
            buttonText: "搜索文档",
            buttonAriaLabel: "搜索文档",
          },
          modal: {
            noResultsText: "无法找到相关结果",
            resetButtonTitle: "清除查询条件",
            footer: {
              selectText: "选择",
              navigateText: "切换",
              closeText: "关闭",
            },
          },
        },
      },
    },

    // Edit link
    editLink: {
      pattern: "https://github.com/oucailab/oucailab-wiki/edit/main/src/:path",
      text: "在 GitHub 上编辑此页",
    },

    // Doc footer
    docFooter: {
      prev: "上一页",
      next: "下一页",
    },

    // Light and dark mode
    darkModeSwitchLabel: "切换主题",

    lightModeSwitchTitle: "切换到浅色主题",
    darkModeSwitchTitle: "切换到深色主题",

    // Return to top
    returnToTopLabel: "返回顶部",

    // Show external link icon in markdown links(only external)
    externalLinkIcon: true,

    // Outline
    outline: {
      level: [2, 3],
      label: "目录",
    },
  },
});
