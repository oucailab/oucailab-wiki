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
    sidebarMenuLabel: "Menu",
    sidebar: [
      { text: "Welcome", link: "/index"},
      { text: "🧑‍💻 小组成员", link: "/member"},
      { text: "📚 论文发表", link: "/resources/papers"},
      { text: "🏆 荣誉奖励", link: "/awards"},
      { 
        text: "🎁 课程学习",  
        items: [  
          {text:"25夏移动软件开发", link: "/classes/MobileDev"},
          {text:"25秋计算机学术英语", link:"/classes/AcademicEnglish"},
          {text:"25秋软件工程原理与实践", link:"/classes/Software"},
        ]
      },
      { 
        text: "❇️ 其他资料",  
        items: [  
          {text:"如何动手科研", link: "/classes/paperwriting"},
          {text:"科研作图指导", link:"/research/picture"},
          {text:"准备开题报告", link:"/research/proposal"},
          {text:"学位论文写作指导", link:"/research/thesis_writing"},
        ]
      },
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
      text: "Last updated",
    },

    // Enable search
    search: {
      provider: "local",
      options: {
        translations: {
          button: {
            buttonText: "Search",
            buttonAriaLabel: "Search",
          },
          modal: {
            noResultsText: "Can not find",
            resetButtonTitle: "Clean search",
            footer: {
              selectText: "Select",
              navigateText: "Change",
              closeText: "Close",
            },
          },
        },
      },
    },

    // Edit link
    editLink: {
      pattern: "https://github.com/oucailab/oucailab-wiki/edit/main/src/:path",
      text: "Edit this page in GitHub",
    },

    // Doc footer
    docFooter: {
      prev: "Previous page",
      next: "Next page",
    },

    // Light and dark mode
    darkModeSwitchLabel: "Change theme",

    lightModeSwitchTitle: "Change to light mode",
    darkModeSwitchTitle: "Change to dark mode",

    // Return to top
    returnToTopLabel: "Return to Top",

    // Show external link icon in markdown links(only external)
    externalLinkIcon: true,

    // Outline
    outline: {
      level: [2, 3],
      label: "Menu",
    },
  },
});
