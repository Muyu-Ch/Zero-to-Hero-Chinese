# Zero to Hero 中文注解 — Andrej Karpathy 教程伴读仓库

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 从零手写并理解神经网络与语言模型。
> 
> 全程跟随 [Andrej Karpathy 的 Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 系列,代码逐行配上中文注释与自己的思考,希望为中文学习者提供一个可以对照教程轻松读懂的伴读仓库。

## ✨ 这个仓库有什么

- **逐行中文注释**:不只是翻译"这行代码做了什么",更记录"为什么这样写"——包括踩过的坑(梯度不清零、广播机制、随机种子与 `generator`、tanh 饱和等)
- **实验驱动**:每个关键结论都尽量用具体数字和可视化验证,而不是"书上这么说"
- **与教程一一对应**:目录按 Zero to Hero 的课程顺序组织,方便边看视频边对照

## 📁 目录与进度

| 文件                                                       | 内容                                                                 | 状态  |
| -------------------------------------------------------- | ------------------------------------------------------------------ | --- |
| [`micrograd/micrograd.ipynb`](micrograd/micrograd.ipynb) | 从零实现反向传播引擎(micrograd)与小型 MLP                                       | ✅   |
| [`makemore/Makemore.ipynb`](makemore/Makemore.ipynb)     | 名字生成模型 Part 1:bigram 计数模型(MLE 闭式解)、负对数似然损失、`multinomial` 采样、神经网络版(one-hot → `W` → softmax,梯度下降学出对数计数)、L2 正则化 | ✅   |
| makemore Part 2: 多层感知机 (MLP)                          | 引入隐藏层与非线性激活(`tanh`)、BatchNorm,从 bigram 升级为真正的神经网络           | 🚧  |

> ✅ 已完成 ｜ 🚧 学习中 ｜ ⬜ 未开始

## 🧭 学习路线(Zero to Hero)

1. **micrograd** ✅ — 手写自动求导引擎,理解反向传播的本质([视频](https://www.youtube.com/watch?v=VMj-3S1tku0) / [代码](https://github.com/karpathy/micrograd))
2. **makemore Part 1** ✅ — bigram 语言模型:计数版 + 神经网络版,负对数似然损失与采样([视频](https://www.youtube.com/watch?v=PaCmpygFfXo) / [代码](https://github.com/karpathy/makemore))
3. **makemore Part 2** 🚧 建设中 — 多层感知机 (MLP):嵌入、隐藏层、`tanh` 与 BatchNorm([视频](https://www.youtube.com/watch?v=TCH_1BHY58) / [代码](https://github.com/karpathy/makemore))

## 🚀 快速开始

### 环境

- 个人使用m芯片macbook进行学习，但全程不涉及复杂环境，可使用任意系统电脑；一下环境安装部分适用于macos与linux环境。windows用户们请自行配置环境

- 仓库自带 `venv/` 虚拟环境(Python 3.14.6 + PyTorch 2.13.0 + NumPy 2.5.2 + Matplotlib 3.11.1):

```bash
source venv/bin/activate
```

- 其他机器可按 [`requirements.txt`](requirements.txt) 一键复现同样的环境:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 运行

- `demo.py` 可直接运行:

```bash
python demo.py
```

- `.ipynb` 建议使用Jupyter Lab打开:

```bash
pip install jupyterlab
jupyter lab
```

## 📚 致谢与说明

- 代码跟写自 Andrej Karpathy 的公开教程与仓库(MIT License),中文注释与整理为个人学习记录
- 本人是初学者,注释里可能有理解不到位的地方,欢迎提 [issue](https://github.com/Muyu-Ch/zero-to-hero-zh/issues) 或 PR 指正

## 📄 License

[MIT](LICENSE)
