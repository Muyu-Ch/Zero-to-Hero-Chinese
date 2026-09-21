# Zero to Hero 中文注解 — Andrej Karpathy 教程伴读仓库

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 从零手写并理解神经网络与语言模型。
> 
> 全程跟随 [Andrej Karpathy 的 Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 系列,代码逐行配上中文注释与自己的思考,希望为中文学习者提供一个可以对照教程轻松读懂的伴读仓库。

## ✨ 这个仓库有什么

- **逐行中文注释**:不只是翻译"这行代码做了什么",更记录"为什么这样写"——包括踩过的坑(梯度不清零、广播机制、随机种子与 `generator`、tanh 饱和、BatchNorm 训练/推理不一致等)
- **实验驱动**:每个关键结论都尽量用具体数字和可视化验证,而不是"书上这么说"——Part 3 的笔记本里画了几十张训练曲线与激活/梯度分布图,就是为了亲眼看见"tanh 饱和""梯度消失"长什么样
- **与教程一一对应**:目录按 Zero to Hero 的课程顺序组织,方便边看视频边对照

每个课程目录里都放了三样东西,基本可以离线自给自足:

| 文件              | 用途                                                    |
| --------------- | ----------------------------------------------------- |
| `*.ipynb`       | **主看这个**:逐行中文注解 + 训练曲线与分布图,可以逐步运行、随手改                  |
| `*.py`          | 把笔记本整理成一份能直接跑的脚本,`python xxx.py` 就能从训练跑到生成名字           |
| `names.txt`     | 课程用的名字语料(32033 个),每个目录一份,互不依赖                        |

## 📁 目录与进度

| 目录                                              | 内容                                                                                                                                          | 状态  |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| [`micrograd/`](micrograd/micrograd.ipynb)       | 从零实现反向传播引擎(micrograd)与小型 MLP:自己写 `Value` 类,把计算图、拓扑排序、链式法则一路手动接起来。两个演示——二进制预测,以及自己设计的"让模型学出 `a/b`"                              | ✅   |
| [`makemore_1/`](makemore_1/makemore_1.ipynb)    | **Part 1** bigram 语言模型:计数版(MLE 闭式解)与神经网络版(one-hot → `W` → softmax),负对数似然损失、`multinomial` 采样、L2 正则化。结尾证明梯度下降能把"对数计数表"重新学出来                     | ✅   |
| [`makemore_2/`](makemore_2/makemore_2.ipynb)    | **Part 2** 多层感知机 (MLP):引入嵌入表、隐藏层与非线性激活(`tanh`),从"只看 1 个字符"升级为"看 3 个字符";超参数旋钮炼丹与学习率搜索。附 [Bengio 2003 论文原文](makemore_2/MLP.pdf) | ✅   |
| [`makemore_3/`](makemore_3/makemore_3.ipynb)    | **Part 3** 激活值 / 梯度 / BatchNorm:手写 `Linear` / `BatchNorm1d` / `Tanh` 并留下每层 `self.out` 做"体检";何恺明初始化;用 BatchNorm 稳定深网络                                                   | ✅   |
| [`demo.ipynb`](demo.ipynb) / [`demo.py`](demo.py) | 开胃小菜:用 C++ 的 `vector<vector<...>>` 类比 PyTorch 张量,顺一遍 shape / view / 广播这些天天要打交道的东西                                                              | ✅   |

> ✅ 已完成 ｜ 🚧 学习中 ｜ ⬜ 未开始

### Part 3 具体做了什么

Part 2 的 MLP 能跑通,但跑得"心里没底":激活值往 tanh 两端跑(饱和)、梯度越传越小、每加一层就得重新调初始化。Part 3 把这些不安全感一个个拆开:

1. **看得见里面**:手写三个层类,每层都把输出存进 `self.out` 并全程 `retain_grad()`,于是激活值分布、梯度分布都能单独画出来体检(笔记本里那些直方图)
2. **何恺明初始化**:`W ~ N(0, 1/fan_in)`,让前向传播中每层输出的方差保持不变;tanh 前面再乘 `5/3` 的 gain 补偿它"压缩"的脾气
3. **BatchNorm**:把每层输出强行拉回均值 0 / 方差 1,训练从此不太看初始化脸色,可以放心往下叠到 5 层
4. **最大的坑**:BatchNorm 训练时用当前 batch 的统计量,推理时必须改用滑动平均,两个分支靠 `self.training` 开关切换。生成名字前忘了关,`batch=1` 会让 `torch.var` 除以 `N-1=0` 算出 `nan`,顺着 softmax 传给 `torch.multinomial` 直接报错——更阴险的是 `nan` 会被写进 `running_var` 永久污染

最终 5 层 MLP + BatchNorm,验证集 loss 从 Part 2 的约 2.19 降到约 2.09。

## 🧭 学习路线(Zero to Hero)

1. **micrograd** ✅ — 手写自动求导引擎,理解反向传播的本质([视频](https://www.youtube.com/watch?v=VMj-3S1tku0) / [代码](https://github.com/karpathy/micrograd))
2. **makemore Part 1** ✅ — bigram 语言模型:计数版 + 神经网络版,负对数似然损失与采样([视频](https://www.youtube.com/watch?v=PaCmpygFfXo) / [代码](https://github.com/karpathy/makemore))
3. **makemore Part 2** ✅ — 多层感知机 (MLP):嵌入、隐藏层、`tanh` 与学习率搜索([视频](https://www.youtube.com/watch?v=TCH_1BHY58) / [代码](https://github.com/karpathy/makemore))
4. **makemore Part 3** ✅ — Activations & Gradients & BatchNorm:激活诊断、死亡神经元、初始化与批归一化([视频](https://www.youtube.com/watch?v=P6sfmUTpUmc) / [代码](https://github.com/karpathy/makemore))
5. **makemore Part 4** ⬜ — Becoming a Backprop Ninja:扔掉 autograd,手动反向传播([视频](https://www.youtube.com/watch?v=q8SA3rM6ckI) / [代码](https://github.com/karpathy/makemore))
6. **makemore Part 5** ⬜ — Building a WaveNet:把上下文从 3 个字符扩到 8 个,用层级结构(膨胀卷积)吃更长历史([视频](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) / [代码](https://github.com/karpathy/makemore))
7. **Let's build GPT** ⬜ — 从零手写 Transformer,拼出第一个 GPT([视频](https://www.youtube.com/watch?v=kCc8FmEb1nY) / [代码](https://github.com/karpathy/ng-video-lecture))
8. **Let's build the GPT Tokenizer** ⬜ — 拆开分词器,手写 BPE([视频](https://www.youtube.com/watch?v=zduSFxRajkE) / [代码](https://github.com/karpathy/minbpe))
9. **Let's reproduce GPT-2 (124M)** ⬜ — 完整复现 GPT-2([视频](https://www.youtube.com/watch?v=l8pRSuU81PU) / [代码](https://github.com/karpathy/build-nanogpt))

> 根目录的 `input.txt` 是 `tinyshakespeare` 语料(40000 行),给后面 GPT 部分备用。

## 🚀 快速开始

### 环境

- 个人使用m芯片macbook进行学习，但全程不涉及复杂环境，可使用任意系统电脑；以下环境安装部分适用于macos与linux环境。windows用户们请自行配置环境

- 仓库自带 `venv/` 虚拟环境(Python 3.14.6 + PyTorch 2.13.0 + NumPy 2.5.2 + Matplotlib 3.11.1 + ipykernel 7.3.0):

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

- 每个课程目录下的 `.py` 可以直接跑,例如:

```bash
python makemore_3/makemore_3.py
```

- `.ipynb` 建议使用Jupyter Lab打开:

```bash
pip install jupyterlab
jupyter lab
```

- 想先试试水和张量打交道的手感,可以跑根目录的 `demo.py`:

```bash
python demo.py
```

## 📚 致谢与说明

- 代码跟写自 Andrej Karpathy 的公开教程与仓库(MIT License),中文注释与整理为个人学习记录
- 本人是初学者,注释里可能有理解不到位的地方,欢迎提 [issue](https://github.com/Muyu-Ch/zero-to-hero-zh/issues) 或 PR 指正

## 📄 License

[MIT](LICENSE)
