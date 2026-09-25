# Zero to Hero 中文注解 — Andrej Karpathy 教程伴读仓库

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 从零手写并理解神经网络与语言模型。
> 
> 全程跟随 [Andrej Karpathy 的 Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 系列,代码逐行配上中文注释与自己的思考,希望为中文学习者提供一个可以对照教程轻松读懂的伴读仓库。

## ✨ 这个仓库有什么

- **逐行中文注释**:不只是翻译"这行代码做了什么",更记录"为什么这样写"——包括踩过的坑(梯度不清零、广播机制、随机种子与 `generator`、tanh 饱和、BatchNorm 训练/推理不一致等)
- **实验驱动**:每个关键结论都尽量用具体数字和可视化验证,而不是"书上这么说"——Part 3 的笔记本里画了几十张训练曲线与激活/梯度分布图,就是为了亲眼看见"tanh 饱和""梯度消失"长什么样
- **与教程一一对应**:目录按 Zero to Hero 的课程顺序组织,方便边看视频边对照

每门课一个目录(`makemore/` 下每个 Part 一个子目录),每个 Part 目录里放两样东西,再加一份五个 Part 共用的语料:

| 文件              | 用途                                                    |
| --------------- | ----------------------------------------------------- |
| `*.ipynb`       | **主看这个**:逐行中文注解 + 训练曲线与分布图,可以逐步运行、随手改                  |
| `*.py`          | 把笔记本整理成一份能直接跑的脚本,`python xxx.py` 就能从训练跑到生成名字           |
| `makemore/names.txt` | 课程用的名字语料(32033 个),五个 Part 共用这一份,脚本各自都能找到它              |

## 📁 目录与进度

| 目录                                              | 内容                                                                                                                                          | 状态  |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| [`micrograd/`](micrograd/micrograd.ipynb)       | 从零实现反向传播引擎(micrograd)与小型 MLP:自己写 `Value` 类,把计算图、拓扑排序、链式法则一路手动接起来。两个演示——二进制预测,以及自己设计的"让模型学出 `a/b`"                              | ✅   |
| [`makemore/makemore_1/`](makemore/makemore_1/makemore_1.ipynb)    | **Part 1** bigram 语言模型:计数版(MLE 闭式解)与神经网络版(one-hot → `W` → softmax),负对数似然损失、`multinomial` 采样、L2 正则化。结尾证明梯度下降能把"对数计数表"重新学出来                     | ✅   |
| [`makemore/makemore_2/`](makemore/makemore_2/makemore_2.ipynb)    | **Part 2** 多层感知机 (MLP):引入嵌入表、隐藏层与非线性激活(`tanh`),从"只看 1 个字符"升级为"看 3 个字符";超参数旋钮炼丹与学习率搜索。附 [Bengio 2003 论文原文](makemore/makemore_2/MLP.pdf) | ✅   |
| [`makemore/makemore_3/`](makemore/makemore_3/makemore_3.ipynb)    | **Part 3** 激活值 / 梯度 / BatchNorm:手写 `Linear` / `BatchNorm1d` / `Tanh` 并留下每层 `self.out` 做"体检";何恺明初始化;用 BatchNorm 稳定深网络                                                   | ✅   |
| [`makemore/makemore_4/`](makemore/makemore_4/makemore_4.ipynb) | **Part 4** 手动反向传播 (Backprop Ninja):删掉 `loss.backward()`,从 `cross_entropy` 一路手推 `dlogits` → `dW2/db2` → tanh → BatchNorm(全篇最难的一步) → `dW1/db1`,最后把 `dC` 用 scatter-add 填回嵌入表;每步都拿 `cmp()` 跟 autograd 对拍 | ✅   |
| [`makemore/makemore_5/`](makemore/makemore_5/makemore_5.ipynb) | **Part 5** 分层网络 (WaveNet 式):上下文拉到 8 个字符,用 `FlattenConsecutive(2)` 把相邻字符两两合并(8→4→2→1),低层看相邻、高层看整块;组件全部改写成 nn.Module 风格的小类再拼成 `Sequential` | ✅   |
| [`demo.ipynb`](demo.ipynb) / [`demo.py`](demo.py) | 开胃小菜:用 C++ 的 `vector<vector<...>>` 类比 PyTorch 张量,顺一遍 shape / view / 广播这些天天要打交道的东西                                                              | ✅   |
| [`namesgenerator.py`](namesgenerator.py)       | **免训练版生成器**:加载 Part 5 训好的权重(`makemore/makemore_5/names_model.pt`),`python3 namesgenerator.py` 直接就出名字,不读语料、不训练                                                                     | ✅   |

> ✅ 已完成 ｜ 🚧 建设中 ｜ ⬜ 未开始
>
> **makemore Part 1 ~ Part 5 已全部完结 🎉**(每一章都有可跑的脚本 + 笔记本),学习路线上的下一站是 **Let's build GPT**,正在建设中。

### Part 3 具体做了什么

Part 2 的 MLP 能跑通,但跑得"心里没底":激活值往 tanh 两端跑(饱和)、梯度越传越小、每加一层就得重新调初始化。Part 3 把这些不安全感一个个拆开:

1. **看得见里面**:手写三个层类,每层都把输出存进 `self.out` 并全程 `retain_grad()`,于是激活值分布、梯度分布都能单独画出来体检(笔记本里那些直方图)
2. **何恺明初始化**:`W ~ N(0, 1/fan_in)`,让前向传播中每层输出的方差保持不变;tanh 前面再乘 `5/3` 的 gain 补偿它"压缩"的脾气
3. **BatchNorm**:把每层输出强行拉回均值 0 / 方差 1,训练从此不太看初始化脸色,可以放心往下叠到 5 层
4. **最大的坑**:BatchNorm 训练时用当前 batch 的统计量,推理时必须改用滑动平均,两个分支靠 `self.training` 开关切换。生成名字前忘了关,`batch=1` 会让 `torch.var` 除以 `N-1=0` 算出 `nan`,顺着 softmax 传给 `torch.multinomial` 直接报错——更阴险的是 `nan` 会被写进 `running_var` 永久污染

最终 5 层 MLP + BatchNorm,验证集 loss 从 Part 2 的约 2.19 降到约 2.09。

### Part 4 具体做了什么

Part 3 的网络能训起来,但梯度是 `loss.backward()` 给的——相信它对,却说不出它为什么对。Part 4 把自动求导整个拆掉,自己当一回求导机器:

1. **从 loss 往回手推**:`cross_entropy` 的梯度为什么正好长得像 `softmax - onehot`(只差一个 `1/n` 的缩放),然后是 `dlogits` → `dW2` / `db2` → tanh 的 `1 - h²` → `dW1` / `db1`,每一步都只是链式法则
2. **最难的一步是 BatchNorm**:`dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))`——一行里套着两层"batch 统计量带来的耦合":每个样本都参与了均值和方差的计算,所以求导时梯度要回头看 batch 里的所有人,`n/(n-1)` 那项就是 `var` 用无偏估计留下的
3. **嵌入表要累加**:同一个字符在一个 batch 里会出现好几次,`dC[ix] += demb[k,j]` 必须用 `+=`(scatter-add),写成 `=` 会把其他位置的贡献丢掉
4. **验收靠对拍**:每一步都用 `cmp()` 把中间量的手算梯度和 `loss.backward()` 的结果比对,全部对齐才继续往下走

这一章的意义不在 loss(网络没换,只是梯度从 autograd 换成手算,下降曲线就该和之前一样),而在于**从此 autograd 不再是黑盒**。

### Part 5 具体做了什么

Part 4 之前的网络都是一根筋:把上下文整段拍平成一长条,一次性全连接。Part 5 换了个组织方式,顺便把上下文从 3 个字符拉到 8 个:

1. **分层合并 (WaveNet 思路)**:`FlattenConsecutive(2)` 把相邻两个时间步拼成一组,`(B,8,24) → (B,4,48) → (B,2,256) → (B,256)`,8 个字符合并成 4 组 → 2 组 → 1 组,每合并一次就过一遍 `Linear + BatchNorm + Tanh`。低层看相邻两字符,高层看整段——信息是逐级融上去的,而不是一上来就全连接
2. **组件化**:`Linear` / `BatchNorm1d` / `Tanh` / `Embedding` / `FlattenConsecutive` 都写成 nn.Module 风格的小类(各自带 `__call__` 和 `parameters()`),再用 `Sequential` 串起来——这就是 PyTorch 里 `nn.Sequential` 的雏形,顺手把 `torch.nn` 那层抽象拆开看了一遍
3. **评估模式收尾**:`layer.training = False` 让 BatchNorm 改用训练时累积的 `running_mean` / `running_var`,不再依赖当前 batch——Part 3 埋下的那个坑在这里正式合上
4. **结果**:7.6 万参数,验证集 loss 从 Part 3 的约 2.09 降到 **1.99**

> 视频末尾还演示了用"膨胀因果卷积"实现同样结构的版本,笔记里没有整理——那部分属于卷积网络的地盘,可以留到以后回来看。

### 🎉 makemore 完结

五个 Part 是同一个任务("生成英文名字")被反复重做,而每一步只解决上一步暴露出来的那一个问题:

| Part | 上一步留下的问题            | 这一章的升级                                        | 验证 loss   |
| --- | ------------------- | --------------------------------------------- | --------- |
| 1   | 起点:模型该长什么样          | bigram:只看前 1 个字符(计数版 + 神经网络版)                  | ~2.45     |
| 2   | 上下文太短、模型太浅          | 嵌入表 + 隐藏层 + `tanh`,看 3 个字符                     | ~2.19     |
| 3   | 一加深就训不动             | 何恺明初始化 + BatchNorm,放心叠到 5 层                   | ~2.09     |
| 4   | autograd 是个黑盒       | 手写反向传播,每个梯度都跟 autograd 对拍                     | 与 Part 3 同水平 |
| 5   | 整段拍平太浪费             | 分层合并 (WaveNet),看 8 个字符                        | **1.99**  |

> Part 1~3 的数字取自各章笔记本的评估输出;Part 5 是本机实跑的结果(训练集 1.79 / 验证集 1.99)。Part 4 网络没换、只换了求导方式,loss 自然持平,它的验收标准是 `cmp()` 全部对齐。

一条主线贯穿始终:**怎么让网络多看几个字符还不崩**。从 1 个字符到 3 个字符,靠嵌入 + 非线性把离散符号变成向量;从 3 层叠到 5 层,靠归一化把每层的分布拉回可控区间;从 3 个字符到 8 个字符,靠换一种组织信息的方式(分层合并)而不是硬堆全连接。而 micrograd 和 Part 4 则是在回答一个更底层的问题:**梯度到底从哪来**。

至此 makemore 系列全部完结 ✅,下一站是 Transformer。

## 🧭 学习路线(Zero to Hero)

1. **micrograd** ✅ — 手写自动求导引擎,理解反向传播的本质([视频](https://www.youtube.com/watch?v=VMj-3S1tku0) / [代码](https://github.com/karpathy/micrograd))
2. **makemore Part 1** ✅ — bigram 语言模型:计数版 + 神经网络版,负对数似然损失与采样([视频](https://www.youtube.com/watch?v=PaCmpygFfXo) / [代码](https://github.com/karpathy/makemore))
3. **makemore Part 2** ✅ — 多层感知机 (MLP):嵌入、隐藏层、`tanh` 与学习率搜索([视频](https://www.youtube.com/watch?v=TCH_1BHY58) / [代码](https://github.com/karpathy/makemore))
4. **makemore Part 3** ✅ — Activations & Gradients & BatchNorm:激活诊断、死亡神经元、初始化与批归一化([视频](https://www.youtube.com/watch?v=P6sfmUTpUmc) / [代码](https://github.com/karpathy/makemore))
5. **makemore Part 4** ✅ — Becoming a Backprop Ninja:扔掉 autograd 手动反向传播,每个中间量的梯度都跟 autograd 对拍([视频](https://www.youtube.com/watch?v=q8SA3rM6ckI) / [代码](https://github.com/karpathy/makemore))
6. **makemore Part 5** ✅ — Building a WaveNet:上下文拉到 8 个字符,分层合并成树状结构,顺手把 PyTorch 的 `nn.Module` 拆开看([视频](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) / [代码](https://github.com/karpathy/makemore))
7. **Let's build GPT** 🚧 建设中 — 从零手写 Transformer / GPT:自注意力、多头、残差、LayerNorm,一路写到 nanoGPT([视频](https://www.youtube.com/watch?v=kCc8FmEb1nY) / [代码](https://github.com/karpathy/ng-video-lecture))

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

- 每个课程目录下的 `.py` 可以直接跑(脚本会自己去 `makemore/names.txt` 找语料,在哪个目录下执行都可以),例如:

```bash
python makemore/makemore_5/makemore_5.py
```

- 不想等训练、只想看成品?根目录的 `namesgenerator.py` 直接加载 Part 5 训好的权重采样名字:权重放在 `makemore/makemore_5/names_model.pt`(由 `makemore_5.py` 训练结束时写出,重训一次就会自动用上新的),模型结构、采样逻辑与 Part 5 一模一样,只是跳过了 20 万步训练:

```bash
python3 namesgenerator.py
```

> 直接敲 `python3` 时用的是系统 python,里面通常没有 torch。脚本会自己检查:发现不在虚拟环境里、而仓库里又有 `venv/`,就自动换成 `venv/bin/python` 重新跑一遍自己,所以不激活环境也能跑;实在找不到 venv 才会提示你先 `source venv/bin/activate`。

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
- README是在我学习完之后由DeepSeek-v4-flash生成

## 📄 License

[MIT](LICENSE)
