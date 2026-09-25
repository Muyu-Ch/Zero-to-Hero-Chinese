"""
MakeMore Part 3: 激活值 / 梯度 / BatchNorm 名字生成模型
=====================================================

Part 2 的 MLP 能跑通, 但跑得"心里没底":
激活值会往 tanh 两端跑 (饱和), 梯度越传越小或越传越大,
每加一层就得重新调初始化和学习率 —— 炼丹全靠手感。

Part 3 把这些不安全感一个个拆开解决:
  1. 手写 Linear / BatchNorm1d / Tanh 三个类, 每层都留一份 self.out,
     于是激活值分布、梯度分布都能单独画出来"体检"(见 makemore_3.ipynb)
  2. 用何恺明初始化 (Kaiming init): W ~ N(0, 1/fan_in),
     让每层输出的方差在前向传播中保持不变, 开局就不炸
  3. 用 BatchNorm1d 把每层输出强行拉回 均值0 / 方差1,
     训练从此不受初始化好坏的影响, 可以放心往下加深

结构: 上下文(3字符) -> 嵌入表 C -> [Linear -> BatchNorm -> Tanh] × 5 -> Linear -> BatchNorm -> 27 个分数

⚠️ 本文件最大的坑 (Part 3 的核心考点, 也是调试最久的地方):
   BatchNorm 训练时用"当前 batch 的均值/方差", 推理时必须改用"滑动平均".
   两个分支靠 self.training 这个开关选择 —— 生成名字前必须把它设成 False!
   否则生成时 batch=1, 求方差的分母 (N-1) 等于 0, 算出 nan,
   再顺着 softmax 传给 torch.multinomial, 直接报错。
   更阴险的是: nan 会被写进 running_var 永久污染, 事后补救也来不及。

流程: 手写工具箱 -> 数据准备 -> 构建数据集 -> 搭网络 -> 训练 -> 切推理模式 -> 生成 20 个名字

参考: Andrej Karpathy - Neural Networks: Zero to Hero (makemore Part 3)
"""

import os
import random
import torch
import torch.nn.functional as F

# ===============================================================
# 1. 手写工具箱: Linear / BatchNorm1d / Tanh
# ===============================================================
# 为什么不用 nn.Linear? 因为要"看得见里面":
# 每个类都把输出存进 self.out, 训练时一路 retain_grad(),
# 事后就能把每一层的激活分布 / 梯度分布画出来 (Part 3 的核心实验)。

class Linear:
    """线性层: y = x @ W + b —— 神经网络里唯一"自己会学"的部分"""
    def __init__(self, fan_in, fan_out, bias=True):  # 这里给出了是否使用偏置项的选项
        # fan_in 和 fan_out 是扇入和扇出, 也就是矩阵的行数和列数
        # 也就是输入特征数量和输出特征数量
        # 何恺明初始化 (Kaiming init): 除以 sqrt(fan_in), gain 默认为 1
        # 不除的话, 100 维点乘 100 维, 输出的标准差会涨到 10 倍
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in ** 0.5
        # 这里的 generator=g 后续会调用全局变量 g
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight                   # 可以直接 (x) 调用乘法
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    """批归一化: 把每个特征的激活值拉成 均值0 / 方差1, 再用 gamma/beta 缩放平移
    训练时用当前 batch 的统计量, 推理时用累积的滑动平均 (running_mean / running_var)"""
    def __init__(self, dim, eps=1e-5, momentum=0.1):

        self.eps = eps
        self.momentum = momentum
        self.training = True                         # ⭐ 这个开关决定前向走哪个分支

        self.gamma = torch.ones(dim)                 # 可学习的缩放 (初始: 原样输出)
        self.beta = torch.zeros(dim)                 # 可学习的平移 (初始: 不平移)

        self.running_mean = torch.zeros(dim)         # 滑动平均 (不是参数, 靠动量更新)
        self.running_var = torch.ones(dim)           # 同上

    def __call__(self, x):
        # ---- 前向传播 ----
        if self.training:
            xmean = x.mean(0, keepdim=True)          # 当前 batch 的均值
            xvar = x.var(0, keepdim=True)            # 当前 batch 的方差
            # ⚠️ torch.var 默认 unbiased=True, 除以 N-1.
            #    训练时 batch=32 无所谓; 推理时 batch=1 → 除以 0 → nan,
            #    所以生成名字之前必须把 self.training 设成 False, 走下面的 else!
        else:
            xmean = self.running_mean                # 推理: 用训练时攒下的滑动平均
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # 这里相比较之前除法多了一个 eps 防止除0
        self.out = self.gamma * xhat + self.beta
        # ---- 更新缓冲区 (只在训练时) ----
        if self.training:
            with torch.no_grad():
                # 指数滑动平均: 新值 = 0.9 × 旧值 + 0.1 × 本批统计量
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    """激活函数: 把值压进 (-1, 1) —— 引入非线性, 但也会"饱和"(两端导数 → 0)"""
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


# ===============================================================
# 2. 数据准备
# ===============================================================
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../names.txt')
words = open(DATA, 'r').read().splitlines()           # 32033 个名字
print(f'[数据] 共 {len(words)} 个名字, 例如: {words[:3]}')

chars = sorted(list(set(''.join(words))))             # 26 个小写字母
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0                                         # '.' 占 0 号位: 开头标记 + 结束标记
itos = {i: s for s, i in stoi.items()}                # 反查表
vocab_size = len(itos)
print(f'[数据] 字符表 {vocab_size} 个: {itos}')

block_size = 3                                        # 上下文长度: 看前几个字符


# ===============================================================
# 3. 构建数据集: 滑动窗口切出 (上下文 -> 下一字符) 样本对
# ===============================================================
def build_dataset(words, block_size):
    X, Y = [], []
    for w in words:

        context = [0] * block_size                    # 开头全是 '.' (0)
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)                         # 问题: 当前上下文
            Y.append(ix)                              # 答案: 实际的下一个字符
            context = context[1:] + [ix]              # 滑动窗口: 丢最老, 接最新

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# 80/10/10 切分: 训练 / 验证 / 测试
random.seed(42)
random.shuffle(words)                                 # 打乱后再切, 避免顺序带来的偏差
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1], block_size)      # 0% ~ 80%: 更新参数用
Xdev, Ydev = build_dataset(words[n1:n2], block_size)  # 80% ~ 90%: 调旋钮时反复看
Xte, Yte = build_dataset(words[n2:], block_size)      # 90% ~ 100%: 只碰一次!
print('[数据] 训练集 %d 个样本 | 验证集 %d | 测试集 %d'
      % (Xtr.shape[0], Xdev.shape[0], Xte.shape[0]))


# ===============================================================
# 4. 旋钮: 超参数集中在这里, 想炼丹就调这里
# ===============================================================
n_embed = 10                                          # 嵌入维度: 每个字符用几维向量表示
n_hidden = 100                                        # 隐藏层宽度
g = torch.Generator().manual_seed(2147483647)         # 固定种子 (初始化用)


# ===============================================================
# 5. 模型: 嵌入表 + 层列表
# ===============================================================
C = torch.randn((vocab_size, n_embed), generator=g)   # 嵌入表 (27, 10)

layers = [
    Linear(n_embed * block_size, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden          ), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden          ), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden          ), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden          ), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size        ), BatchNorm1d(vocab_size),
]
# 五组 "线性 -> 批归一化 -> tanh" 叠罗汉; 最后一层只出 27 个分数, 不过 tanh

with torch.no_grad():
    # 最后一层 gamma 乘 0.1: 让初始 logits 接近 0, 开局 loss 才不爆炸
    # (logits 全 0 时 softmax 是均匀分布, loss = -log(1/27) ≈ 3.30)
    layers[-1].gamma *= 0.1                          # 最后一层换成 batchnorm

    # 前面每个 Linear 的权重乘 5/3: tanh 的推荐 gain 就是 5/3
    # 用来抵消 tanh 的"压缩"效果, 让前向和反向的方差保持稳定
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= (5 / 3)

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(f'[模型] 5 层 MLP + BatchNorm, 总参数量 {sum(p.nelement() for p in parameters)}')
for p in parameters:
    p.requires_grad = True


# ===============================================================
# 6. 训练: 前向 -> 算 loss -> 清零梯度 -> 反向 -> 更新
# ===============================================================
max_steps = 200000
batch_size = 32
print(f'[训练] {max_steps} 步, batch {batch_size}, lr 前一半 0.1 / 后一半 0.01')

for i in range(max_steps):

    # ① 随机抽一个 minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # ② 前向传播: 嵌入 -> 展平 -> 依次过每一层
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)                    # (32, 3, 10) -> (32, 30)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)                     # 内部 = softmax + 取正确答案的 -log + 平均

    # ③ 让每一层的 out 的 grad 保留下来 (Part 3 的诊断实验要用, 平时可以删掉)
    for layer in layers:
        layer.out.retain_grad()

    # ④ 清空梯度 (不清零会累加!)
    for p in parameters:
        p.grad = None

    # ⑤ 反向传播
    loss.backward()

    # ⑥ 更新: 学习率阶梯衰减 —— 前 10 万步 0.1 猛冲, 后 10 万步 0.01 精修
    lr = 0.1 if i < max_steps / 2 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # 进度打印
    if i % 20000 == 0 or i == max_steps - 1:
        print(f'  第 {i:6d} 步 minibatch loss = {loss.item():.4f}')


# ===============================================================
# 7. 切换到推理模式 + 生成 20 个名字
# ===============================================================
# ⭐ Part 3 最关键的两行: 把每个 BatchNorm 的 self.training 关掉!
#    不关的话, 生成时 batch=1 → x.var() 除以 (1-1)=0 → nan → multinomial 报错
for layer in layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False

g = torch.Generator().manual_seed(2147483647 + 10)    # 换个种子, 生成新鲜的名字
print('[生成] 20 个名字:')

for i in range(20):

    context = [0] * block_size                        # 上下文 = 3 个 '.' (名字还没开始)
    out = []

    while True:

        # embedding: (3,) -> (3, 10)
        emb = C[torch.tensor(context)]

        # 展平 + 过每一层: (1, 30) -> ... -> (1, 27)
        # ⚠️ 这里的批大小是 1! 所以上面必须先切到推理模式
        x = emb.view(1, -1)
        for layer in layers:
            x = layer(x)

        # x 现在就是 logits, softmax 成概率 (沿 27 个候选归一化)
        probs = F.softmax(x, dim=1)

        # 按概率采样字符
        ix = torch.multinomial(
            probs,
            num_samples=1,
            generator=g
        ).item()

        # 更新上下文: 滑动窗口, 丢最老, 接最新
        context = context[1:] + [ix]

        # 保存字符
        out.append(itos[ix])

        # 遇到结束符
        if ix == 0:
            break

    print('  ' + ''.join(out))
