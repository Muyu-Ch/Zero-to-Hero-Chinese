"""
MakeMore Part 2: 多层感知机 (MLP) 名字生成模型
============================================

Part 1 的 bigram 只看 1 个字符 (loss 下限 2.45);
Part 2 装上"看 3 个字符的眼睛" + 隐藏层, 把 loss 打到 2.1 附近。

结构: 上下文(3字符) -> 嵌入表 C -> 多层 tanh 网络 -> 27 个分数 -> softmax
全部超参数集中成"旋钮", 可以随意炼丹。

流程: 构建数据集 -> 搭网络 -> 训练 -> 三集评估 -> 生成 20 个名字
训练/评估/生成共用同一个 forward 函数 (只是批大小不同)。

参考: Andrej Karpathy - Neural Networks: Zero to Hero (makemore Part 2)
"""

import os
import random
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------
# 0. 数据准备
# ---------------------------------------------------------------
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'names.txt')
words = open(DATA, 'r').read().splitlines()      # 32033 个名字

chars = sorted(list(set(''.join(words))))        # 26 个小写字母
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0                                    # '.' 占 0 号位: 开头标记 + 结束标记
itos = {i: s for s, i in stoi.items()}           # 反查表

# ---------------------------------------------------------------
# 旋钮: 所有超参数集中在这里, 想炼丹就调这里
# ---------------------------------------------------------------
vocab_size = 27                                  # 固定参数: 字符表大小
g = torch.Generator().manual_seed(2147483647)    # 固定种子 (初始化用)

block_size = 3                                   # 上下文长度: 看前几个字符
n_embd = 40                                      # 嵌入维度: 每个字符用几维向量表示
n_steps = 100000                                 # 训练步数
batch_size = 50                                  # 每个 minibatch 的样本数
lr = 0.03                                        # 学习率 (永远正数, 负号写在更新公式里)
# 每层大小: 输入 = 3个字符 × 40维 = 120; 最后必须 = vocab_size (27 个候选分数)
layer_size = [block_size * n_embd, 800, 100, vocab_size]

# ---------------------------------------------------------------
# 1. 构建数据集: 滑动窗口切出 (上下文 -> 下一字符) 样本对
# ---------------------------------------------------------------
def build_dataset(words, block_size):
    X, Y = [], []
    for w in words:
        context = [0] * block_size               # 开头全是 '.' (0)
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)                    # 问题: 当前上下文
            Y.append(ix)                         # 答案: 实际的下一个字符
            context = context[1:] + [ix]         # 滑动窗口: 丢最老, 接最新
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# 80/10/10 切分: 训练 / 验证 / 测试
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1], block_size)     # 训练集: 更新参数用
Xdev, Ydev = build_dataset(words[n1:n2], block_size) # 验证集: 调旋钮时反复看
Xte, Yte = build_dataset(words[n2:], block_size)     # 测试集: 只碰一次!

print('[数据] 训练集 %d 个样本 | 验证集 %d | 测试集 %d'
      % (Xtr.shape[0], Xdev.shape[0], Xte.shape[0]))

# ---------------------------------------------------------------
# 2. 模型: 嵌入表 + 层列表
# ---------------------------------------------------------------
C = torch.randn((vocab_size, n_embd), generator=g)   # 嵌入表 (27, 40)

Ws, bs = [], []                                      # 每层一块 W 和 b
for n1, n2 in zip(layer_size[:-1], layer_size[1:]):
    Ws.append(torch.randn((n1, n2), generator=g))
    bs.append(torch.randn(n2, generator=g))

parameters = [C] + Ws + bs                           # 拼接, 方便统一清零梯度
print(f'[模型] 层结构 {layer_size}, 总参数量 {sum(p.nelement() for p in parameters)}')
for p in parameters:
    p.requires_grad = True

def forward(emb):
    """前向传播: 嵌入后的 (N, block_size, n_embd) -> logits (N, vocab_size)
    训练(批32/50)、评估(整集)、生成(批1) 共用这一个函数 —— 只有批大小不同"""
    x = emb.view(-1, block_size * n_embd)            # 3 个向量拼成一条 30/120 维
    for W, b in zip(Ws[:-1], bs[:-1]):
        x = torch.tanh(x @ W + b)                    # 隐藏层: 线性 + tanh
    logits = x @ Ws[-1] + bs[-1]                     # 最后一层: 只线性, 不过 tanh
    return logits

# ---------------------------------------------------------------
# 3. 训练: 前向 -> 算 loss -> 清零梯度 -> 反向 -> 更新
# ---------------------------------------------------------------
print(f'[训练] {n_steps} 步, batch {batch_size}, lr {lr}')
for i in range(n_steps):
    # ① 前向传播 (F.cross_entropy 内部 = softmax + 取正确答案的 -log + 平均)
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    logits = forward(C[Xtr[ix]])
    loss = F.cross_entropy(logits, Ytr[ix])
    # ② 清零梯度 (不清零会累加!)
    for p in parameters:
        p.grad = None
    # ③ 反向传播
    loss.backward()
    # ④ 更新
    for p in parameters:
        p.data += -lr * p.grad
    # 进度打印
    if i % 5000 == 0 or i == n_steps - 1:
        print(f'  第 {i:6d} 步 minibatch loss = {loss.item():.4f}')

# ---------------------------------------------------------------
# 4. 评估: 三集各算一次 loss (测试集只碰这一次!)
# ---------------------------------------------------------------
print('[评估]')
for name, X, Y in [('训练集', Xtr, Ytr), ('验证集', Xdev, Ydev), ('测试集', Xte, Yte)]:
    loss = F.cross_entropy(forward(C[X]), Y)
    print(f'  {name} loss = {loss.item():.4f}')
print('  (验证/测试 与 训练 的差距 = 过拟合的体温计, 差距越小越好)')

# ---------------------------------------------------------------
# 5. 生成 20 个名字: 滑动窗口滚出新字符, 直到抽到 '.'
# ---------------------------------------------------------------
print('[生成] 20 个名字:')
g = torch.Generator().manual_seed(2147483647 + 10)   # 换个种子, 生成新鲜的名字
for _ in range(20):
    context = [0] * block_size                        # 上下文 = 3 个 '.' (名字还没开始)
    out = []
    while True:
        emb = C[torch.tensor([context])]              # 括号造批维度: (1, 3, 40)
        probs = F.softmax(forward(emb), dim=1)        # 分数 -> 概率 (沿 27 个候选归一化)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]                  # 滑动窗口: 丢最老, 接最新
        out.append(itos[ix])
        if ix == 0:                                   # 抽到 '.' 名字结束
            break
    print('  ' + ''.join(out))
