"""
MakeMore Part 1: bigram 语言模型 —— 两种实现
===========================================

1. 计数版 (count-based)
   频数矩阵 N -> 概率矩阵 P = N / 行和
   这是最大似然估计 (MLE) 的闭式解: 数一数就行, 不需要训练
   loss ≈ 2.4541, 是 bigram 模型家族的成绩下限

2. 神经网络版 (neural net)
   one-hot 输入 -> 权重矩阵 W(27x27) -> exp -> 归一化 (softmax)
   W 随机初始化, 用梯度下降最小化负对数似然
   训练后 W 收敛为"对数计数表": W ≈ log(N) + 每行常数, loss ≈ 2.46

结尾用同一颗随机种子采样对比: 两种实现生成的"名字"几乎一模一样
—— 神经网络用梯度下降, 把计数表重新学了出来。

参考: Andrej Karpathy - Neural Networks: Zero to Hero (makemore Part 1)
"""

import os
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------
# 0. 数据准备 (两种模型共用)
# ---------------------------------------------------------------
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'names.txt')
words = open(DATA, 'r').read().splitlines()     # 32033 个名字

chars = sorted(list(set(''.join(words))))       # 26 个小写字母
# 字符 <-> 数字对照表: '.' 占 0 号位, 同时充当"开头标记"和"结束标记"
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}          # 反查表

# 频数矩阵 N[ch1, ch2]: ch2 跟在 ch1 后面出现了多少次
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']               # 给名字包上开头和结尾
    for ch1, ch2 in zip(chs, chs[1:]):          # 两两配对成 bigram
        N[stoi[ch1], stoi[ch2]] += 1

print(f'数据: {len(words)} 个名字, 共 {N.sum().item()} 个 bigram')
print()


# ===============================================================
# 模型一: bigram 计数版 —— 不需要训练的模型
# ===============================================================
# MLE 的闭式解: 最优概率 = 频率本身
# P[ch1, ch2] = N[ch1, ch2] / 第 ch1 行的总和
P = N.float()
P /= P.sum(1, keepdim=True)
# ⚠️ keepdim 不可省!
# P.sum(1) 是形状 (27,), 和 P(27,27) 相除时被当成 (1,27) 竖向广播,
# 变成"逐列相除"——不报错但每行和不再是 1, 静默算错。
# P.sum(1, keepdim=True) 是 (27,1), 横向广播, 每行除以自己的行和, 正确。

# --- 用 loss 评估: 平均负对数似然 ---
# 对每个真实 bigram, 查"正确答案"被模型给了多少概率, 取 -log 再平均
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
        n += 1
print(f'[计数版] 平均负对数似然 loss = {(-log_likelihood / n).item():.4f}  (bigram 家族下限)')

# --- 平滑 (+1): 给所有计数加 1 ---
# 否则数据里没出现过的组合概率为 0, 评估时 log(0) = -inf
P_smooth = (N + 1).float()
P_smooth /= P_smooth.sum(1, keepdim=True)

# --- 采样生成名字 ---
# 生成循环: 从 '.' 开始, 按概率抽下一个字符, 抽到 '.' 结束
def sample_names(p_func, n_names=20, seed=2147483647):
    """p_func(ix): 给定当前字符下标, 返回 27 个"下一字符"的概率"""
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(n_names):
        out = []
        ix = 0
        while True:
            ix = torch.multinomial(p_func(ix), num_samples=1,
                                   replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        names.append(''.join(out))
    return names

print('[计数版] 生成的 10 个名字:')
print('  ', sample_names(lambda ix: P[ix], n_names=10))
print()


# ===============================================================
# 模型二: bigram 神经网络版 —— 用梯度下降学出同一张表
# ===============================================================
# 训练集: "问题" xs (当前字符) -> "答案" ys (实际的下一个字符), 一一对应
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()                             # 22.8 万个例子

# one-hot 编码: 把"按索引取第 ix 行"变成可求导的矩阵乘法 one_hot(ix) @ W
xenc = F.one_hot(xs, num_classes=27).float()

# 权重矩阵: 27 个神经元, 第 j 列负责"下一个字符是 j"的分数 (logits)
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
# ⚠️ 不传 generator=g 会改用全局随机流, 每次运行初始 W 都不同, 无法复现

# --- 训练: 梯度下降最小化负对数似然 ---
h = -50                                         # 学习率
for k in range(200):
    # 前向传播: logits -> exp(把负数拉正) -> 归一化(和为1) = softmax
    # logits 里存的是"对数空间的分数", 所以 W 最终会收敛成对数计数表
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    # 损失: 对每个例子, 挑出"正确答案"那格的概率, 取 -log 再平均
    loss = -probs[torch.arange(num), ys].log().mean()
    # 反向传播 + 更新
    W.grad = None
    loss.backward()
    W.data += h * W.grad
    if k % 50 == 0:
        print(f'[神经网络] 第 {k:3d} 步 loss = {loss.item():.4f}')
print(f'[神经网络] 训练完成, loss = {loss.item():.4f}  (计数版是 2.4541)')

# --- 训练后 W 变成了什么? 对数计数表! ---
# 第 'e' 行 (下标 5): exp(W) 和频数 N 只差一个常数倍
print('行 e 的对比 (exp(W) 与 N 成比例):')
print('  N     :', N[5, :6].tolist())
print('  exp(W):', [round(v, 1) for v in W.detach().exp()[5, :6].tolist()])
print()

# --- 神经网络版查表: 当前字符 -> softmax(one_hot(ix) @ W) ---
def nn_probs(ix):
    x = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = x @ W
    counts = logits.exp()
    return counts / counts.sum(1, keepdim=True)

print('[神经网络] 生成的 10 个名字:')
print('  ', sample_names(nn_probs, n_names=10))
print()


# ===============================================================
# L2 正则化 (可选) —— 神经网络版的"+1 平滑"
# ===============================================================
# loss = 负对数似然 + 0.01*(W**2).mean()
# 惩罚"权重太大": 把 W 往 0 拉  <=>  把 exp(W) 往 1 拉  <=>  相当于给所有计数 +1
# 效果: 训练 loss 略升, 但 W 的最大值被压小, 模型对稀有事件更"谦逊"
g2 = torch.Generator().manual_seed(2147483647)
W_reg = torch.randn((27, 27), generator=g2, requires_grad=True)
for k in range(200):
    logits = xenc @ W_reg
    probs = logits.exp() / logits.exp().sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W_reg ** 2).mean()
    W_reg.grad = None
    loss.backward()
    W_reg.data += -50 * W_reg.grad
print(f'[正则化] loss = {loss.item():.4f} (无正则 2.4624), '
      f'W 最大值 {W_reg.max().item():.2f} (无正则 4.26)')
print()


# ===============================================================
# 完结撒花: 同一颗种子, 两个模型, 殊途同归
# ===============================================================
# 生成循环一字未动, 只有"概率从哪来"这一行换了:
#   计数版:  p = P[ix]                      查表 (MLE 闭式解)
#   神经网络: p = softmax(one_hot(ix) @ W)   现场计算 (梯度下降学出来的)

print('同一颗种子 (2147483647) 各生成 5 个名字:')
print('  计数版  :', sample_names(lambda ix: P[ix], n_names=5))
print('  神经网络:', sample_names(nn_probs, n_names=5))
print()
print('几乎一模一样 —— 神经网络用梯度下降重新学出了计数表, 殊途同归。')
