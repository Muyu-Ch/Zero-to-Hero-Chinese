"""
名字生成器: 加载 makemore Part 5 训练好的权重, 直接采样名字
=========================================================

不训练、也不读 names.txt —— 层定义 / 模型结构 / 采样逻辑全部来自 makemore_5.py,
唯一的区别是把"训练 20 万步"换成了"加载权重文件":

    makemore/makemore_5/makemore_5.py     训练 -> 存出 names_model.pt
    namesgenerator.py                     加载 names_model.pt -> 秒出名字

权重文件是训练脚本负责产出的(就放在它自己旁边), 想重新训练或换个语料,
去跑 makemore_5.py 就行, 它会覆盖那个文件, 这里会自动用上新的。

用法(在仓库根目录):
    python3 namesgenerator.py
"""

import os
import sys

# 直接 `python3 namesgenerator.py` 时, 用的可能是没装 torch 的系统 python;
# 仓库里自带 venv 的话, 就自动换成 venv 的 python 重跑一遍自己 (这样才 import 得到 torch)
try:
    import torch
except ModuleNotFoundError:
    _root = os.path.dirname(os.path.abspath(__file__))
    _venv_py = os.path.join(_root, 'venv', 'bin', 'python')
    # 判据是 sys.prefix: venv/bin/python 只是个指向系统 python 的软链接, 两者 realpath 一模一样,
    # 真正区分"在不在 venv 里"的是解释器旁边的 pyvenv.cfg (Python 把它读进 sys.prefix)
    if sys.prefix == sys.base_prefix and os.path.exists(_venv_py):
        os.execv(_venv_py, [_venv_py, os.path.abspath(__file__)] + sys.argv[1:])
    raise SystemExit('没装 torch。先激活虚拟环境再跑:\n'
                     '  source venv/bin/activate\n'
                     '  python namesgenerator.py')

import torch.nn.functional as F

# ---------------------------------------------------------------
# 1. 各种层: 与 makemore_5.py 完全相同的定义
# ---------------------------------------------------------------
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):

        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:#修改逻辑
            if x.ndim==2:
                dim=0
            else:
                dim=(0,1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

class Embedding:

  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))

  def __call__(self, IX):
    self.out = self.weight[IX]
    return self.out

  def parameters(self):
    return [self.weight]

class FlattenConsecutive:
    def __init__(self,n):
        self.n=n

    def __call__(self, x):
        B,T,C=x.shape
        x=x.view(B,T//self.n,C*self.n)
        if x.shape[1]==1:
            x=x.squeeze() #最后一组只剩 1 个时间步, 挤掉它变成 (B, C)
        self.out = x
        return self.out

    def parameters(self):
        return []

class Sequential:
    def __init__(self,layers):
        self.layers=layers

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        self.out=x
        return self.out

    def parameters(self):
        return[p for layer in self.layers for p in layer.parameters()]

# ---------------------------------------------------------------
# 2. 先读权重文件: 词表要从里面拿 (它决定 Embedding 和最后一层的形状)
#    权重由 makemore_5.py 训练后写出, 就放在它自己旁边
# ---------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, 'makemore', 'makemore_5', 'names_model.pt')
if not os.path.exists(CKPT):
    raise SystemExit(f'找不到权重文件\n  {CKPT}\n'
                     f'先跑一次 makemore/makemore_5/makemore_5.py 训练并生成它。')

state = torch.load(CKPT, weights_only=True)
itos = {i: c for i, c in enumerate(state['vocab'])}   # '.abcdefghijklmnopqrstuvwxyz'
vocab_size = len(itos)

# ---------------------------------------------------------------
# 3. 模型: 结构必须和训练时一字不差, 否则权重对不上
# ---------------------------------------------------------------
n_embd=24
n_hidden=128
block_size=8

layers=[
    Embedding(vocab_size,n_embd),
    FlattenConsecutive(2),Linear(n_embd*2,  n_hidden,bias=False),BatchNorm1d(n_hidden),Tanh(),
    FlattenConsecutive(2),Linear(n_hidden*2,n_hidden,bias=False),BatchNorm1d(n_hidden),Tanh(),
    FlattenConsecutive(2),Linear(n_hidden*2,n_hidden,bias=False),BatchNorm1d(n_hidden),Tanh(),
    Linear(n_hidden,vocab_size),
]

model=Sequential(layers)
parameters=model.parameters()
# 注: makemore_5.py 里那句 layers[-1].weight*=0.1 是给随机初始化用的,
#     这里的权重马上会被 names_model.pt 覆盖, 所以不需要。

# ---------------------------------------------------------------
# 4. 灌入训练好的权重
# ---------------------------------------------------------------
assert len(state['params'])==len(parameters), '权重文件与模型结构对不上, 重新训练一次吧'
for p, saved in zip(parameters, state['params']):
    p.data = saved          # 顺序由 layers 列表决定, 和训练时一致

# BatchNorm 的滑动统计量不在 parameters() 里, 要单独恢复
bn_layers=[l for l in model.layers if isinstance(l,BatchNorm1d)]
for layer, (mean, var) in zip(bn_layers, state['bn_running']):
    layer.running_mean, layer.running_var = mean, var

# 切到评估模式: BatchNorm 改用训练时累积的 running 统计量, 不再看当前 batch
# (附带一个隐藏效果: running_mean 的形状是 (1,1,128) / (1,128) 而不是 (128,) ——
#  生成时 batch=1, 最后一个 FlattenConsecutive 会把张量压成 1 维,
#  正是靠这个形状把批维度广播回来, 最后 logits 才是 (1,27), softmax(dim=1) 才成立)
for layer in model.layers:
    layer.training=False

print(f'[生成器] 已加载 {os.path.relpath(CKPT, ROOT)}: {sum(p.nelement() for p in parameters)} 个参数, 评估模式')

# ---------------------------------------------------------------
# 5. 采样: 与 makemore_5.py 相同的滑动窗口, 抽到 '.' 就收工
# ---------------------------------------------------------------
SEED=2147483647+10   # 固定种子: 每次跑出来是同一批名字, 想换一批就改这里

@torch.no_grad()
def generate(n=20, seed=SEED):
    g=torch.Generator().manual_seed(seed)
    for _ in range(n):
        context=[0]*block_size      # 开头全是 '.', 名字还没开始
        out=[]
        while True:
            logits=model(torch.tensor([context]))      # 括号造批维度: (1, 8)
            probs=F.softmax(logits,dim=1)
            ix=torch.multinomial(probs,num_samples=1).item()
            context=context[1:]+[ix]                   # 滑动窗口: 丢最老, 接最新
            out.append(itos[ix])
            if ix==0:                                  # 抽到 '.' 名字结束
                break
        print('  '+''.join(out))

print("生成几个名字？")
n=int(input())
print('[生成]',n,'个名字:')
generate(n)
