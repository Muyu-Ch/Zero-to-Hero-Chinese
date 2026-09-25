"""
MakeMore Part 5: WaveNet 式分层网络 名字生成模型
==============================================

Part 4 把上下文一次性拍平成一长条喂给 MLP;
Part 5 换成"分层两两合并": block_size 拉到 8, 每层把相邻两个时间步拼成一组,
8 -> 4 -> 2 -> 1 逐层融合, 结构像一棵树 (WaveNet 的做法), 参数反而更少。

顺带把每个组件都改写成 nn.Module 风格的小类:
Linear / BatchNorm1d / Tanh / Embedding / FlattenConsecutive / Sequential。

流程: 构建数据集 -> 定义各种层 -> 搭网络 -> 训练 -> 评估 -> 生成 20 个名字

参考: Andrej Karpathy - Neural Networks: Zero to Hero (makemore Part 5)
"""

import torch
import torch.nn.functional as F
import random

# ---------------------------------------------------------------
# 0. 数据准备
# ---------------------------------------------------------------
words=open("./makemore_5/names.txt","r").read().splitlines()

chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
vocab_size=len(itos)
print(f'[数据] 词表大小 vocab_size = {vocab_size} (26 个字母 + 1 个 ".")')

# ---------------------------------------------------------------
# 1. 构建数据集: 滑动窗口切出 (上下文 -> 下一字符) 样本对
# ---------------------------------------------------------------
def build_dataset(words,block_size):
  X, Y = [], []
  for w in words:

    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)  
      context = context[1:] + [ix] 

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y

# 80/10/10 切分: 训练集(更新参数) / 验证集(调旋钮时反复看) / 测试集(只碰一次)
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

block_size=8
print(f'[数据] 上下文长度 block_size = {block_size} (Part 5 看得比 Part 4 远)')

Xtr, Ytr = build_dataset(words[:n1],block_size)#0%~80%
Xdev, Ydev = build_dataset(words[n1:n2],block_size)#80%~90%
Xte, Yte = build_dataset(words[n2:],block_size)#90~100%

print(f'[数据] 训练集 {Xtr.shape[0]} 个样本 | 验证集 {Xdev.shape[0]} | 测试集 {Xte.shape[0]}')

# ---------------------------------------------------------------
# 2. 各种层: 手写 nn.Module 风格的小类, 都有 __call__ 和 parameters
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
# 3. 模型: 分层堆叠 (旋钮: n_embd 嵌入维度 / n_hidden 隐藏层宽度)
# ---------------------------------------------------------------
n_embd=24
n_hidden=128

# 每层的形状变化 (以 batch=B 为例):
#   Embedding             (B, 8, 24)
#   FlattenConsecutive(2) (B, 4, 48)  -> Linear -> (B, 4, 128)
#   FlattenConsecutive(2) (B, 2, 256) -> Linear -> (B, 2, 128)
#   FlattenConsecutive(2) (B, 1, 256) -> squeeze -> (B, 256) -> Linear -> (B, 128)
#   最后 Linear(n_hidden, vocab_size) -> (B, 27)
layers=[
    Embedding(vocab_size,n_embd),
    FlattenConsecutive(2),Linear(n_embd*2,  n_hidden,bias=False),BatchNorm1d(n_hidden),Tanh(),
    FlattenConsecutive(2),Linear(n_hidden*2,n_hidden,bias=False),BatchNorm1d(n_hidden),Tanh(),
    FlattenConsecutive(2),Linear(n_hidden*2,n_hidden,bias=False),BatchNorm1d(n_hidden),Tanh(),
    Linear(n_hidden,vocab_size),
]

with torch.no_grad():
    layers[-1].weight*=0.1 #最后一层缩小: 一开始别让 logits 太大, softmax 别太自信

model=Sequential(layers)

parameters=model.parameters()
print(f'[模型] 层结构 ({len(layers)} 层, {n_embd} 维嵌入 -> 3 次相邻合并 -> {n_hidden} 宽):')
for layer in layers:
    shapes=[tuple(p.shape) for p in layer.parameters()]
    if shapes:#Tanh / FlattenConsecutive 没有参数, 跳过
        print(f'    {type(layer).__name__:<18} 参数 {shapes}')
print(f'[模型] 总参数量 {sum(p.nelement() for p in parameters)}')
for p in parameters:
    p.requires_grad=True

# ---------------------------------------------------------------
# 4. 训练: 前向 -> 算 loss -> retain_grad -> 清梯度 -> 反向 -> 更新
# ---------------------------------------------------------------
max_steps=200000
batch_size=32
lossi=[]

print(f'[训练] {max_steps} 步, batch {batch_size}, lr 0.1 -> 0.01 (第 100000 步衰减)')

for i in range(max_steps):

    #batch
    ix = torch.randint(0,Xtr.shape[0],(batch_size,))
    Xb,Yb=Xtr[ix],Ytr[ix]

    #前向传播
    logits=model(Xb)
    loss=F.cross_entropy(logits,Yb)

    #让每一层的out的grad保留下来
    for layer in layers:
        layer.out.retain_grad()

    #清空梯度
    for p in parameters:
        p.grad=None

    #反向传播
    loss.backward()

    #更新
    lr = 0.1 if i < max_steps/2 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i%10000==0:
        print(f'  第 {i:6d}/{max_steps} 步 minibatch loss = {loss.item():.4f}')

    lossi.append(loss.log10().item())

print(f'[训练] 结束, 最后一步 minibatch loss = {loss.item():.4f}')

# ---------------------------------------------------------------
# 5. 评估: 训练集 / 验证集 各算一次 loss
# ---------------------------------------------------------------
# 切到 eval 模式: BatchNorm 改用训练时累积的 running_mean / running_var, 不再看当前 batch
for layer in model.layers:
    layer.training=False

print('[评估]')
@torch.no_grad()
def split_loss(split):
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]

    logits=model(x)
    loss = F.cross_entropy(logits, y)

    print(f'  {split} loss = {loss.item():.4f}')

split_loss('train')
split_loss('val')
print('  (验证 loss 和训练 loss 的差距 = 过拟合的体温计, 差距越小越好)')

# ---------------------------------------------------------------
# 6. 生成 20 个名字: 滑动窗口滚出新字符, 直到抽到 '.'
# ---------------------------------------------------------------
print('[生成] 20 个名字:')
for i in range(20):
    context = [0] * block_size
    out = []
    while True:
        logits=model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(
            probs,
            num_samples=1
        ).item()
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break

    print('  ' + ''.join(out))