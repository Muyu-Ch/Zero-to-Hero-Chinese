import torch
import torch.nn.functional as F

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):#这里给出了是否使用偏置项的选项
        #fan_in和fan_out是扇入和扇出，也就是矩阵的行数和列数
        #也就是输入特征数量和输出特征数量
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5#使用了何恺明初始化法，gain默认为1
        #这里的generator=g后续会调用全局变量g
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        #可以直接(x)调用乘法
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
        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self,x):
       self.out=torch.tanh(x)
       return self.out
    def parameters(self):
        return []

words=open("names.txt","r").read().splitlines()
words[:3]

chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
print(itos)
vocab_size=len(itos)

block_size=3

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

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1],block_size)#0%~80%
Xdev, Ydev = build_dataset(words[n1:n2],block_size)#80%~90%
Xte, Yte = build_dataset(words[n2:],block_size)#90~100%

n_embed=10
n_hidden=100
g=torch.Generator().manual_seed(2147483647)

C=torch.randn((vocab_size,n_embed),generator=g)

layers=[
    Linear(n_embed*block_size,n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden,n_hidden          ), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden,n_hidden          ), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden,n_hidden          ), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden,n_hidden          ), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden,vocab_size        ), BatchNorm1d(vocab_size), 
]

with torch.no_grad():
    layers[-1].gamma *= 0.1 #最后一层换成batchnorm

    for layer in layers[:-1]:
        if isinstance(layer,Linear):
            layer.weight*=(5/3)

parameters=[C]+[p for layer in layers for p in layer.parameters()]
print("参数总量：",sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad=True

max_steps=200000
batch_size=32

for i in range(max_steps):

    #batch
    ix = torch.randint(0,Xtr.shape[0],(batch_size,),generator=g)
    Xb,Yb=Xtr[ix],Ytr[ix]

    #前向传播
    emb=C[Xb]
    x=emb.view(emb.shape[0],-1)
    for layer in layers:
        x=layer(x)
    loss=F.cross_entropy(x,Yb)

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

for layer in layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False

g = torch.Generator().manual_seed(2147483647+10)

for i in range(20):

    context = [0] * block_size
    out = []

    while True:

        # embedding
        emb = C[torch.tensor(context)]

        # 展平
        x = emb.view(1, -1)

        # forward
        for layer in layers:
            x = layer(x)

        # x现在就是logits
        probs = F.softmax(x, dim=1)

        # 按概率采样字符
        ix = torch.multinomial(
            probs,
            num_samples=1,
            generator=g
        ).item()

        # 更新上下文
        context = context[1:] + [ix]

        # 保存字符
        out.append(itos[ix])

        # 遇到结束符
        if ix == 0:
            break

    print(''.join(out))
