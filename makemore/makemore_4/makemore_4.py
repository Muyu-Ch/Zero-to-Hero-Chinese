"""
MakeMore Part 4: 手动反向传播 (backprop ninja) 名字生成模型
=========================================================

Part 3 用 BatchNorm 把 loss 压到 2.1 附近, 但梯度全靠 loss.backward() 自动求。
Part 4 把自动求导拿掉: 每个中间量的梯度都手写公式算出来, 只留前向 + 手写反向。

结构: 上下文(3字符) -> 嵌入表 C -> 线性 -> BatchNorm -> tanh -> 线性 -> 27 个分数

流程: 构建数据集 -> 搭网络 -> 手动训练 -> 生成 20 个名字

参考: Andrej Karpathy - Neural Networks: Zero to Hero (makemore Part 4)
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------
# 0. 数据准备
# ---------------------------------------------------------------
words=open("../names.txt","r").read().splitlines()
chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}

vocab_size=len(itos)
print(f'[数据] 词表大小 vocab_size = {vocab_size} (26 个字母 + 1 个 ".")')

block_size=3
print(f'[数据] 上下文长度 block_size = {block_size} (每个样本看前 {block_size} 个字符)')

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
  print(f'[数据] X {tuple(X.shape)} | Y {tuple(Y.shape)}')#打印数据集的大小
  return X, Y

import random
# 80/10/10 切分: 训练集(更新参数) / 验证集(调旋钮时反复看) / 测试集(只碰一次)
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1],block_size)#0%~80%
Xdev, Ydev = build_dataset(words[n1:n2],block_size)#80%~90%
Xte, Yte = build_dataset(words[n2:],block_size)#90~100%

#所以接下来我们可以用我们的算法了

# ---------------------------------------------------------------
# 2. 模型参数 (旋钮: n_embd 嵌入维度 / n_hidden 隐藏层宽度)
# ---------------------------------------------------------------
g = torch.Generator().manual_seed(2147483647)

n_embd=10
n_hidden=200

C = torch.randn((vocab_size, n_embd),          generator=g)         
W1 = torch.randn((block_size*n_embd,n_hidden), generator=g)*(5/3)/((block_size*n_embd)**0.5)
b1 = torch.randn(n_hidden,                     generator=g)*0.1  
W2 = torch.randn((n_hidden,vocab_size),        generator=g)*0.1  
b2 = torch.randn(vocab_size,                   generator=g)*0.1      

bngain=torch.ones((1,n_hidden))*0.1+1.0
bnbias=torch.zeros((1,n_hidden))*0.1

parameters = [C, W1, b1, W2, b2,bngain,bnbias]
print(f'[模型] 形状 C{tuple(C.shape)} W1{tuple(W1.shape)} b1{tuple(b1.shape)} '
      f'W2{tuple(W2.shape)} b2{tuple(b2.shape)} bngain{tuple(bngain.shape)} bnbias{tuple(bnbias.shape)}')
print(f'[模型] 总参数量 {sum(p.nelement() for p in parameters)}')
for p in parameters:
    p.requires_grad=True

batch_size=32
n=batch_size
max_steps=200000
lossi=[]

# ---------------------------------------------------------------
# 3. 手动训练: 前向照常, 把 loss.backward() 换成手写反向公式
# ---------------------------------------------------------------
print(f'[训练] {max_steps} 步, batch {batch_size}, lr 0.1 -> 0.01 (第 100000 步衰减)')
print('[训练] 整段包在 no_grad 里: 不用 autograd, 梯度全靠手算')

with torch.no_grad():#不使用pytorch的反向传播了
    for i in range(max_steps):
        ix=torch.randint(0,Xtr.shape[0],(batch_size,),generator=g)
        Xb,Yb=Xtr[ix],Ytr[ix]
        
        emb = C[Xb]
        embcat=emb.view(batch_size, block_size*n_embd)
        hprebn=embcat@ W1 + b1#h pre bn 在batchnorm之前的h
        
        #BatchNorm Layer
        bnmean=hprebn.mean(0,keepdim=True)#这里就是求mean
        bnvar=hprebn.var(0,keepdim=True,unbiased=True)
        bnvar_inv=(bnvar+1e-5)**-0.5
        bnraw=(hprebn-bnmean)*bnvar_inv
        hpreact=bngain*bnraw+bnbias
    
        
        h=torch.tanh(hpreact)
        logits=h@W2+b2 
        loss=F.cross_entropy(logits,Yb)
        
        for p in parameters:
            p.grad=None
        #loss.backward()
        #不使用backward了！
        
        # 手动反向传播！
        # ---------------------------
        dlogits = F.softmax(logits, 1)
        dlogits[range(n), Yb] -= 1
        dlogits /= n
        # 2nd layer backprop
        dh = dlogits @ W2.T
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)
        # tanh
        dhpreact = (1.0 - h**2) * dh
        # batchnorm backprop
        dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
        dbnbias = dhpreact.sum(0, keepdim=True)
        dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
        # 1st layer
        dembcat = dhprebn @ W1.T
        dW1 = embcat.T @ dhprebn
        db1 = dhprebn.sum(0)
        # embedding
        demb = dembcat.view(emb.shape)
        dC = torch.zeros_like(C)
        for k in range(Xb.shape[0]):
          for j in range(Xb.shape[1]):
            ix = Xb[k,j]
            dC[ix] += demb[k,j]
        grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
        # ---------------------------
    
        # update
        lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
        for p, grad in zip(parameters, grads):
          #p.data += -lr * p.grad #torch.backward()
          p.data += -lr * grad #手动方法
        
        # track stats
        if i % 10000 == 0: # print every once in a while
          print(f'  第 {i:6d}/{max_steps} 步 minibatch loss = {loss.item():.4f}')
        lossi.append(loss.log10().item())

print(f'[训练] 结束, 最后一步 minibatch loss = {loss.item():.4f}')

# ---------------------------------------------------------------
# 4. 生成 20 个名字: 滑动窗口滚出新字符, 直到抽到 '.'
# ---------------------------------------------------------------
g=torch.Generator().manual_seed(2147483647+10) #换个种子, 生成新鲜的名字

print('[生成] 20 个名字:')
for _ in range(20):
  out = []
  context = [0] * block_size # initialize with all ...
  while True:
    # ----------
    # forward pass:
    # Embedding
    emb = C[torch.tensor([context])] # (1,block_size,d)
    embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
    hpreact = embcat @ W1 + b1
    # 这里用的 bnmean / bnvar 是训练最后一轮 batch 算出来的 (近似做法, 严格说该用全量统计量)
    hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
    h = torch.tanh(hpreact) # (N, n_hidden)
    logits = h @ W2 + b2 # (N, vocab_size)
    # ----------
    # Sample
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    context = context[1:] + [ix]
    out.append(ix)
    if ix == 0:
      break

  print('  ' + ''.join(itos[i] for i in out))
