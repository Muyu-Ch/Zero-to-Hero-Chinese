import torch
import torch.nn.functional as F
import random

#导入数据集
words=open("names.txt","r").read().splitlines()

#建立字符表
chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}

#固定参数
vocab_size=27
g = torch.Generator().manual_seed(2147483647)

#旋钮
block_size=3
n_embd=40
n_steps=100000
batch_size=50
lr=0.03
layer_size=[block_size*n_embd,800,100,vocab_size]


#构建数据集
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

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1],block_size)
Xdev, Ydev = build_dataset(words[n1:n2],block_size)
Xte, Yte = build_dataset(words[n2:],block_size)

#超参数
#嵌入层
C = torch.randn((vocab_size, n_embd), generator=g)#嵌入层

#神经网络层
Ws=[]
bs=[]
for n1,n2 in zip(layer_size[:-1],layer_size[1:]):
    Ws.append(torch.randn((n1,n2), generator=g))
    bs.append(torch.randn(n2,generator=g))
    
parameters = [C]+Ws+bs

print(f"总参数量：{sum(p.nelement() for p in parameters)}")
for p in parameters:
    p.requires_grad=True

#训练
for i in range(n_steps):
    
    #前向传播
    ix=torch.randint(1,Xtr.shape[0],(batch_size,))
    emb = C[Xtr[ix]] 
    x = emb.view(-1, block_size * n_embd)
    for W, b in zip(Ws[:-1], bs[:-1]):
        x = torch.tanh(x @ W + b)     
    logits = x @ Ws[-1] + bs[-1]                    
    loss=F.cross_entropy(logits,Ytr[ix])
    
    #清空梯度
    for p in parameters:
        p.grad=None
    
    #反向传播
    loss.backward()

    #更新梯度
    for p in parameters:
        p.data+=-lr*p.grad

emb = C[Xtr] 
x = emb.view(-1, block_size*n_embd)
for W, b in zip(Ws[:-1], bs[:-1]):
    x = torch.tanh(x @ W + b)     
logits = x @ Ws[-1] + bs[-1]               
loss=F.cross_entropy(logits,Ytr)
print(f"训练集loss={loss.item()}")


emb = C[Xdev] 
x = emb.view(-1, block_size*n_embd)
for W, b in zip(Ws[:-1], bs[:-1]):
    x = torch.tanh(x @ W + b)     
logits = x @ Ws[-1] + bs[-1]                  
loss=F.cross_entropy(logits,Ydev)
print(f"验证集loss={loss.item()}")

#接下来既然我们已经生成了模型，那下一步要做的事情就是要用我们的模型来生成姓名
g = torch.Generator().manual_seed(2147483647+10)#换个种子

for i in range(20):#生成20个名字
    context=[0]*block_size#上下文。由于我们所有名字的开头全部都是一串点，所以我们可以直接用这个
    out=[]#输出

    while True:#用循环来生成姓名，直到出现'.'
        emb = C[torch.tensor(context)] 
        x = emb.view(-1, block_size*n_embd)
        for W, b in zip(Ws[:-1], bs[:-1]):
            x = torch.tanh(x @ W + b)     
        logits = x @ Ws[-1] + bs[-1] #一样计算出logits
        probs=F.softmax(logits,dim=1)#loss的计算用的是cross_entropy，现在用softmax计算probs
        ix=torch.multinomial(probs,num_samples=1,generator=g).item()#根据概率输出词汇
        context=context[1:]+[ix]
        out.append(itos[ix])
        if ix==0:
            break
    print(''.join(s for s in out))