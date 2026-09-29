import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters 超参数
batch_size = 64 # how many independent sequences will we process in parallel? 并行处理多少个独立的序列？
block_size = 256 # what is the maximum context length for predictions? 预测时最大的上下文长度是多少？
max_iters = 5000 # 最大训练迭代次数
eval_interval = 500 # 每隔多少次迭代评估一次损失
learning_rate = 3e-4 # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 有 GPU 就用 GPU，否则用 CPU
eval_iters = 200 # 评估时平均多少个 batch 的损失
n_embd = 384 # 嵌入向量（特征）的维度
n_head = 6 # 自注意力头的数量
n_layer = 6 # Transformer 块的层数
dropout = 0.2 # dropout 丢弃概率
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text 这里是这段文本中出现的所有不重复字符
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers 建立字符到整数的映射
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers 编码器：输入一个字符串，输出一个整数列表
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string 解码器：输入一个整数列表，输出一个字符串

# Train and test splits 划分训练集和验证集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val 前 90% 作为训练集，其余作为验证集
train_data = data[:n]
val_data = data[n:]

# data loading 数据加载
def get_batch(split):
    # generate a small batch of data of inputs x and targets y 生成一小批数据，包括输入 x 和目标 y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention 自注意力机制中的单个注意力头 """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels) 输入尺寸为 (batch, 时间步, 通道数)
        # output of size (batch, time-step, head size) 输出尺寸为 (batch, 时间步, 头维度)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities") 计算注意力分数（“亲和度”）
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values 对 value 做加权聚合
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel 并行运行的多个自注意力头 """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity 一个简单的线性层，后接非线性激活函数 """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation Transformer 块：先通信（注意力），再计算（前馈网络） """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like n_embd 是嵌入维度，n_head 是我们想要的注意力头数量
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table 每个 token 直接从查找表中读出下一个 token 的 logits
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm 最后一层 LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video 更好的参数初始化，原版 GPT 视频里没讲，但很重要，会在后续视频中讲解
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers idx 和 targets 都是形状为 (B,T) 的整数张量
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context idx 是当前上下文的索引数组，形状为 (B, T)
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens 把 idx 裁剪到最近的 block_size 个 token
            idx_cond = idx[:, -block_size:]
            # get the predictions 获取预测结果
            logits, loss = self(idx_cond)
            # focus only on the last time step 只关注最后一个时间步
            logits = logits[:, -1, :] # becomes (B, C) 变成 (B, C)
            # apply softmax to get probabilities 用 softmax 得到概率分布
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution 从该分布中采样
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence 把采样到的索引追加到当前序列末尾
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model 打印模型的参数量
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer 创建一个 PyTorch 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets 每隔一段时间在训练集和验证集上评估损失
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data 采样一批数据
    xb, yb = get_batch('train')

    # evaluate the loss 计算损失
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model 用模型生成文本
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
