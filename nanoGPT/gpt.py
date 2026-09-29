# ============================================================
# Andrej Karpathy「Let's build GPT」视频中的完整 GPT 实现
# 建议阅读顺序：
#   超参数 → 数据准备 → Head（单头注意力）→ MultiHeadAttention（多头）
#   → FeedFoward（前馈）→ Block（Transformer 块）→ GPTLanguageModel（整体）→ 训练循环
# ============================================================

import torch
# 导入 PyTorch 主库：张量（Tensor）运算、自动求导、GPU 支持都在这里

import torch.nn as nn
# 导入神经网络模块：nn.Module（所有层的基类）、nn.Linear、nn.Embedding、nn.LayerNorm 等

from torch.nn import functional as F
# 导入函数式接口：softmax、cross_entropy 这类没有可学习参数的运算，习惯简写成 F

# hyperparameters 超参数
# 下面这些是训练前就定好的配置，决定模型多大、训练多久

batch_size = 64 # how many independent sequences will we process in parallel? 并行处理多少个独立的序列？
# 每次训练同时处理 64 条独立的文本序列（一个 batch 里的样本数）

block_size = 256 # what is the maximum context length for predictions? 预测时最大的上下文长度是多少？
# 每条序列的长度上限（上下文窗口）：模型一次最多能看到 256 个字符的历史

max_iters = 5000 # 最大训练迭代次数
# 训练循环总共跑 5000 步，每一步处理一个 batch

eval_interval = 500 # 每隔多少次迭代评估一次损失
# 每训练 500 步，就在训练集和验证集上评估一次

learning_rate = 3e-4 # 学习率
# 学习率 0.0003，即每一步参数更新的步长。对这类小 GPT，3e-4 是比较安全的默认值

device = 'cuda' if torch.cuda.is_available() else 'cpu' # 有 GPU 就用 GPU，否则用 CPU
# 三元表达式（条件 ? A : B 的 Python 写法）：检测到 NVIDIA GPU 就用 CUDA，否则退回 CPU

eval_iters = 200 # 评估时平均多少个 batch 的损失
# 评估时连续抽 200 个 batch 求平均：单次 loss 噪声太大，平均后曲线才平滑可靠

n_embd = 384 # 嵌入向量（特征）的维度
# 每个 token 用 384 个数字表示，也就是代码里到处出现的通道数 C

n_head = 6 # 自注意力头的数量
# 多头注意力拆成 6 个头并行计算，所以每个头的维度 = 384 / 6 = 64

n_layer = 6 # Transformer 块的层数
# 把 Transformer Block 堆叠 6 层，层数越深、表达能力越强

dropout = 0.2 # dropout 丢弃概率
# 训练时随机丢弃 20% 的中间激活值，起正则化作用，防止过拟合

# ------------

torch.manual_seed(1337)
# 固定全局随机数种子：让每次运行的参数初始化、数据采样都一致，结果可复现

with open('input.txt', 'r', encoding='utf-8') as f:
    # with 是上下文管理器：打开文件，用完后自动关闭（即使中途报错也会关）
    text = f.read()
    # 一次性把整个数据集读成一个大字符串（这里是 tinyshakespeare，约 111 万字符）

# here are all the unique characters that occur in this text 这里是这段文本中出现的所有不重复字符
chars = sorted(list(set(text)))
# set(text) 去重 → list() 转成列表 → sorted() 排序，得到文本里出现过的所有不重复字符（约 65 个）

vocab_size = len(chars)
# 词表大小 = 不重复字符的个数，也就是模型每个位置要预测的类别数

# create a mapping from characters to integers 建立字符到整数的映射
stoi = { ch:i for i,ch in enumerate(chars) }
# 字典推导式：字符 → 整数编号（string to integer），如 '\n':0, ' ':1, '!':2 ...

itos = { i:ch for i,ch in enumerate(chars) }
# 反过来：整数编号 → 字符（integer to string）

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers 编码器：输入一个字符串，输出一个整数列表
# lambda 是匿名函数：把字符串里每个字符查表换成整数，得到整数列表

decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string 解码器：输入一个整数列表，输出一个字符串
# 把整数列表查表拼回字符串。''.join(...) 表示用空字符串作为连接符，即直接首尾拼接

# Train and test splits 划分训练集和验证集
data = torch.tensor(encode(text), dtype=torch.long)
# 把整段文本编码成一个一维张量。dtype=torch.long 即 int64，因为 Embedding 的索引必须是整数类型

n = int(0.9*len(data)) # first 90% will be train, rest val 前 90% 作为训练集，其余作为验证集
# 算出切分位置（90% 处），int() 向下取整

train_data = data[:n]
# 前 90% 用于训练

val_data = data[n:]
# 后 10% 用于验证：模型没见过这些文本，用它来判断有没有过拟合

# data loading 数据加载
def get_batch(split):
    # 随机抽一小批 (输入 x, 目标 y) 数据对
    # generate a small batch of data of inputs x and targets y 生成一小批数据，包括输入 x 和目标 y
    data = train_data if split == 'train' else val_data
    # 参数 split 是 'train' 或 'val'，三元表达式决定从哪个数据集里抽
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 随机抽 64 个起始位置，每个在 [0, len(data)-block_size) 之间
    # 上界减去 block_size 是为了保证后面 i+block_size+1 不会越界
    x = torch.stack([data[i:i+block_size] for i in ix])
    # 每个起点取连续 256 个字符作为输入；torch.stack 把 64 个长度 256 的向量叠成 (64, 256)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # y 就是 x 整体右移一位：x 在位置 t 的目标，是原文的「下一个字符」x[t+1]
    # 所以 y 的每一行是同一段文本往后错一格，语言模型学的正是「给定前文预测下一个」
    x, y = x.to(device), y.to(device)
    # 把数据搬到和模型相同的设备（GPU 或 CPU）上，否则计算时会报设备不匹配
    return x, y
    # 返回输入 (64, 256) 和目标 (64, 256)

@torch.no_grad()
# 装饰器：这个函数内部不构建计算图、不计算梯度，省显存也更快（评估不需要反向传播）
def estimate_loss():
    # 在训练集和验证集上各抽 200 个 batch，估计当前的平均损失
    out = {}
    # 用字典装两种 split 的结果
    model.eval()
    # 切换到「评估模式」：关闭 dropout（推理时不应该随机丢神经元）。这是全局状态
    for split in ['train', 'val']:
        # 训练集和验证集各评估一遍
        losses = torch.zeros(eval_iters)
        # 建一个长度 200 的全 0 张量，准备装每个 batch 的 loss
        for k in range(eval_iters):
            # 循环 200 次
            X, Y = get_batch(split)
            # 抽一个 batch
            logits, loss = model(X, Y)
            # 前向传播算 loss（此时 loss 是 0 维张量，还挂在计算图上）
            losses[k] = loss.item()
            # .item() 把单元素张量取出成 Python 浮点数，同时脱离计算图
        out[split] = losses.mean()
        # 200 个 batch 的 loss 求平均，作为这个 split 的损失估计
    model.train()
    # 切回「训练模式」，恢复 dropout（这步千万别忘了）
    return out
    # 返回形如 {'train': tensor(...), 'val': tensor(...)}

class Head(nn.Module):
    """ one head of self-attention 自注意力机制中的单个注意力头 """

    # 继承 nn.Module，PyTorch 才能自动管理它的参数，支持 .to(device)、.parameters()、.train() 等

    def __init__(self, head_size):
        # head_size 是这个头输出的维度，这里 = n_embd // n_head = 384 // 6 = 64
        super().__init__()
        # 先初始化父类 nn.Module，这必须是 __init__ 的第一步
        self.key = nn.Linear(n_embd, head_size, bias=False)
        # key 线性层：把 384 维输入投影成 64 维的「键」；bias=False 是因为后面接 LayerNorm，偏置会被抵消掉
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # query 线性层：投影成 64 维的「查询」
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # value 线性层：投影成 64 维的「值」
        # 同一个 x 经过三个不同矩阵，扮演三种角色：我在找什么(q)、我有什么(k)、我能提供什么(v)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # torch.ones(256,256) 全 1 矩阵，torch.tril 取「下三角」（含对角线），其余位置为 0
        # 这就是因果掩码：第 t 行只有前 t+1 个位置是 1，用来禁止看到未来
        # register_buffer 把它注册为缓冲区：会跟着 .to(device) 移动、会存进 state_dict，但不算可训练参数

        self.dropout = nn.Dropout(dropout)
        # dropout 层，作用在注意力权重上

    def forward(self, x):
        # x 形状 (B, T, C) = (64, 256, 384)
        # input of size (batch, time-step, channels) 输入尺寸为 (batch, 时间步, 通道数)
        # output of size (batch, time-step, head size) 输出尺寸为 (batch, 时间步, 头维度)
        B,T,C = x.shape
        # 解包出 batch 大小、时间步（token 数）、通道数（特征维度）
        k = self.key(x)   # (B,T,hs)
        # 算出 key，形状 (64, 256, 64)
        q = self.query(x) # (B,T,hs)
        # 算出 query，形状 (64, 256, 64)
        # compute attention scores ("affinities") 计算注意力分数（“亲和度”）
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # q @ k^T：每个位置 t 的 query 与每个位置 s 的 key 做点积，衡量两者有多「相关」，结果形状 (64, 256, 256)
        # transpose(-2,-1) 交换最后两维，相当于对每个 batch 单独把 (T,hs) 转置成 (hs,T)
        # 乘 k.shape[-1]**-0.5 即 1/sqrt(64)=1/8 做缩放：维度越大点积数值越大，softmax 会变得极端尖锐、梯度消失
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # 把「上三角」（未来位置）的分数全填成 -inf，这样 softmax 之后那些位置的权重正好是 0
        # 用 [:T,:T] 切片是因为 block_size(256) 可能大于当前实际序列长度 T，(B,T,T) 需要能对得上
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # 沿最后一维做 softmax：每个位置对所有历史位置的权重归一化，每一行和为 1
        # 填过 -inf 的位置 e^(-inf)=0，完全不影响结果
        wei = self.dropout(wei)
        # 对注意力权重做 dropout，进一步正则化
        # perform the weighted aggregation of the values 对 value 做加权聚合
        v = self.value(x) # (B,T,hs)
        # 算出 value，形状 (64, 256, 64)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        # 用权重矩阵对 value 加权求和：(64,256,256) @ (64,256,64) → (64,256,64)
        # 效果是：每个位置都汇聚了它「允许看到」的历史信息，权重由相似度决定
        return out
        # 返回 (B, T, 64)，即这个头对每个位置的输出

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel 并行运行的多个自注意力头 """

    # 为什么要多头：一个头只能学到一种「关注模式」，多个头可以并行关注不同模式（如语法、指代、位置）
    # 然后把各自的结果拼起来，让模型自己决定怎么用

    def __init__(self, num_heads, head_size):
        # num_heads=6 个头，每个头 head_size=64 维
        super().__init__()
        # 初始化父类
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 创建 6 个结构相同但参数独立的 Head
        # 必须用 nn.ModuleList 而不是普通 list，否则 PyTorch 不会注册这些子模块的参数（也不会同步到 GPU）
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        # 输出投影层：64*6=384 → 384，把拼接结果融合，并投影回残差路径需要的维度
        self.dropout = nn.Dropout(dropout)
        # 输出上的 dropout

    def forward(self, x):
        # x: (B, T, 384)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 6 个头各自输出 (B,T,64)，沿最后一维拼接成 (B,T,384)
        out = self.dropout(self.proj(out))
        # 先投影融合再做 dropout
        return out
        # 返回 (B, T, 384)，形状与输入一致，方便做残差相加

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity 一个简单的线性层，后接非线性激活函数 """

    # 注意类名 FeedFoward 是原作者拼错了（少一个 r），为保持和视频一致没有改动

    def __init__(self, n_embd):
        super().__init__()
        # 初始化父类
        self.net = nn.Sequential(
            # Sequential 把下面的层按顺序串起来，调用时自动依次执行
            nn.Linear(n_embd, 4 * n_embd),
            # 升维：384 → 1536（4 倍，这是 Transformer 论文的惯例）
            nn.ReLU(),
            # 非线性激活：负数变 0，正数不变。没有它，两层线性层叠加仍等价于一层线性层
            nn.Linear(4 * n_embd, n_embd),
            # 降维回来：1536 → 384
            nn.Dropout(dropout),
            # dropout 正则化
        )

    def forward(self, x):
        # x: (B, T, 384)
        return self.net(x)
        # 返回 (B, T, 384)
        # 如果说注意力层是让不同 token「互相交流」，那这一层就是每个 token 各自「独立思考」

class Block(nn.Module):
    """ Transformer block: communication followed by computation Transformer 块：先通信（注意力），再计算（前馈网络） """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like n_embd 是嵌入维度，n_head 是我们想要的注意力头数量
        super().__init__()
        # 初始化父类
        head_size = n_embd // n_head
        # 每个头的维度：384 // 6 = 64（// 是整除，向下取整）
        self.sa = MultiHeadAttention(n_head, head_size)
        # 子层 1：多头自注意力（负责 token 之间的「通信」）
        self.ffwd = FeedFoward(n_embd)
        # 子层 2：前馈网络（负责每个 token 自己的「计算」）
        self.ln1 = nn.LayerNorm(n_embd)
        # LayerNorm：对每个 token 的 384 维特征做归一化（均值 0、方差 1），让训练更稳定
        self.ln2 = nn.LayerNorm(n_embd)
        # 第二个 LayerNorm，用在进前馈网络之前

    def forward(self, x):
        # x: (B, T, 384)
        x = x + self.sa(self.ln1(x))
        # 先 LayerNorm，再进注意力，最后和原 x 相加 —— 这就是残差连接
        # 残差让梯度可以「抄近路」直接回传，是深层网络能训起来的关键；也要求子层输出维度和 x 一致
        x = x + self.ffwd(self.ln2(x))
        # 同样结构：LayerNorm → 前馈网络 → 残差相加
        # 这种「先归一化再进子层」的写法叫 Pre-Norm，比原论文的 Post-Norm 更容易训练
        return x
        # 输出 (B, T, 384)，形状不变，所以 Block 可以任意层数地堆叠

class GPTLanguageModel(nn.Module):
    # 完整的 GPT 语言模型：嵌入 + 6 个 Block + 输出头

    def __init__(self):
        super().__init__()
        # 初始化父类
        # each token directly reads off the logits for the next token from a lookup table 每个 token 直接从查找表中读出下一个 token 的 logits
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # token 嵌入表：(65, 384)。输入 token 编号，输出对应的 384 维向量
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # 位置嵌入表：(256, 384)。第 t 个位置有自己专属的一行向量
        # 为什么需要位置信息：注意力本质是「对集合加权求和」，本身不区分顺序，必须显式把位置喂进去
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # 生成 6 个 Block 并按顺序串联；* 把列表解包成一个个位置参数传给 Sequential
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm 最后一层 LayerNorm
        # 所有 Block 之后的最后一次归一化
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # 语言模型头：384 → 65，把每个位置的特征映射成「下一个字符是各个字符」的原始分数（logits）

        # better init, not covered in the original GPT video, but important, will cover in followup video 更好的参数初始化，原版 GPT 视频里没讲，但很重要，会在后续视频中讲解
        self.apply(self._init_weights)
        # self.apply 会递归地把 _init_weights 作用到每一个子模块上，覆盖 PyTorch 的默认初始化

    def _init_weights(self, module):
        # 函数名结尾的下划线是 PyTorch 的约定，表示「原地操作」或「内部使用」
        if isinstance(module, nn.Linear):
            # 只处理线性层
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 权重用「均值 0、标准差 0.02」的正态分布初始化（初始值太大会让 logits 爆炸、loss 下不去）
            if module.bias is not None:
                # 如果这层有偏置
                torch.nn.init.zeros_(module.bias)
                # 偏置初始化为 0
        elif isinstance(module, nn.Embedding):
            # 嵌入层做同样处理
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 嵌入表也用正态分布初始化

    def forward(self, idx, targets=None):
        # idx: (B, T) 的整数 token 编号；targets 可选，生成文本时不传
        B, T = idx.shape
        # 取出 batch 大小和序列长度

        # idx and targets are both (B,T) tensor of integers idx 和 targets 都是形状为 (B,T) 的整数张量
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # 查 token 嵌入表：每个整数变成一个 384 维向量，形状 (B, T, 384)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        # torch.arange(T) 生成 [0,1,...,T-1] 作为位置编号，查表得到 (T, 384)
        # 注意 device=device：位置编号张量也必须和模型在同一设备上
        x = tok_emb + pos_emb # (B,T,C)
        # 两者相加；广播机制会把 (T,384) 自动扩展到每个 batch 上，得到 (B,T,384)
        x = self.blocks(x) # (B,T,C)
        # 依次通过 6 个 Block，token 之间不断交换信息
        x = self.ln_f(x) # (B,T,C)
        # 最后归一化一次
        logits = self.lm_head(x) # (B,T,vocab_size)
        # 映射成每个位置对 65 个字符的打分，形状 (B, T, 65)

        if targets is None:
            # 没传目标（生成模式）
            loss = None
            # 就不计算损失
        else:
            # 传了目标（训练模式），计算交叉熵损失
            B, T, C = logits.shape
            # 重新取形状，此时 C = vocab_size = 65
            logits = logits.view(B*T, C)
            # 展平成 (B*T, 65)，因为 cross_entropy 要求输入是 (N, C) 的二维张量
            targets = targets.view(B*T)
            # 目标展平成 (B*T,)，与上面对应
            loss = F.cross_entropy(logits, targets)
            # 交叉熵：内部会自动做 log_softmax，所以传进来的必须是没归一化的原始 logits
            # 每个位置的预测各贡献一份损失，最终取平均

        return logits, loss
        # 返回打分和损失（生成时 loss 为 None）

    def generate(self, idx, max_new_tokens):
        # 自回归生成：一次预测一个 token，拼到序列末尾，再整段喂回去预测下一个
        # idx is (B, T) array of indices in the current context idx 是当前上下文的索引数组，形状为 (B, T)
        for _ in range(max_new_tokens):
            # 循环 max_new_tokens 次，每次生成一个字符
            # crop idx to the last block_size tokens 把 idx 裁剪到最近的 block_size 个 token
            idx_cond = idx[:, -block_size:]
            # 只保留最后 256 个 token：位置嵌入表只有 256 行，而且上下文越长计算越慢
            # get the predictions 获取预测结果
            logits, loss = self(idx_cond)
            # 前向传播（不传 targets，所以 loss 是 None）
            # focus only on the last time step 只关注最后一个时间步
            logits = logits[:, -1, :] # becomes (B, C) 变成 (B, C)
            # 只取最后一个位置的预测结果，因为只有它代表「序列接下来该是什么」的分布
            # apply softmax to get probabilities 用 softmax 得到概率分布
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 把原始打分变成概率（非负、总和为 1）
            # sample from the distribution 从该分布中采样
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 按概率随机抽一个字符，形状 (B, 1)
            # 注意不是取 argmax：随机采样才能让生成结果有多样性，否则每次输出都一样
            # append sampled index to the running sequence 把采样到的索引追加到当前序列末尾
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            # 沿时间维拼接，序列长度 +1，下一轮循环继续用更长的上下文预测
        return idx
        # 返回生成好的完整序列 (B, T+max_new_tokens)

model = GPTLanguageModel()
# 实例化模型（内部会执行上面自定义的那套参数初始化）
m = model.to(device)
# 把模型所有参数搬到 device 上；m 和 model 是同一个对象，只是取个短名字
# print the number of parameters in the model 打印模型的参数量
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
# p.numel() 是这个参数张量里的元素个数（number of elements）；全部求和后除以 1e6 换算成「百万」为单位

# create a PyTorch optimizer 创建一个 PyTorch 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# AdamW 优化器，负责根据梯度更新参数；model.parameters() 告诉它要更新哪些张量

for iter in range(max_iters):
    # 主训练循环，共 5000 步
    # 注意 iter 这个名字覆盖了 Python 内置函数 iter()，这是原作者的写法

    # every once in a while evaluate the loss on train and val sets 每隔一段时间在训练集和验证集上评估损失
    if iter % eval_interval == 0 or iter == max_iters - 1:
        # 每 500 步评估一次；另外最后一步（4999）也评估，方便看到最终效果
        losses = estimate_loss()
        # 得到 {'train': ..., 'val': ...} 两个平均损失
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # f-string 格式化输出；:.4f 表示保留 4 位小数

    # sample a batch of data 采样一批数据
    xb, yb = get_batch('train')
    # 抽 64 条长度为 256 的序列作为输入 xb 和目标 yb

    # evaluate the loss 计算损失
    logits, loss = model(xb, yb)
    # 前向传播，得到 loss（标量张量）
    optimizer.zero_grad(set_to_none=True)
    # 清空上一步攒下的梯度。PyTorch 的梯度默认是「累加」的，不清零就会越加越大
    # set_to_none=True 把 .grad 设成 None 而不是全 0，更省显存也更快
    loss.backward()
    # 反向传播：自动求出 loss 对每个参数的梯度，存进各参数的 .grad 属性
    optimizer.step()
    # 用梯度更新所有参数 —— 这一步才是真正的「学习」

# generate from the model 用模型生成文本
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# 生成起点：batch=1、长度=1，内容是 0 —— 0 号字符就是换行符 '\n'
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# generate 返回 (1, 501) → [0] 取出第一个（也是唯一一个）batch 得到 (501,) → tolist() 转成 Python 列表 → decode 转回文字
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
# 被注释掉的一行：生成 10000 个字符并写入 more.txt
