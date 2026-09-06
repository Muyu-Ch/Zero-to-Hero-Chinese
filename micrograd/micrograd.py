"""
micrograd —— 从零手写反向传播引擎 + 小型 MLP
===========================================

Value: 一个会记录"自己是怎么被算出来的"的标量
   每个运算 (+, -, *, /, **, exp, tanh) 都会:
   1. 算出数值
   2. 记住参与运算的子节点 (构成计算图)
   3. 带上局部求导规则 (_backward, 即链式法则的一环)

   调用 loss.backward() 时:
   拓扑排序整张计算图 -> 从输出出发, 梯度一路传回所有叶子参数

Neuron / Layer / MLP: 用 Value 搭出来的小型神经网络
   训练四步曲 = 前向传播 -> 算 loss -> 清零梯度 -> 反向传播 -> 更新参数

两个演示:
1. 示例模型: 4 组数据训练 3x4x4x1 的 MLP 做"二进制预测"
2. 除法模型: 输入 a 和 b, 让模型学出 a/b (自己设计的实验)

参考: Andrej Karpathy - Neural Networks: Zero to Hero (micrograd)
"""

import math
import random

# ===============================================================
# 0. Value: 反向传播引擎的核心
# ===============================================================
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data            # 数值
        self.grad = 0.0             # 梯度 (先默认 0, 反向传播时累加进来)
        self._prev = set(_children) # 计算图: 记录自己由哪些节点算出来
        self._op = _op              # 运算符号
        self.label = label          # 标签
        self._backward = lambda: None

    # 更直观地展示 Value
    def __repr__(self):
        return f"Value(data={self.data})"

    # a + b; a + 2 (自动把 int/float 包装成 Value)
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # 加法: 导数恒为 1, 梯度原样传给两个输入
            # 用 += 而不是 =, 因为一个节点可能被多个路径使用 (梯度累加)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    # 2 + a
    def __radd__(self, other):
        return self + other

    # a - b
    def __sub__(self, other):
        return self + (-other)

    # a * b; a * 2
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # 乘法: d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # 2 * a
    def __rmul__(self, other):
        return self * other

    # a / b = a * (b ** -1)
    def __truediv__(self, other):
        return self * (other ** -1)

    # a ** k (k 是 int 或 float)
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # 幂函数: d(x^k)/dx = k * x^(k-1)
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out

    # e^x
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x = 输出本身
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    # 压缩函数: 把任意实数压进 (-1, 1), 作为神经元的激活函数
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh)/dx = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # 反向传播: 拓扑排序整张计算图, 从自己出发把梯度传回所有叶子
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0                # 输出的梯度是 1 (dL/dL)
        for node in reversed(topo):    # 从输出到输入, 依次执行局部求导
            node._backward()


# ===============================================================
# 1. 用 Value 搭建神经网络: Neuron -> Layer -> MLP
# ===============================================================
class Neuron:  # 单个神经元
    def __init__(self, nin):  # 输入的数量
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # 均匀随机初始化
        self.b = Value(random.uniform(-1, 1))                        # 偏置

    def __call__(self, x):
        # 加权求和 (x 可以是 Value, 也可以是 int/float, Value 会自动包装)
        act = sum((xi * wi for xi, wi in zip(self.w, x)), self.b)
        return act.tanh()  # 过激活函数

    def parameters(self):
        return self.w + [self.b]


class Layer:  # 单层神经元
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]  # nout 个神经元, 共享同一组输入

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # 只有 1 个神经元时返回单个 Value 而不是列表 (自己的修改, 让最后一层用起来更顺手)
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts  # 每层的规模, 如 [3, 4, 4, 1] = 3 输入 -> 4 -> 4 -> 1 输出
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)    # 一层一层串起来, 最后一层的输出就是结果
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ===============================================================
# 演示一: 示例模型 —— 4 组数据的"二进制预测"
# ===============================================================
# xs 是四组输入, ys 是四组答案
# 目标: 训练一个神经网络, 让它接收 xs 之后能预测出对应的 ys
# 就像 LLM 用大量的输入输出组合去训练模型
print('示例模型 =============================================')
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]
n = MLP(3, [4, 4, 1])   # 3 输入 -> 4 -> 4 -> 1 输出
params = n.parameters()
ypred = [n(x) for x in xs]
print('xs   :', xs)
print('ys   :', ys)
print('ypred:', [round(y.data, 4) for y in ypred], ' ← 随机初始化, 离正确答案很远')

# 训练四步曲: ①前向 ②清零梯度 ③反向 ④更新
h = 0.1                 # 学习率
for k in range(20):
    # ① 前向传播
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
    # ② 清零梯度 (⚠️ 不清零梯度会一直累加, 这是 micrograd 踩过的大坑)
    for param in params:
        param.grad = 0.0
    # ③ 反向传播
    loss.backward()
    # ④ 沿梯度方向更新参数
    for param in params:
        param.data -= param.grad * h
    print(f'第 {k:2d} 轮 loss = {loss.data:.6f}')

# 继续训练, 直到 loss < 0.0001
while True:
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
    if loss.data < 0.0001:
        break
    for param in params:
        param.grad = 0.0
    loss.backward()
    for param in params:
        param.data -= param.grad * h
print(f'训练完成, loss = {loss.data:.6f} < 0.0001')
print('ys   :', ys)
print('ypred:', [round(y.data, 4) for y in ypred])
print('训练成功')
print()


# ===============================================================
# 演示二: 除法模型 —— 输入 a 和 b, 让模型学出 a/b
# ===============================================================
# 数据: 所有"分子小于分母"的分数对 (j, i), 标签 = j/i
print('除法模型 =============================================')
xs = []
for i in range(2, 6):       # 分母 2..5
    for j in range(1, i):   # 分子 1..i-1
        xs.append([j, i])
ys = [x[0] / x[1] for x in xs]  # 正确答案: j/i
n = MLP(2, [10, 10, 10, 10, 1])  # 2 输入 -> 4 个隐藏层 -> 1 输出
print('xs:', xs)
print('ys:', ys)

h = 0.01                # 学习率
for k in range(20):
    # ① 前向
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
    # ② 清零梯度 ③ 反向 ④ 更新
    for param in n.parameters():
        param.grad = 0.0
    loss.backward()
    for param in n.parameters():
        param.data -= param.grad * h
    print(f'第 {k:2d} 轮 loss = {loss.data:.6f}')

# 继续训练, 直到 loss < 0.01
while True:
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
    if loss.data < 0.01:
        break
    for param in n.parameters():
        param.grad = 0.0
    loss.backward()
    for param in n.parameters():
        param.data -= param.grad * h
print(f'训练完成, loss = {loss.data:.6f} < 0.01')
print('ys   :', ys)
print('ypred:', [round(y.data, 4) for y in ypred])
print()

# 测试: 注意分母 6/8/9 在训练数据里从没出现过, 考的是模型的泛化能力
print('测试 (训练时没见过的数据):')
for a, b in [(2, 6), (3, 9), (2, 8)]:
    print(f'  n([{a},{b}]) = {n([a, b]).data:.4f}   (正确答案 {a}/{b} = {a/b:.4f})')
print('精确度一般, 但明显有向除法靠拢的样子 —— 模型学习成功')
