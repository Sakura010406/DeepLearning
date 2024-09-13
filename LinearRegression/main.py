"""
   深度学习：线性回归实战
"""

# 内嵌画图
import matplotlib.pyplot as plt
import torch
import random
from synthetic_data import synthetic_data
# from data_iter import data_iter
from linreg import linreg
from squared_loss import squared_loss
from sgd import sgd
from d2l import torch as d2l
from torch.utils import data as data
import numpy as np
# nn是神经网络的缩写
from torch import nn


# a = torch.arrange(12, dtype=torch.float).reshape(3, 4)
# print(torch.__version__)
# print(a)
# print(a.size(), len(a))
"""生成数据集及其标签"""

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()

# 每一次读取10个feature
batchsize = 10

"""
# 手动实现线性回归模型
# 定义初始模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练过程
# lr = 0.03
# lr = 0.001
lr = 1
# 把整个数据扫三遍
num_epochs = 10
# 模型
net = linreg
# 损失函数
loss = squared_loss
# 优化函数
optalg = sgd

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # x和y的小批量损失
        # 因为l的形状是('batch_size', 1)，而不是一个标量.l中的所有元素被加到一起,并以此计算关于['w', 'b']的梯度
        # print(X, '\n', y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}, loss{float(train_l.mean()):f}')  # :.nf的作用是以标准格式输出一个浮点数,前可加数字,:f默认6位小数
        # 暂时只用一个
        # break
# 比较真实的差别
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')


"""


# 简洁实现线性回归模型
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    # *表示接受任意多个参数并将其放在一个元组中
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


data_iter = load_array((features, labels), batchsize)
print(next(iter(data_iter)))


net = nn.Sequential(nn.Linear(2, 1))
# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 定义损失函数
loss = nn.MSELoss()
# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()  # 更新模型参数
    los = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {los:.5f}')
