"""
评估准确率
"""
import torch
import accuracy
from Accumulator import Accumulator


def evaluate_accuracy(net, data_iter, W, b):  # @save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy.accuracy(net(X, W, b), y), y.numel())
    return metric[0] / metric[1]
