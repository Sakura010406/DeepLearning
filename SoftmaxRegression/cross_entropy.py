"""
计算交叉熵
"""
import torch
import accuracy


def cross_entropy(y_hat, y):
    """写函数"""
    # 准确率
    # print(f'accuracy rate:{float(accuracy.accuracy(y_hat, y) / len(y)):.4f}')
    return -torch.log(y_hat[range(len(y)), y])
