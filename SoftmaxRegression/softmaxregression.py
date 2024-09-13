"""
定义回归模型
"""
import torch
from softmax import softmax as softmax_module


def softreg(X, W, b):
    """回归模型"""
    return softmax_module(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
