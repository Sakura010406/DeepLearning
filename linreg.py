"""
定义线性回归模型
"""
import torch


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b
