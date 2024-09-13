"""
定义softmax函数
"""
import torch

def softmax(X):
    """写函数"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition  # 广播机制
