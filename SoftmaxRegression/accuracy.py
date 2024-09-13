"""
计算准确率
"""
import numpy as np
import torch
"""
type返回数据结构类型（list、dict、numpy.ndarray 等）
dtype 返回数据元素的数据类型（int、float等）
"""

def accuracy(y_hat, y):
    """准确率"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        # y_hat.type(y.type)
    # print(cmp.dtype, cmp, y.dtype, y.type)
    return float(cmp.type(y.dtype).sum())
