"""
update函数
"""

from d2l import torch as d2l


def updater(batch_size, W, b, lr):
    """更新参数"""
    return d2l.sgd([W, b], lr, batch_size)
