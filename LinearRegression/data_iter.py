"""
   接收批量大小、矩阵特征和标签向量作为输入
"""
import torch
import random


def data_iter(batch_size, features, labels):
    """ 接收数据 """
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
