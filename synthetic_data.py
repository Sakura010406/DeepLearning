"""
   生成人造数据
"""
import torch


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    # z = torch.normal(0, 1, (num_examples, 1, 2))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    """-1表示行数由pytorch自行推断，列数则为1"""
    return X, y.reshape(-1, 1)

