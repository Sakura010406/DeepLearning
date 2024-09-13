"""
定义损失函数
"""


def squared_loss(y_true, y_pred):
    """均方损失"""
    return (y_true - y_pred.reshape(y_true.shape)) ** 2 / 2
