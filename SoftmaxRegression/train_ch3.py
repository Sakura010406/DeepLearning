"""多
个周期
"""
from Animator import Animator
from train_epoch_ch3 import train_epoch_ch3
from evaluate_accuracy import evaluate_accuracy


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, W, b, lr):  # @save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater, W, b, lr)
        test_acc = evaluate_accuracy(net, test_iter, W, b)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
