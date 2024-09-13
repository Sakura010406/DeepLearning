"""
softmax回归
"""
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from load_data import load_data_fashion_mnist as load_data_fashion_mnist
from softmaxregression import softreg as softreg
from cross_entropy import cross_entropy as cross_entropy
from evaluate_accuracy import evaluate_accuracy as evaluate_accuracy
from updater import updater as updater
from train_ch3 import train_ch3 as train_ch3

d2l.use_svg_display()  # 用svg显示图片清晰度更高

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
print(len(mnist_train), len(mnist_test))
# 第零个example(样本)下第一张图片
print(mnist_train[0][0].shape)


def get_fashion_mnist_labels(labels):  # @save
    # 返回Fashion - MNIST数据集的文本标签
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


"""
可视化数据集函数
# Fashion-MNIST中包含的10个类别，分别为t-shirt（T恤）、trouser（裤子）
# 、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）
# 、bag（包）和ankle boot（短靴）。 以下函数用于在数字标签索引及其文本名称之间进行转换。

# 可视化样本
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    绘制图像列表
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# 训练数据集中前[几个样本的图像及其相应的标签]
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
d2l.plt.show()


batch_size = 256


def get_dataloader_workers():  # @save
    使用4个进程来读取数据
    return 4


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

"""

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# for X, y in train_iter:
#     print(cross_entropy(softreg(X, W, b), y))
#     print(evaluate_accuracy(softreg, test_iter, W, b))
#     break


lr = 0.1
num_epochs = 5

train_ch3(softreg, train_iter, test_iter, cross_entropy, num_epochs, updater, W, b, lr)


def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X, W, b).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


predict_ch3(softreg, test_iter)
d2l.plt.show()
