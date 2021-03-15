import numpy as np
import matplotlib.pyplot as plt


epochs = 100  # 循环次数
lr = 1e-1  # 学习率
num_observations = 500  # 生成样本个数


def random_dots(num):
    # 多元正态分布生成样本
    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num)

    # 一半为正样本，赋值1  一半为负样本，赋值-1
    x = np.vstack((x1, x2)).astype(np.float32)
    y = np.hstack((np.ones(num), -np.ones(num)))

    return x, y


def show_clf(x, y, b, w):
    plt.figure()

    # 散点图，画出正负样本
    plt.scatter(x[:, 0], x[:, 1],
                c=["b" if i > 0 else "k" for i in y],
                alpha=0.9, s=10,
                marker="o")

    # 绘制等高线
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 9, 500))
    xy = np.c_[xx.ravel(), yy.ravel()]
    z = np.dot(xy, w) + b
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, 15, alpha=0.5, cmap=plt.cm.bone)
    plt.contour(xx, yy, z, levels=[0], colors='r', linewidths=3)

    plt.show()


def main():

    # 随机初始化参数
    x, y = random_dots(num_observations)
    w = np.random.randn(2, 1)
    b = np.random.randn(1, 1)

    for i in range(epochs):

        outputs = np.dot(x, w) + b   # y=w*x+b
        loss = y - outputs.squeeze(-1)
        w[0] += np.mean(lr * loss * x[:, 0])
        w[1] += np.mean(lr * loss * x[:, 1])
        b += np.mean(lr * loss * 1)

        acc = np.sum(((outputs > 0) * 2 - 1).squeeze(-1) == y) / len(y) * 100
        print("Epochs %3d, loss: %.4f, acc: %.2f%%" %
              (i + 1, np.mean(loss), acc))

    show_clf(x, y, b, w)


if __name__ == '__main__':
    main()
