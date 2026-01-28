import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import pandas as pd

iris = sklearn.datasets.load_iris()
# 我们用sklaern中的target和feature_names两个key
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['x', 'y', 'column3', 'column4', 'label']

data = np.array(df.iloc[:100, [0, 1, 4]])  # 数据提取前100行，第1，2，5列
x, y = data[:, :-1], data[:, -1]  # x是前两列，y是最后一列
y = np.array([1 if i == 1 else -1 for i in y])


class Precept:
    # 初始化
    def __init__(self):
        self.w = np.zeros(len(data[0]) - 1)
        self.b = 0
        self.lr = 0.1  # 公式中的lambda

    def sign(self, x, w, b):
        y = np.dot(x, w) + b  # y = wx+b
        return y

    def fit(self, x_train, y_train):
        is_wrong = 0
        while is_wrong == 0:
            wrong_count = 0
            for d in range(len(x_train)):
                x = x_train[d]
                y = y_train[d]
                if y * self.sign(x, self.w, self.b) <= 0:  # yi * w * xi<=0 --> 分类错误
                    self.w = self.w + self.lr * np.dot(x, y)  # 更新w
                    self.b = self.b + self.lr * y  # 更新b
                    wrong_count += 1
            if wrong_count == 0:  # 若无错误分类则结束
                is_wrong = 1


Preceptron = Precept()
Preceptron.fit(x, y)

# 绘图
# 生成直线上的点
line_x = np.linspace(4, 7, 10)
line_y = (-Preceptron.w[0] * line_x - Preceptron.b) / Preceptron.w[1]

plt.plot(line_x, line_y)
plt.scatter(df[:50]['x'], df[:50]['y'], c='b',marker='o', label='negative')  # 选取数据前50个点做负例
plt.scatter(df[50:100]['x'], df[50:100]['y'], c='g',marker='v', label='positive')  # 再选取50个点做正例
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

