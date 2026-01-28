import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class perceptron:

    def __init__(self, x, y, alpha=0.001, circle=500, batchlength=20):
        self.x = x            # 训练样本
        self.y = y            # 训练样本中各组数据对应的类别
        self.alpha = alpha    # 学习率
        self.circle = circle  # 学习次数
        self.n = x.shape[0]   # 样本个数
        self.p = x.shape[1]   # 样本指标个数
        self.w = np.random.normal(size=(self.p, 1))
        self.b = np.random.normal(size=1)
        self.batchlength = batchlength   # 每次训练样本中使用的数据个数

    def batches(self):
        data = list(zip(self.x, self.y))
        np.random.shuffle(data)
        batches = [data[i:i+self.batchlength] for i in range(0, self.n, self.batchlength)]
        return batches

    def sign(self, x):
        '''sign激活函数'''
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def train(self):
        for i in range(self.circle):
            print('the {} circle'.format(i))
            for batch in self.batches():
                dw = db = 0
                num = 1
                for x, y in batch:
                    if y * (np.dot(self.w.T, x.T)+self.b) >= 0:
                        continue
                    else:
                        dw += -y * x.T
                        db += -y
                        num += 1
                if num != 0:
                    self.w -= self.alpha * dw / num
                    self.b -= self.alpha * db / num
                else:
                    continue
            color = []
            for c in self.y:
                if c == 1:
                    color.append('green')
                else:
                    color.append('red')
            x = np.arange(180, 470, 1)
            y = -self.w[0] * x/self.w[1]-self.b/self.w[1]  #分割线
            plt.plot(x, y)
            plt.scatter(np.array(self.x[:, 0]), np.array(self.x[:, 1]), color=color)
            plt.xlim([180,470])
            plt.ylim([160,320])
            plt.pause(0.1)
            plt.clf()

    def prediction(self, x):
        '''根据数据x判断该数据属于哪一类'''
        s = np.dot(self.w.T, x)+self.b
        output = self.sign(s)
        return output

if __name__ == '__main__':
    df = pd.read_excel('Dry_Bean_Dataset.xlsx')
    x = df.loc[0:3320, 'MajorAxisLength':'MinorAxisLength']
    y = df.loc[0:3320, 'Class']
    X = np.mat(x)
    Y = []
    for c in y:
        if c =='SEKER':
            Y.append(1)
        else:
            Y.append(-1)
    p = perceptron(x=X, y=Y)
    p.train()