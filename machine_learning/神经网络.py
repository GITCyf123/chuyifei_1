# 导入所需库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 加载数据
iris = load_iris()
X = iris.data  # 特征数据，包括花萼长度、花萼宽度、花瓣长度、花瓣宽度
y = iris.target  # 标签数据，0、1、2分别代表三种鸢尾花

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,      # 测试集占30%
    random_state=42     # 随机种子，保证结果可重现
)

# 创建BP神经网络模型
model = MLPClassifier(
    hidden_layer_sizes=(10,),  # 隐藏层有10个神经元
    activation='tanh',         # 使用ReLU激活函数
    solver='adam',             # 使用adam优化器
    max_iter=300,              # 最大迭代次数
    random_state=42            # 随机种子
)

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型在测试集上的准确率为：{accuracy:.2f}")

# 可选：输出更多模型信息
print(f"\n模型详细信息：")
print(f"迭代次数：{model.n_iter_}")
print(f"最终损失值：{model.loss_:.4f}")
print(f"隐藏层数量：{model.n_layers_ - 2}")  # 减去输入层和输出层