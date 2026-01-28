import pandas as pd
import numpy as np
import json


class NaiveBayes:
    def __init__(self):
        self.model = {}  # 存储训练好的朴素贝叶斯模型，包括每个类别的先验概率和每个特征的条件概率

    def calEntropy(self, y):  # 计算熵
        valRate = y.value_counts().apply(lambda x: x / y.size)  # 计算每个取值的概率
        valEntropy = np.inner(valRate, np.log2(valRate)) * -1  # 计算熵
        return valEntropy

    def fit(self, xTrain, yTrain=pd.Series()):
        """
        训练朴素贝叶斯模型

        参数:
            xTrain: 特征数据
            yTrain: 标签数据（如果为空，则使用xTrain的最后一列作为标签）
        """
        if not yTrain.empty:  # 如果没有传入分类标签，则默认将最后一列作为分类标签
            xTrain = pd.concat([xTrain, yTrain], axis=1)
        self.model = self.buildNaiveBayes(xTrain)  # 训练朴素贝叶斯模型
        return self.model

    def buildNaiveBayes(self, xTrain):
        """
        构建朴素贝叶斯模型

        返回:
            retModel: 训练好的模型，包含先验概率和条件概率
        """
        yTrain = xTrain.iloc[:, -1]  # 获取分类标签（最后一列）
        yTrainCounts = yTrain.value_counts()  # 统计每个类别的样本数量

        # 使用拉普拉斯平滑计算每个类别的先验概率
        yTrainCounts = yTrainCounts.apply(lambda x: (x + 1) / (yTrain.size + yTrainCounts.size))

        retModel = {}
        for nameClass, val in yTrainCounts.items():
            retModel[nameClass] = {'PClass': val, 'PFeature': {}}  # 存储每个类别的先验概率和条件概率

        propNamesAll = xTrain.columns[:-1]  # 获取所有特征的名称（排除最后一列标签）
        allPropByFeature = {}

        # 获取每个特征的所有可能取值
        for nameFeature in propNamesAll:
            allPropByFeature[nameFeature] = list(xTrain[nameFeature].value_counts().index)

        # 按类别分组计算条件概率
        for nameClass, group in xTrain.groupby(xTrain.columns[-1]):
            for nameFeature in propNamesAll:
                eachClassPFeature = {}
                propDatas = group[nameFeature]
                propClassSummary = propDatas.value_counts()  # 统计每个特征取值的数量

                # 处理未出现的特征值（拉普拉斯平滑）
                for propName in allPropByFeature[nameFeature]:
                    if propName not in propClassSummary:
                        propClassSummary[propName] = 0

                Ni = len(allPropByFeature[nameFeature])  # 该特征的可能取值数量
                # 使用拉普拉斯平滑计算条件概率
                propClassSummary = propClassSummary.apply(lambda x: (x + 1) / (propDatas.size + Ni))

                for nameFeatureProp, valP in propClassSummary.items():
                    eachClassPFeature[nameFeatureProp] = valP

                retModel[nameClass]['PFeature'][nameFeature] = eachClassPFeature

        return retModel

    def predictBySeries(self, data):
        """
        预测单个样本

        参数:
            data: 单个样本数据（pd.Series）

        返回:
            curClassSelect: 预测的类别
        """
        curMaxRate = None
        curClassSelect = None

        for nameClass, infoModel in self.model.items():
            rate = 0
            rate += np.log(infoModel['PClass'])  # 先验概率的对数

            PFeature = infoModel['PFeature']

            # 计算所有特征的条件概率乘积（使用对数相加）
            for nameFeature, val in data.items():
                if nameFeature in PFeature:
                    feature_probs = PFeature[nameFeature]
                    prob = feature_probs.get(val, 1e-6)  # 如果特征值未出现，使用很小的概率值
                    rate += np.log(prob)

            # 选择概率最大的类别
            if curMaxRate is None or rate > curMaxRate:
                curMaxRate = rate
                curClassSelect = nameClass

        return curClassSelect

    def predict(self, data):
        """
        预测多个样本

        参数:
            data: 样本数据（DataFrame或Series）

        返回:
            预测结果
        """
        if isinstance(data, pd.Series):
            return self.predictBySeries(data)
        return data.apply(lambda d: self.predictBySeries(d), axis=1)

    def display_model(self):
        """以易读格式显示模型"""
        print("\n朴素贝叶斯模型详情:")
        print("=" * 50)
        for class_name, class_info in self.model.items():
            print(f"\n类别: {class_name}")
            print(f"  先验概率: {class_info['PClass']:.4f}")
            print("  条件概率:")
            for feature_name, feature_probs in class_info['PFeature'].items():
                print(f"    {feature_name}:")
                for value, prob in feature_probs.items():
                    print(f"      {value}: {prob:.4f}")


# 主程序
if __name__ == '__main__':
    print('开始加载数据...')

    try:
        # 加载数据（请确保文件路径正确）
        dataTrain = pd.read_csv("D:/csv/watermelon_reference.txt", encoding='utf-8', sep=' ')
        print(f'数据加载成功，形状: {dataTrain.shape}')
        print(f'数据列名: {list(dataTrain.columns)}')
        print('\n前5行数据:')
        print(dataTrain.head())

    except FileNotFoundError:
        print("文件未找到，创建示例数据...")
        # 创建示例数据（如果文件不存在）
        dataTrain = pd.DataFrame({
            '色泽': ['青绿', '乌黑', '乌黑', '青绿', '浅白', '青绿', '乌黑', '乌黑', '乌黑', '青绿'],
            '根蒂': ['蜷缩', '蜷缩', '蜷缩', '蜷缩', '蜷缩', '稍蜷', '稍蜷', '稍蜷', '蜷缩', '稍蜷'],
            '敲声': ['浊响', '沉闷', '浊响', '沉闷', '浊响', '浊响', '浊响', '浊响', '沉闷', '沉闷'],
            '纹理': ['清晰', '清晰', '清晰', '清晰', '清晰', '清晰', '稍糊', '清晰', '稍糊', '稍糊'],
            '脐部': ['凹陷', '凹陷', '凹陷', '凹陷', '凹陷', '稍凹', '稍凹', '稍凹', '凹陷', '稍凹'],
            '触感': ['硬滑', '硬滑', '硬滑', '硬滑', '硬滑', '软粘', '软粘', '硬滑', '硬滑', '软粘'],
            '好瓜': ['是', '是', '是', '是', '是', '否', '否', '否', '否', '否']
        })
        print('使用示例数据')

    print('\n开始训练模型...')
    naiveBayes = NaiveBayes()
    treeData = naiveBayes.fit(dataTrain)

    # 显示模型详情
    naiveBayes.display_model()

    print('\n模型JSON格式:')
    print(json.dumps(treeData, ensure_ascii=False, indent=2))

    # 在训练集上进行预测
    predictions = naiveBayes.predict(dataTrain.iloc[:, :-1])
    results_df = pd.DataFrame({
        '预测值': predictions,
        '真实值': dataTrain.iloc[:, -1]
    })

    print('\n预测结果:')
    print(results_df)

    # 计算准确率
    correct_predictions = (results_df['预测值'] == results_df['真实值']).sum()
    accuracy = correct_predictions * 100.0 / len(results_df)
    print(f'\n正确率: {accuracy:.2f}%')

    print('\n训练完成!')




# import pandas as pd
# from sklearn import metrics
# # 加载莺尾花数据集
# from sklearn import datasets
# # 导入高斯朴素贝叶斯分类器
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
#
# # 导入数据集
# data = datasets.load_iris()  # 导入iris数据集
# iris_target = data.target  # 得到数据对应的标签
# iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)  # 利用Pandas转化为DataFrame格式
#
# # 查看数据集基本信息
# print("数据集特征形状:", iris_features.shape)
# print("数据集标签形状:", iris_target.shape)
# print("\n特征名称:", data.feature_names)
# print("目标类别:", data.target_names)
# print("\n前5行数据:")
# print(iris_features.head())
# print("\n对应的标签:")
# print(iris_target[:5])
#
# # 将数据集分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(
#     iris_features,
#     iris_target,
#     test_size=0.2,      # 20%作为测试集
#     random_state=0      # 随机种子，保证结果可重现
# )
#
# print(f"\n训练集大小: {X_train.shape}")
# print(f"测试集大小: {X_test.shape}")
#
# # 使用高斯朴素贝叶斯进行训练
# clf = GaussianNB()
# clf.fit(X_train, y_train)
#
# # 评估模型
# test_predict = clf.predict(X_test)
# accuracy = metrics.accuracy_score(y_test, test_predict)
#
# print('\n' + '='*50)
# print('模型评估结果:')
# print('='*50)
# print('The accuracy of the NB for Test Set is: %d%%' % (accuracy * 100))
#
# print('\n预测结果 vs 实际结果:')
# print('预测结果:', test_predict)
# print('实际结果:', y_test)
#
# # 详细分类报告
# print('\n详细分类报告:')
# print(metrics.classification_report(y_test, test_predict, target_names=data.target_names))
#
# # 预测单个样本
# print('\n' + '='*50)
# print('单样本预测演示:')
# print('='*50)
# print('待预测的样本:')
# print(X_test[:1])
#
# y_proba = clf.predict_proba(X_test[:1])  # 预测样本的概率
# y_pred = clf.predict(X_test[:1])  # 对这个样本进行预测
#
# print('预测结果:', y_pred)
# print('预测类别名称:', data.target_names[y_pred][0])
# print("预测的概率值:")
# for i, prob in enumerate(y_proba[0]):
#     print(f"  属于 {data.target_names[i]} 的概率: {prob:.4f} ({prob*100:.2f}%)")