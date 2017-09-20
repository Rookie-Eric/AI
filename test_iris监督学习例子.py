#!/usr/bin/python3
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# 对三种用户分类

# 导入数据部分
iris = load_iris()  # 加载数据
# print('类型：', type(iris))
print('打印iris的特征数据:', iris.feature_names)  # 打印iris的特征数据
print('打印iris的标签名称:', iris.target_names)  # 打印iris的标签名称
print('打印iris第一行的样本数据:', iris.data[0])  # 打印iris的样本数据
# print('类型：', type(iris.data[0]))
print('打印iris所属的标签用符号0,1,2分别表示垃圾用户,普通用户和大R用户:', iris.target[0])  # 打印iris所属的标签用符号0,1,2分别表示垃圾用户,普通用户和大R用户

for i in range(len(iris.target)):
    print("样本 %d: 标签 %s, 特征 %s" % (i, iris.target[i], iris.data[i]))

# 把数据分为训练数据和测试数据
test_idx = [148, 50, 100, 1]
print("测试下第%d %d %d %d行样本的数据" % (test_idx[0], test_idx[1], test_idx[2], test_idx[3]))

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)
# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
# 创建一个分类器
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print('预测结果:', test_target)
print('测试结果:', clf.predict(test_data))  # 把第[148, 50, 100, 1]行测试样本的特征值给分类器让它预测属于哪个标签