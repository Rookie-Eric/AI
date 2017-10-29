# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

#解决画图产生的中文乱码问题
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#测试一下创建了X和Y的矩阵
X = np.random.random((3,3))
Y = np.arange(1,4)
print X,Y

def FeatureSelection(X,Y):
    # Build a classification task using 3 informative features
    '''
    X, Y = make_classification(n_samples=10,  #该函数负责创建一个自定义的矩阵（X->Y）的关系
                               n_features=10,
                               n_informative=3,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               random_state=0,
                               shuffle=False)'''
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=10,random_state=0) #创建一个额外树
    #forest = RandomForestClassifier (n_estimators = 10)  #创建一个随机树

    #计算X特征值对于Y的影响，并排序出来
    forest.fit(X, Y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    print indices
    # Print the feature ranking
    print(u"特征排名 :")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title(u"特征选择")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

FeatureSelection(X,Y)