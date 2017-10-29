# coding=utf-8
'''
下面的代码是用一个小例子解决线性回归过拟合问题(弹性网络回归：结合了Rideg回归和lasso回归的优点)
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model.coordinate_descent import ConvergenceWarning

#解决画图产生的中文乱码问题
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#生成一段数据来测试一下拟合问题
np.random.seed(100)
np.set_printoptions(linewidth=1000, suppress=True)
N = 10
x = np.linspace(0, 6, N) + np.random.randn(N)
y = 1.8*x**3 + x**2 - 14*x - 7 + np.random.randn(N)
x.shape = -1, 1
y.shape = -1, 1

#模型
models = [
    Pipeline([
            ('Poly', PolynomialFeatures()),
            ('Linear', LinearRegression(fit_intercept=False))
        ]),
    Pipeline([
            ('Poly', PolynomialFeatures()),
            ('Linear', RidgeCV(alphas=np.logspace(-3,2,50), fit_intercept=False))
        ]),
    Pipeline([
            ('Poly', PolynomialFeatures()),
            ('Linear', LassoCV(alphas=np.logspace(-3,2,50), fit_intercept=False))
        ]),
    Pipeline([
            ('Poly', PolynomialFeatures()),
            ('Linear', ElasticNetCV(alphas=np.logspace(-3,2,50), l1_ratio=[.1, .5, .7, .9, .95, 1], fit_intercept=False))
        ])
]

plt.figure(facecolor='W')
#degree = np.arange (1, N, 2)  # 定义函数的阶数  X^2代表2阶函数 N表示N阶
#dm = degree.size
#print dm
colors = []  # 颜色
for c in np.linspace (16711680, 255, 5):
    colors.append ('#%06x' % c)
titles = [u'线性回归', u'Ridge回归', u'Lasso回归', u'ElasticNet']


model = models[3]
#plt.subplot(2, 2, 3 + 1)
plt.plot(x, y, 'ro', ms=5, zorder=N)


model.set_params(Poly__degree=5)

model.fit(x, y.ravel())

lin = model.get_params('Linear')['Linear']

output = u'%s:%d阶，系数为：' % (titles[3], 3)
print output, lin.coef_.ravel()

x_hat = np.linspace(x.min(), x.max(), num=100)
x_hat.shape = -1, 1

y_hat = model.predict(x_hat)

s = model.score(x, y)
z = N - 1
#z = N - 1 if (d == 2) else 0
label = u'%d阶, 正确率=%.3f' % (5, s)
plt.plot(x_hat, y_hat, color=colors[3], lw=2, alpha=0.75, label=label, zorder=z)

plt.legend(loc='upper left')
plt.grid(True)
plt.title(titles[3])
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)

plt.tight_layout(1, rect=(0, 0, 1, 0.95))
plt.suptitle(u'弹性网络下回归过拟合显示', fontsize=22)
plt.show()