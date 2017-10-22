# coding: utf-8

'''
这是第一个线性回归的简单例子，基于单项式的线性方程，类似于求y = ax + b;   b是偏差值，根据方程得出 ,b = y - ax ,y是真实结果
用来求时间与功率的关系，把时间看做X,把功率比作Y;
根据画图的实验结果得出时间与功率的拟合程度很低说明时间和功率没有关系
'''

#导入需要的模块、包
import sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#导入数据
path='datas/household_power_consumption_200.txt' #路径一定要看数据放的位置，写自己的对应存储路径即可
df = pd.read_csv(path,sep=';')
# help(pd.read_csv)
#查看数据，只需要看数据的头部
df.head()
#获取列名称
df.columns
names=['Date', 'Time', 'Global_active_power', 'Global_reactive_power',
       'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
       'Sub_metering_3']

#查看数据的所有分布
#for i in df.columns:
#    print df[i].value_counts()

##异常数据的处理
new_df = df.replace('?',np.nan) #？替换成np.nan
#对于空值，看数据来源，一般是从数据库取数据，你取的时候一定知道空值是什么，别人给你的，你去问一下
datas = new_df.dropna(how='any')#删除空值，any是只要有空值我就删，all是这行都是空值我就删除

#创建一个时间字符串格式化字符串
# 时间字符串换成时间元祖
def date_format(dt):
    import time
    t = time.strptime(' '.join(dt),'%d/%m/%Y %H:%M:%S')
    return(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)#检查细致，看定义的函数跟我是否一样

#获取x和y变量，并将时间转换成数值型的连续变量
X = datas[names[0:2]]
X = X.apply(lambda x:pd.Series(date_format(x)),axis=1) #axis=1是列，按行取列的数据
Y = datas[names[2]]

print X.head(3)
print Y.head(3)

#对数据集进行测试数据集和训练数据集划分
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

#数据标准化：3种 
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#模型的构建与训练过程
from sklearn.linear_model import LinearRegression
lr = LinearRegression()#用的具体的模型
lr.fit(X_train,Y_train) #用训练数据来训练模型

#模型的预测，对测试数据用构建的lr模型来预测
y_predict = lr.predict(X_test) #Y_test

#模型效果的评估
print "训练数据的拟合程度R方：",lr.score(X_train,Y_train) #R方越趋近于1越好
#RMSE  

print "测试数据的拟合程度R方：",lr.score(X_test,Y_test)

mse = np.average((y_predict-np.array(Y_test))**2)
rmse = np.sqrt(mse)
print "MSE:" ,mse
print "RMSE:",rmse

#返回的模型的具体参数，就是我们讲的theta
lr.coef_

# #模型保存/持久化
# from sklearn.externals import joblib
# #保存模型
# joblib.dump(ss,'data_ss.model')
# joblib.dump(lr,'data_lr.model')
# #加载
# joblib.load('data_ss.model')
# joblib.load('data_lr.model')

#防止中文乱码问题
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#画图
t=np.arange(len(X_test))
# plt.figure(facecolor='w')#可以不设置
plt.plot(t, Y_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc = 'lower left') #图例的位置
plt.title(u"线性回归预测时间和功率之间的关系", fontsize=20)
plt.grid(b=True)
plt.show()

# 数据：日期、时间、有功功率、无功功率、电压、电流、厨房电功率、洗衣机电功率、热水器的电功率


