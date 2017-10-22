# coding: utf-8
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

df.head()

df.columns

names=['Date', 'Time', 'Global_active_power', 'Global_reactive_power',
       'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
       'Sub_metering_3']

#查看数据的所有分布
# for i in df.columns:
#     print df[i].value_counts()


##异常数据的处理
new_df = df.replace('?',np.nan)
datas = new_df.dropna(how='any')

#创建一个时间字符串格式化字符串
# 时间字符串换成时间元祖
def date_format(dt):
    import time
    t = time.strptime(' '.join(dt),'%d/%m/%Y %H:%M:%S')
    return(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)

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

lr = LinearRegression()
lr.fit(X_train,Y_train)

print "训练数据的拟合程度R方：",lr.score(X_train,Y_train) #R方越趋近于1越好
#RMSE

y_predict = lr.predict(X_test)
print "测试数据的拟合程度R方：",lr.score(X_test,Y_test) #R方越趋近于1越好
# #模型保存/持久化
# from sklearn.externals import joblib
# #保存模型
# joblib.dump(ss,'data_ss.model')
# joblib.dump(lr,'data_lr.model')
# #加载
# joblib.load('data_ss.model')
# joblib.load('data_lr.model')

