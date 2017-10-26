# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
''''
import pandas as pd
import numpy as np

data = np.arange(30).reshape(3,-1)
data = pd.DataFrame(data)

def f(x):
    return x*x

print data
print data.apply(lambda x:x*x,axis=1)
print data.apply(lambda x:x*x,axis=0)


print list.dtype
# 导入数据
path = 'datas/household_power_consumption_200.txt'  # 路径一定要看数据放的位置，写自己的对应存储路径即可
df = pd.read_csv(path, sep=';')
# help(pd.read_csv)
# 查看数据，只需要看数据的头部
names = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power',
         'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
         'Sub_metering_3']
t= df[names[0:2]]
print type(t)
'''

#获取数据源
path='datas/household_power_consumption_200.txt'
df = pd.read_csv(path,sep=';')
#重命名列
#print df.columns
names = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power',
       'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
       'Sub_metering_3']

#对数据进行处理
new_df = df.replace('?',np.nan)
datas = new_df.dropna(how='any')

#定义X和Y
X = datas[names[8:9]] #代表电热水器的电功率
Y = datas[names[5:6]] #代表电流
#X = datas[names[2:4]]
#Y = datas[names[5:6]]

print X.head(5)
print Y.head(5)
#对数据进行训练数据和测试数据的划分
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)


#对数据进标准化为了符合正态分布
#ss = StandardScaler()

#X_train = ss.fit_transform(X_train) # 对训练数据进行正态分布处理并且训练
#X_test = ss.transform(X_test) #由于之前已经对

#对数据进行训练
lr=LinearRegression()

lr.fit(X_train,Y_train)

#对数据进行预测
y_predict = lr.predict(X_test)
#对数据进行评估

#画图
t=np.arange(len(X_test))

plt.plot(t, Y_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc = 'lower left') #图例的位置
plt.title(u"电热水器功率和电流之间的关系", fontsize=20)
plt.grid(b=True)
plt.show()