#coding:utf-8

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.scimath import logn
from math import e
import matplotlib as mpl
#防止中文乱码问题
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False

x=np.linspace(0,6,120)

plt.plot(x,np.log(x)/np.log(0.5),'y-', linewidth=2, label=u'log0.5(x)')
plt.plot(x,logn(e,x),'b-',linewidth=2, label=u'loge(x)')
#plt.plot(x,np.log(x)/np.log(5))
plt.plot(x,logn(5,x),'g-',linewidth=2, label=u'loge(x)')
plt.plot(x,np.log10(x),'r-',linewidth=2, label=u'loge(x)')
plt.plot([1,1,1,1],[-3,0,1,5],'--',color='darkgray')

plt.axis([0, 2.5, -3.5, 5.5])
plt.legend(loc='lower right') #图例的位置
plt.grid(True)
plt.show()

x=[0.5,1.0,1.5,2.0,2.5,3.0];
y=[1.75,2.45,3.81,4.80,7.00,8.60];
p=polyfit(x,y,2)
x1=0.5:0.5:3.0;
y1=polyval(p,x1);
plot(x,y,'*r',x1,y1,'-b')