#coding:utf-8

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.scimath import logn
from math import e
import matplotlib as mpl
#防止中文乱码问题
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False

x=np.linspace(0,6,100)

plt.plot(x,np.log(x)/np.log(0.5),'y-', linewidth=2, label=u'log0.5(x)')
plt.plot(x,logn(e,x),'b-',linewidth=2, label=u'loge(x)')
#plt.plot(x,np.log(x)/np.log(5))
plt.plot(x,logn(5,x),'g-',linewidth=2, label=u'loge(x)')
plt.plot(x,np.log10(x),'r-',linewidth=2, label=u'loge(x)')

plt.axis([0, 2.5, -3., 5.])
plt.legend(loc = 'lower right') #图例的位置
plt.grid(True)
plt.show()