import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#包括绘制原始数据图表，以及滑动窗口滤波器



def mean_filter(data, size=5):#滑动滤波器（均值）
    conv = np.ones(size)
    result = np.convolve(data, conv, mode='same')
    result = result / size
    return result


one = pd.read_csv('../data/private/dongzuo4.csv')
flex1 = one.iloc[:,0]
flex2 = one.iloc[:,1]
flex3 = one.iloc[:,2]
flex4 = one.iloc[:,3]
flex5 = one.iloc[:,4]
accx = one.iloc[:,5]
accy = one.iloc[:,6]
accz = one.iloc[:,7]
gyrx = one.iloc[:,8]
gyry = one.iloc[:,9]
gyrz = one.iloc[:,10]
# print(flex1)

plt.figure(dpi=100)
plt.plot(flex1,label='flex1')
plt.plot(flex2,label='flex2')
plt.plot(flex3,label='flex3')
plt.plot(flex4,label='flex4')
plt.plot(flex5,label='flex5')
plt.legend()
plt.savefig('../data/private/elas2d.jpg',dpi=150)

plt.figure(dpi=100)
plt.plot(accx, label='accx')
plt.plot(accy, label='accy')
plt.plot(accz, label='accz')
plt.legend()
plt.savefig('../data/private/acc2d.jpg',dpi=150)

plt.figure(dpi=100)
plt.plot(gyrx,label='gyrx')
plt.plot(gyry,label='gyry')
plt.plot(gyrz,label='gyrz')
plt.legend()
plt.savefig('../data/private/gyro2d.jpg',dpi=150)



plt.show()