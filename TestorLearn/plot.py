import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def mean_filter(data, size=5):#滑动滤波器（均值）
    conv = np.ones(size)
    result = np.convolve(data, conv, mode='valid')
    result = result / size
    return result


one = pd.read_csv('../data/dynamic/data/n_sample_0_65.txt', header=None, delim_whitespace=True)
print(one)
flex1 = one.iloc[:,7]
flex2 = one.iloc[:,8]
flex3 = one.iloc[:,9]

# accx = one.iloc[:,5]
# accy = one.iloc[:,6]
# accz = one.iloc[:,7]
# gyrx = one.iloc[:,8]
# gyry = one.iloc[:,9]
# gyrz = one.iloc[:,10]

# flex1_pro = mean_filter(flex1)
# flex2_pro = mean_filter(flex2)
# flex3_pro = mean_filter(flex3)
# flex4_pro = mean_filter(flex4)
# flex5_pro = mean_filter(flex5)

# accx_pro = mean_filter(accx)
# accy_pro = mean_filter(accy)
# accz_pro = mean_filter(accz)

# print(flex1)

plt.figure(dpi=100)
plt.plot(flex1,label='flex1')
plt.plot(flex2,label='flex2')
plt.plot(flex3,label='flex3')

plt.title('原始数据')
# plt.plot(accx,label='accx')
# plt.plot(accy,label='accy')
# plt.plot(accz,label='accz')
# plt.plot(gyrx,label='gyrx')
# plt.plot(gyry,label='gyry')
# plt.plot(gyrz,label='gyrz')
# plt.legend()
# plt.show()


# a = pd.read_csv('../data/public/a_sample_0_0.txt',delim_whitespace=True,header=None)
# b = pd.read_csv('../data/public/a_sample_0_1.txt',delim_whitespace=True,header=None)
# print(a)
# onet = a.iloc[:,13]
# onett = b.iloc[:,13]
# plt.figure(dpi=100)
# plt.plot(onet)
# plt.plot(onett)
two = pd.read_csv('../data/dynamic_unify/n_sample_0_65.csv')
plt.figure(dpi=100)
flex1 = two.iloc[:,7]
flex2 = two.iloc[:,8]
flex3 = two.iloc[:,9]
plt.plot(flex1,label='flex1')
plt.plot(flex2,label='flex2')
plt.plot(flex3,label='flex3')
# plt.plot(flex4_pro,label='flex4')
# plt.plot(flex5_pro,label='flex5')
# plt.plot(accx_pro,label='accx')
# plt.plot(accy_pro,label='accy')
# plt.plot(accz_pro,label='accz')
# plt.legend()
plt.title('重采样数据')
plt.show()