import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
import os


# 扩展到45
def up_sample(data):
    length = len(data)
    x = np.arange(0, length)

    # 采用一维插值进行上采样
    fc = interp1d(x, data, kind='cubic', axis=0)
    # fc2 = interp1d(x, data, kind='nearest')
    # fc3 = interp1d(x, data, kind='linear')

    xint = np.linspace(x.min(), x.max(), 45)
    yint = fc(xint)
    # yint2 = fc2(xint)
    # yint3 = fc3(xint)

    # plt.figure(dpi=100)
    # plt.plot(yint, color='red', label='cubic')
    # plt.legend()
    #
    # plt.figure(dpi=100)
    # plt.plot(yint2, color='green', label='nearest')
    # plt.legend()
    #
    # plt.figure(dpi=100)
    # plt.plot(yint3, color='blue', label='linear')  # 最牛逼的
    # plt.legend()
    #
    # plt.figure(dpi=100)
    # plt.plot(data, color='red', label='raw')
    #
    # plt.legend()
    # plt.show()


    return yint


# 下采样到45
def down_sample(data):
    length = len(data)
    x = np.arange(0, length)
    # print(x)

    # 利用插值法下采样
    fc = interp1d(x, data, kind='cubic', axis=0)
    # fc2 = interp1d(x, data, kind='nearest')
    # fc3 = interp1d(x, data, kind='linear')

    # 控制为45个采样点
    xint = np.linspace(x.min(), x.max(), 45)
    # print(xint)

    # 进行插值downsample
    yint = fc(xint)
    # yint2 = fc2(xint)
    # yint3 = fc3(xint)
    # print(yint)

    # plt.figure(dpi=100)
    # plt.plot(yint, color='red', label='one')
    #
    # plt.figure(dpi=100)
    # plt.plot(yint2, color='green', label='nearest')
    #
    # plt.figure(dpi=100)
    # plt.plot(yint3, color='blue', label='linear')
    #
    # plt.figure(dpi=100)
    # plt.plot(data, color='red', label='two')
    #
    # plt.legend()
    # plt.show()

    return yint



if __name__ == '__main__':
    # one = pd.read_csv('../data/dynamic/data/a_sample_0_0.txt', header=None, delim_whitespace=True)
    # kkk = pd.read_csv('../data/dynamic/data/k_sample_0_46.txt', header=None, delim_whitespace=True)
    # print(len(one), len(kkk))
    # flex1_one = one.iloc[:, 0]
    # flex1_k = kkk.iloc[:, 0]
    # flex2 = one.iloc[:, 1]
    # flex3 = one.iloc[:, 2]
    # flex4 = one.iloc[:, 3]
    # flex5 = one.iloc[:, 4]

    # print(flex1)
    # up_sample(flex1_one)
    # down_sample(flex1_k)


    # 批量进行数据长度固定
    dir_path = '../data/dynamic/data/'
    target_path = '../data/dynamic_unify/'
    files = os.listdir(dir_path)
    # print(files)
    for i in files:
        print(i)
        data = pd.read_csv(dir_path + i, header=None, delim_whitespace=True)
        print(data)
        sam_len = len(data)
        if sam_len >= 45:
            data_unify = pd.DataFrame(down_sample(data))
        elif sam_len < 45:
            data_unify = pd.DataFrame(up_sample(data))
        else:
            data_unify = data

        if not os.path.exists(target_path):
            os.mkdir(target_path)
        else:
            print('文件夹已经存在')
        data_unify.to_csv(target_path + i, header=None, index_label=None, index=None)

