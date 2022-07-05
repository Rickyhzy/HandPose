import h5py
from scipy.io import loadmat
import numpy as np
import pandas as pd

# np.set_printoptions(threshold=np.inf)

data_dir = '../data/public/datasetStaticGestures.mat'
data_static = loadmat(data_dir)
# print(data_static)
data = data_static['DataTrainSG']
print(data_static['DataTrainSG'].shape, type(data_static['DataTrainSG']))


# 获取key
key_name1 = list(data_static.keys())[-1]
key_name2 = list(data_static.keys())[-2]
# 根据key获取数据
data1 = data_static[key_name1]
data2 = data_static[key_name2]
# 获取数据的shape,这里数据的shape是(1,7)
# 后面表示的是图片的数量
print(data1.shape)
print(data2.shape)

print(data1[0][0])
# print(data2[0][0])


# 遍历数据
# for line in data[0, :]:
#     print(line)