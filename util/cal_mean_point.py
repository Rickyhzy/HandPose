import pandas as pd
import numpy as np
import os
import sys
from itertools import *


# a = 'a_sample_0_11.txt'
# b = 'a_sample_0_1.txt'
# print(a[11:-4],b[11:-4])

dir_path = '../data/dynamic/data/'
files = os.listdir(dir_path)
files.sort(key=lambda x: int(x[11:-4])+(ord(x[0])-96)*10000)
# print(files)
# data = pd.read_csv(path, header=None)
series = []
length = []
sum_data = 0
k = 1
temp = 0
for i in range(len(files)):
    # print(files[i])
    data = pd.read_csv(dir_path + files[i], header=None)
    series.append(len(data))
    sum_data += len(data)

    if k > 80:
        k = 0
        print(i, k)
        print(f'第{temp}个动作')
        print(series, len(series), sum(series), sum_data, sum_data/len(series))
        length.append(sum_data/len(series))
        series = []
        sum_data = 0
        temp += 1
    k += 1

print(length)
print(sum(length)/len(length))
