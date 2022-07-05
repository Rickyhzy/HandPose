import pandas as pd
import numpy as np
import os
import sys

dir_path = '../data/private/'
dir_path_new = '../data/dataset/'
files = os.listdir(dir_path)
head = ['flex1','flex2','flex3','flex4','flex5','accx','accy','accz','gyrx','gyry','gyrz','label']

def rename():
    name = ['7EIGHT.csv', '16FIRE.csv', '4FIVE.csv', '3FOUR.csv', '13FREEZE.csv', '14HEAR.csv', '11I.csv', '8NINE.csv','17OK.csv', '0ONE.csv', '6SEVEN.csv', '5SIX.csv', '12STOP.csv', '9TEN.csv', '2THREE.csv', '1TWO.csv','10U.csv', '15WATCH.csv']
    for i in range(len(files)):
        path = dir_path + files[i]
        path_new = dir_path + name[i]
        print(path)
        os.rename(path, path_new)

def to_label_data():
    for i in range(len(files)):
        path = dir_path + files[i]
        print(path)
        label = int(files[i][0:2])
        print(label)
        df = pd.read_csv(path, header=None)
        df.insert(loc=11, column='label', value=label)
        print(df)
        if (os.path.exists(path)):
            # 存在，则删除文件
            os.remove(path)
        df.to_csv('../data/dataset/{}'.format(files[i]), index_label=None, header=head, index=None)

#数据集合并
def merge_dataset():
    all_data = []
    print(len(files))
    for i in range(len(files)):
        path = dir_path_new + files[i]
        print(path)
        label = int(files[i][0:2])
        print(label)
        df = pd.read_csv(path)
        all_data.append(df)
    result = pd.concat(all_data)
    print(result)
    result.to_csv('../data/dataset/all_data.csv', index_label=None, index=None)

# to_label_data()
# merge_dataset()

#获取文件当前工作目录路径（绝对路径）
# print(sys.argv[0])
# print(__file__)
