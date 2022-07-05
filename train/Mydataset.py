import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True)

def data_prepare(mode='train'):
    data = pd.read_csv('../data/dataset/all_data.csv')
    # print(data)

    # 11特征
    data_set = data.iloc[:, :11]  # band
    label_set = data.iloc[:, -1]

    #训练集与测试集切分
    train_data,test_data,train_label,test_label = train_test_split(data_set,label_set,test_size=0.4)

    # #归一化操作  standard
    # scaler = StandardScaler().fit(train_data)
    # train_data_tf = scaler.transform(train_data)
    # test_data_tf = scaler.transform(test_data)

    # 归一化操作  minmax
    scaler = MinMaxScaler()
    train_data_tf = scaler.fit_transform(train_data)
    test_data_tf = scaler.fit_transform(test_data)

    if mode.__eq__('train'):
        Data = train_data_tf
        Label = np.array(train_label).reshape(-1, 1)
        return Data, Label

    if mode.__eq__('test'):
        Data = test_data_tf
        Label = np.array(test_label).reshape(-1, 1)
        return Data, Label


# 初始化dataset
class MyDataset(Dataset):
    def __init__(self, Data, Label):#初始化数据集与标签
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        # dataset = torch.Tensor(self.Data[index]).unsqueeze(0)
        dataset = torch.Tensor(self.Data[index])   #bpnet
        label = torch.LongTensor(self.Label[index]).squeeze(-1)
        return dataset,label

if __name__ == '__main__':
    data_prepare()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Data, Label = data_prepare(mode='test')
    print(Data.shape,Label.shape)
    dataset_test = MyDataset(Data, Label)

    Data, Label = data_prepare(mode='train')
    print(Data.shape, Label.shape)
    dataset_train = MyDataset(Data, Label)

    trainloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=64, shuffle=True)

    for i, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device)
        print(data.shape,label.shape)

    # print(dataset)
    #
    # # 划分测试集和训练集torch版本
    # train_size = int(len(dataset) * 0.75)
    # test_size = len(dataset) - train_size
    # train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    # print(len(train_set),len(test_set))
    #
    # 数据加载器

    pass