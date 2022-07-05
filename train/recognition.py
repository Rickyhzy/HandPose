import torch
import numpy as np
import pandas as pd
from BPNet import *
from torch.utils.data import DataLoader
from Mydataset import *
import os
import matplotlib.pyplot as plt


def train(model, device, loss_fun, optimizer, trainloader, testloader, epoch_num):
    # train_loss = 0
    # train_acc = 0
    loss_show = []
    acc_list = []
    weights = '../model/net.pth'

    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights))
        print("loading successful")

    for epoch in range(epoch_num):
        train_loss = 0
        model.train()
        for i,(data,label) in enumerate(trainloader):
            data, label = data.to(device), label.to(device)
            # print(i, data.shape,label.shape)

            #前向计算，计算loss
            out = model(data)
            loss = loss_fun(out, label)
            # print(f'{epoch}-{i}-train_loss:{loss.item()}')

            #反向传播，梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #损失/准确率计算
            train_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, train_loss))
                loss_show.append(train_loss)
                train_loss = 0.0

        #以下是测试集准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for tmp in testloader:
                data, label = tmp
                outputs = model(data)
                _, predicted = torch.max(outputs.data, dim=1)
                # print(predicted,predicted.shape)
                total += label.size(0)
                correct += predicted.eq(label).float().sum()
                # correct += (predicted == label).sum().item()
        print('accuracy on test set: %d %% ' % (100 * correct / total))
        acc = 100 * correct / total
        acc_list.append(acc)
        # if epoch % 1 == 0:
        #     print('Epoch:{},   acc:{},   loss:{}'.format(epoch, accuracy, train_loss))
        #     torch.save(model.state_dict(), f'model/net.pth')
        #     print('save successfully')
    loss_to_show = pd.DataFrame(data=loss_show)
    loss_to_show.to_csv('BP_100F_{:.2f}%_loss.csv'.format(acc))
    acc_to_show = pd.DataFrame(data=acc_list)
    acc_to_show.to_csv('BP_100F_{:.2f}%_acc.csv'.format(acc))

    plt.figure('GS-DL', dpi=100)
    # plt.subplot(211)
    plt.title('loss')
    plt.plot(loss_show, linewidth='2',  color='orchid')

    plt.figure('GS-DL2', dpi=100)
    # plt.subplot(212)
    plt.title('accuracy')
    plt.plot(acc_list, linewidth='2', color='wheat')
    plt.show()


def test(test_loader):
    acc_list = []
    correct = 0
    total = 0
    with torch.no_grad():
        for tmp in test_loader:
            data, labels = tmp
            outputs = model(data)
            _, predicted = torch.max(outputs.data, dim=1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100*correct/total
            acc_list.append(acc)
    print('accuracy on test set: %d %% ' % (100*correct/total))



if __name__ == '__main__':
    #模型参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BP().to(device)   #BPnet
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fun = torch.nn.CrossEntropyLoss()

    #数据准备
    Data, Label = data_prepare(mode='train')
    dataset_train = MyDataset(Data, Label)
    Data, Label = data_prepare(mode='test')
    dataset_test = MyDataset(Data, Label)
    trainloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=64, shuffle=True)

    #训练
    train(model,device,loss_fun,optimizer,trainloader,testloader,epoch_num=180)