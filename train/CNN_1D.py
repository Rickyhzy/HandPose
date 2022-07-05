import torch
import pandas as pd
import numpy as np
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(5, 8, kernel_size=3, stride=1),
            nn.Conv1d(8, 8, kernel_size=3, stride=1),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(5, 8, kernel_size=3, stride=1),
            nn.Conv1d(8, 8, kernel_size=3, stride=1),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=3, stride=1),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(104, 100),
            nn.Linear(100, 10)
        )

    def forward(self, x1, x2):
        out1 = self.layer1(x1)
        out2 = self.layer2(x2)
        out3 = torch.cat((out1, out2), dim=1)
        out4 = self.layer3(out3)
        nn
        return out4
    pass


if __name__ == '__main__':
    # X_train = np.linspace(0, 10, 200)
    # print(X_train)
    X_train1 = torch.rand((64, 5, 60))
    X_train2 = torch.rand((64, 5, 60))
    print(X_train1.shape)
    Y_train = np.random.randint(0, 10, 100)
    print(Y_train, type(Y_train), Y_train.shape)
    Y_train = torch.Tensor(Y_train)
    print(Y_train, type(Y_train))
    Y_train = Y_train.clone().detach()
    # Y_train = torch.tensor(Y_train)
    print(Y_train, type(Y_train))

    net = CNN1D()
    out = net(X_train1, X_train2)
    print(out.shape)
