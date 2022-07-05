# import pandas as pd
# import numpy as np
# # import argparse
# #
# # parser = argparse.ArgumentParser()
# # parser.add_argument('--sparse', action='store_true',default=False, help='GAT with sparse version or not.')
# # parser.add_argument('--seed', type=int, default=72, help='Random seed.')
# # parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
# #
# # args = parser.parse_args()
# # print (f"sparse = {args.sparse}, seed = {args.seed}, epochs = {args.epochs}")
#
#
# def sliding_window(train, sw_width=7, n_out=7, in_start=0):
#     '''
#     该函数实现窗口宽度为7、滑动步长为1的滑动窗口截取序列数据
#     '''
#     data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))  # 将以周为单位的样本展平为以天为单位的序列
#     X, y = [], []
#
#     for _ in range(len(data)):
#         in_end = in_start + sw_width
#         out_end = in_end + n_out
#
#         # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
#         if out_end < len(data):
#             # 训练数据以滑动步长1截取
#             train_seq = data[in_start:in_end, 0]
#             train_seq = train_seq.reshape((len(train_seq), 1))
#             X.append(train_seq)
#             y.append(data[in_end:out_end, 0])
#         in_start += 1
#
#     return np.array(X), np.array(y)
#
#
# def foo():
#     print("starting...")
#     while True:
#         res = yield 4
#         print("res:",res)
# g = foo()
# print(g)
# print(next(g))
# print("*"*20)
# print(next(g))
#
# def foo(num):
#     print("starting...")
#     while num<10:
#         num=num+1
#         yield num
# for n in foo(0):
#     print(n)
# x = [1,2,3]
# b = lambda x:len(x)
# a = b(x)
# print(a)

import numpy as np
import torch
import pandas as pd

# torch.seed()
# a = np.random.randn(1,2,3)
# b = torch.randn(1, 2, 3)
# print(a)
# bs, T, size = a.shape
# print(bs, T, size)
# print(a.shape)
# for t in range(T):
#     x = a[:, t, :]
#     print(x)
#
# print(b)
# temp = b.unsqueeze(0)
# print(temp)
# temp2 = temp.tile(1, 2, 1)
# print(temp2)
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

a = np.arange(9).reshape(3, 3)
print(a)
b = a.transpose()#转置
print(b, b.shape)
c = np.ones(9).reshape(3,3)
print(c)
k = np.transpose(
    [a, c]
)
print(k)

data = [torch.tensor([9]),
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([5, 6])]

seq_len = [s.size(0) for s in data]
data = pad_sequence(data, batch_first=True)
data = pack_padded_sequence(data, seq_len, batch_first=True, enforce_sorted=False)
print(data)



import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# x = np.linspace(0, 10, 11)
# print(x)
# y = np.sin(x)
# plt.plot(x,y,'o')
# # plt.show()
# x_new = np.linspace(0,10,101)
# print(x_new)
# kind_lst = ['nearest', 'zero', 'linear', 'cubic', 'previous',  'next']
# for k in kind_lst:
#     f = interp1d(x,y,kind=k)
#     y_new = f(x_new)
#     plt.plot(x_new,y_new,label=k)
# plt.legend(loc='lower right')
# plt.show()


# X = np.linspace(-1, 1, 200)
# np.random.shuffle(X)
# print(X)
# Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plt.scatter(X, Y)
# plt.show()
# X_train, Y_train = X[:160], Y[:160]     # 把前160个数据放到训练集
# X_test, Y_test = X[160:], Y[160:]
# print(X_train.shape, Y_train.shape)

a = np.random.randint(0,9,(2,20))
b = np.array(a)
print(b, type(b))

from scipy import signal

x = [i for i in range(1, 201)]
y = signal.resample(x, 100)
print(len(x), len(y))
tx = np.linspace(0, 10, 200, endpoint=False)
ty = np.linspace(0, 10, 100, endpoint=False)
plt.plot(tx, x, '-')
plt.plot(ty, y, '.-')

t = np.linspace(0, 5, 100)
print(t[::4])

plt.show()

