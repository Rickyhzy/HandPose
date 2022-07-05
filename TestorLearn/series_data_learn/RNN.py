import torch
import torch.nn as nn
import numpy as np

single_rnn = nn.RNN(4, 5, 1,batch_first=True)
input = torch.randn(1, 2, 4)
print(input)
output, h_n = single_rnn(input)
print(output, output.shape)
print(h_n, h_n.shape)
print('*************')

# x = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
# torch_x = torch.FloatTensor([x])
# output, h_n = single_rnn(torch_x)
# print(output, output.shape)
# print(h_n, h_n.shape)

print('*************')
#双向RNN
bi_rnn = nn.RNN(4, 3, 1, batch_first=True, bidirectional=True)
bi_output, bi_h_n = bi_rnn(input)
print(bi_output, bi_output.shape)
print(bi_h_n, bi_h_n.shape)
