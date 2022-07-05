import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
from torch.nn import Embedding

class MinimalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


DATA = [
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9],
    [4, 6, 2, 9, 0],
    [1, 3, 8]
]

DATA = list(map(lambda x: torch.tensor(x), DATA))
print(DATA)
# 词典大小，包含了padding token 0
NUM_WORDS = 10
BATCH_SIZE = 3
LSTM_DIM = 5  # hidden dim

dataset = MinimalDataset(DATA)
data_loader = DataLoader(dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         collate_fn=lambda x: x)

# print(next(iter(data_loader)))

# iterate through the dataset:
for i, batch in enumerate(data_loader):
    print(f'{i}, {batch}')

it = iter(data_loader)
batch = next(it)
padded = pad_sequence(batch, batch_first=True)
print(f'[0] padded: \n{padded}\n')
batch = next(it)
padded = pad_sequence(batch, batch_first=True)
print(f'[1] padded: \n{padded}\n')

lens = Embedding(NUM_WORDS,LSTM_DIM)






# import torch
# import torch.utils.data as Data
# import numpy as np
#
# def collate_batch(batch_list):
#     assert type(batch_list) == list, f"Error"
#     batch_size = len(batch_list)
#     data = torch.cat([item[0] for item in batch_list]).reshape(batch_size, -1)
#     labels = torch.cat([item[1] for item in batch_list]).reshape(batch_size, -1)
#     return data, labels
#
# test = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
#
# inputing = torch.tensor(np.array([test[i:i + 3] for i in range(10)]))
# target = torch.tensor(np.array([test[i:i + 1] for i in range(10)]))
#
# torch_dataset = Data.TensorDataset(inputing,target)
# batch = 3
#
# loader = Data.DataLoader(dataset=torch_dataset,batch_size=batch,collate_fn=collate_batch)
#
# # for i,j in enumerate(loader):
# #     print(f"num {i} batch, item : {j}")
#
# for data, labels in loader:
#     print(f"data {data} batch,labels: {labels}")
#     print(f"data {data} batch,labels: {labels}")