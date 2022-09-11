import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from util.mypath import Path
import sys
import os


class CnnDataset(Dataset):
    def __init__(self, dataset='elas_hand', split='train', preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.root_dir, split)
        self.split = split
        print(folder)


        # 这里不需要预处理
        if (not self.check_preprocess()) or preprocess:
            print('preprocess is doing')
            self.preprocess()

        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder + '/', label + '/', fname))
                labels.append(label)

        print(self.fnames, labels)
        assert len(labels) == len(self.fnames)
        print('Number of {} dataSamples: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        print(self.label2index, self.label_array)

        if dataset == "elas_hand":
            if not os.path.exists('../dataloaders/elas_labels.txt'):
                with open('../dataloaders/elas_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id + 1) + ' ' + label + '\n')

        if dataset == 'elas_hand':
            if not os.path.exists('dataloader/ucf_labels.txt'):
                print('OK')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = pd.read_csv(self.fnames[index], header=None)
        buffer1 = buffer.iloc[:, 0:5]
        buffer2 = buffer.iloc[:, 5:]
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = pd.read_csv(self.fnames[index], header=None)
            buffer1 = buffer.iloc[:, 0:5]
            buffer2 = buffer.iloc[:, 5:]

        # print(buffer1.shape, buffer2.shape)
        # buffer = self.normalize(buffer)
        # buffer = self.to_tensor(buffer)

        return torch.from_numpy(buffer1.values), torch.from_numpy(buffer2.values), torch.from_numpy(labels)

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def preprocess(self):
        pass

    def check_preprocess(self):
        if not os.path.exists(self.output_dir):
            return False
        else:
            return True

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from torch.utils.data import DataLoader

    train_data = CnnDataset(dataset='elas_hand', split='train', preprocess=False)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs1 = sample[0]
        inputs2 = sample[1]
        labels = sample[2]
        inputs1, labels = inputs1.to(device), labels.to(device)
        print(inputs1.shape, inputs2.shape, labels.shape)

    # dataset = CnnDataset()
    # print(dataset.__len__())

    # root_dir, output_dir = Path.db_dir('elas_hand')
    # folder = os.path.join(root_dir, 'train')
    # print(folder)
    # dirs = os.listdir(folder)

