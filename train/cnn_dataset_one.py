import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from util.mypath import Path
import sys
import os


class CnnDataset(Dataset):
    def __init__(self, dataset='dynamic_unify', split='train', preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.split = split
        print(folder)

        if not self.check_preprocess():
            print('preprocess is doing')
        else:
            print('do not')

        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        print(self.fnames, labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return 0


    def check_preprocess(self):
        if not os.path.exists(self.output_dir):
            return False
        else:
            return True
if __name__ == '__main__':
    # dataset = CnnDataset()
    root_dir, output_dir = Path.db_dir('elas_hand')
    folder = os.path.join(root_dir, 'train')
    print(folder)
    os.listdir(folder)