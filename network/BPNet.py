import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class BP(nn.Module):
    def __init__(self):
        super(BP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(11, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.Linear(64,128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128,64),
            nn.Linear(64, 18)
        )

    def forward(self, x):
        out = self.layer(x)
        return out

if __name__ == '__main__':
    input = torch.randn(64, 11)
    bp = BP()
    out = bp(input)
    print(out.shape)