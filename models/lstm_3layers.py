import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load h5 spectra and model parameters
import numpy as np
import pandas as pd
import os
import h5py
import tables
from sklearn.metrics import r2_score
import time

from pathlib import Path
PATH = Path("/mnt/wamri/brainlabs/lexie/spectrome-ai/data")

def parse_data(path: str):
    # load data from the disk
    data = np.load(PATH/path, allow_pickle=True)
    df = pd.DataFrame(data=data, index=data[:,0:5])
    df = df.drop([0, 1, 2, 3, 4], axis=1)
    df = df.reset_index()
    df.columns = ['params', 'freq']
    return df


# Pytorch Dataset
class SPDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.x = np.abs(np.array(self.df['freq']))
        self.y = np.array(list(self.df['params']))

    def __len__(self):
        """ Length of the dataset """
        N, _ = self.df.shape
        return N

    def __getitem__(self, idx):
        """ returns x[idx], y[idx] for this dataset"""
        x = self.x[idx]
        y = self.y[idx]
        return x, y


# Model
class LSTMNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, seq_len, batch_size, drop_prob=0.5):
        super(LSTMNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.linear1 = nn.Linear(hidden_dim, 1)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear4 = nn.Linear(hidden_dim, 1)
        self.linear5 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        lstm_out, (lstm_ht, lstm_ct) = self.lstm(x)
        lstm_ht[-1] = self.dropout(lstm_ht[-1])
        out1 = self.linear1(lstm_ht[-1])
        out2 = self.linear2(lstm_ht[-1])
        out3 = self.linear3(lstm_ht[-1])
        out4 = self.linear4(lstm_ht[-1])
        out5 = self.linear5(lstm_ht[-1])
        out = torch.cat([out1, out2, out3, out4, out5], 1)
        return out


