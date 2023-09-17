import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from utils import *


class SesnorDataset(Dataset):
    def __init__(self, subset, dates, scaler, win_len, use_consecutive, source):
        self.subset = subset
        self.dates = dates
        self.scaler = scaler
        self.win_len = win_len
        self.use_consecutive = use_consecutive
        self.source = source
        
        spec_data, sgx_data, ref_data = readData()
        data = pd.concat([ref_data, sgx_data, spec_data], axis = 1)
        data = data.reindex(data.index, fill_value=np.nan)
        data.dropna(subset=self.subset, inplace=True)
        data = data.abs()
        data = data.loc[self.dates, self.subset]
        data = np.transpose(self.scaler.transform(data))
        self.X, self.y, self.dates = create_sequences(data, self.win_len, self.dates, self.use_consecutive)
        if self.source:
            self.domain_labels = np.ones((len(self.X), 1))
        else:
            self.domain_labels = np.zeros((len(self.X), 1))
            
    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], torch.tensor(self.domain_labels[idx], dtype=torch.float32)