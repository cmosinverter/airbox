import torch.nn as nn
import torch
import numpy as np
from model import CNN_GRU

data = torch.randn(1468, 4)
data = torch.transpose(data, 0, 1)


def create_sequences(data, window_len):
    xs = []
    ys = []

    for i in range(data.shape[1]-window_len+1):
        x = data[:-1, i:i+window_len]
        y = data[-1:, i+window_len-1]
        xs.append(x)
        ys.append(y)

    return torch.unsqueeze(torch.tensor(np.stack(xs), dtype=torch.float32), dim=1), torch.tensor(np.stack(ys), dtype=torch.float32)



input_features = data.shape[0] - 1
kernel_width = 12
win_len = 60
batch_size = 32
hidden_size = 32

model = CNN_GRU(kernel_width=kernel_width, input_features=input_features, hidden_size=hidden_size)
X, y = create_sequences(data, win_len)
X_batch, y_batch = X[:batch_size], y[:batch_size]

outputs = model(X_batch)
print(outputs.shape, y_batch.shape)


