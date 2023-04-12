import torch.nn as nn
import torch
import numpy as np

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


win_len = 6
X, y = create_sequences(data, win_len)

print(X.shape, y.shape)

conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
gru1 = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
fc1 = nn.Linear(64, 1)
out = conv1(X)
print(out.shape)
out = out.permute(0, 3, 1, 2).reshape(out.size(0), out.size(3), -1)
print(out.shape)
out, _ = gru1(out)
print(out.shape)
out = fc1(out[:, -1, :])
print(out.shape)
