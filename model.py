import torch.nn as nn

class CNN_GRU(nn.Module):
    def __init__(self, kernel_width, input_features, hidden_size):
        super(CNN_GRU, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(input_features, kernel_width))
        self.gru1 = nn.GRU(input_size=32, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1)

        
    def forward(self, x):

        out = self.conv1(x)
        out = out.permute(0, 3, 1, 2).reshape(out.size(0), out.size(3), -1)
        out, _ = self.gru1(out)
        out = self.fc1(out[:, -1, :])

        return out

