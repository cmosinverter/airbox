import torch.nn as nn
import torch

class GRU(nn.Module):
    def __init__(self, input_features, hidden_size):
        super(GRU, self).__init__()

        self.gru1 = nn.GRU(input_size=3, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1)

        
    def forward(self, x):
        out = x.permute(0, 3, 1, 2).reshape(x.size(0), x.size(3), -1)
        out, _ = self.gru1(out)
        out = self.fc1(out[:, -1, :])

        return out


class Simple_CNN_GRU(nn.Module):
    def __init__(self, kernel_width, input_features, hidden_size, conv_out_channels=32):
        super(Simple_CNN_GRU, self).__init__()

        self.kernel_width = kernel_width
        self.conv_out_channels = conv_out_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=(input_features, kernel_width))
        self.gru1 = nn.GRU(input_size=conv_out_channels, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1)

        
    def forward(self, x):
        out = self.conv1(x)
        out = out.permute(0, 3, 1, 2).reshape(out.size(0), out.size(3), -1)
        out, _ = self.gru1(out)
        out = self.fc1(out[:, -1, :])

        return out

class CNN_GRU(nn.Module):
    def __init__(self, kernel_width, input_features, hidden_size, conv_out_channels=32):
        super(CNN_GRU, self).__init__()

        self.kernel_width = kernel_width
        self.conv_out_channels = conv_out_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=(input_features, kernel_width))
        self.gru1 = nn.GRU(input_size=conv_out_channels+1, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1)

        
    def forward(self, x):
        # print("The shape of input: ", x.shape)
        x_t = x[:, :, 0, self.kernel_width - 1:].unsqueeze(1)
        # print("The shape of x_T: ", x_t.shape)
        out = self.conv1(x)
        # print("The shape of encoded feature: ", out.shape)
        out = torch.cat((out, x_t), dim=1)
        # print("The shape of all feature: ", out.shape)
        out = out.permute(0, 3, 1, 2).reshape(out.size(0), out.size(3), -1)
        out, _ = self.gru1(out)
        # print("The shape of GRU output: ", out.shape)
        out = self.fc1(out[:, -1, :])
        # print("The shape of output: ", out.shape)
        return out

