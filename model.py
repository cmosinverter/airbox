import torch.nn as nn
import torch
import torch.optim as optim
import math
from torch.autograd import Function

class GRU(nn.Module):
    def __init__(self, input_features, hidden_size):
        super(GRU, self).__init__()

        self.gru1 = nn.GRU(input_size=input_features, hidden_size=hidden_size, batch_first=True)
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
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=(input_features, kernel_width))
        self.gru0 = nn.GRU(input_size=conv_out_channels, hidden_size=hidden_size, batch_first=True)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=(input_features, kernel_width))
        self.gru1 = nn.GRU(input_size=conv_out_channels, hidden_size=hidden_size, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 1)

        
    def forward(self, x1):
        out = self.conv1(x1)
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
        self.gru1 = nn.GRU(input_size=conv_out_channels+1, hidden_size=hidden_size, batch_first=True, num_layers=1)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)


        
    def forward(self, x):
        x_t = x[:, :, 0, self.kernel_width - 1:].unsqueeze(1)
        out = self.conv1(x)
        out = torch.cat((out, x_t), dim=1)
        out = out.permute(0, 3, 1, 2).reshape(out.size(0), out.size(3), -1)
        out, _ = self.gru1(out)
        out = self.fc1(out[:, -1, :])
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
    return GradientReversalFunction.apply(x, alpha)

class DAN(nn.Module):
    def __init__(self, kernel_width, input_features, hidden_size, conv_out_channels=32):
        super(DAN, self).__init__()

        # Feature Extraction
        self.kernel_width = kernel_width
        self.conv_out_channels = conv_out_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_out_channels, kernel_size=(input_features, kernel_width))
        self.relu1 = nn.ReLU()
        self.gru1 = nn.GRU(input_size=conv_out_channels, hidden_size=hidden_size, batch_first=True, num_layers=1)
        
        # Regression
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        
        # Domain Classification 
        self.fc3 = nn.Linear(hidden_size, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)  # Renamed for clarity
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x, alpha):
        # Feature Extraction
        out = self.conv1(x)
        out = self.relu1(out)
        out = out.permute(0, 3, 1, 2).reshape(out.size(0), out.size(3), -1)
        out, _ = self.gru1(out)
        encoded = out[:, -1, :]
        
        # Domain Adversarial Training: Gradient Reversal
        reversed_encoded = grad_reverse(encoded, alpha)
        
        # Regression
        reg = self.fc1(encoded)
        reg = self.relu2(reg)
        reg = self.fc2(reg)
        
        # Domain Classification 
        domain = self.fc3(reversed_encoded)  # Use reversed_encoded
        domain = self.relu3(domain)
        domain = self.fc4(domain)  # Use fc4 instead of fc3
        domain = self.sigmoid(domain)
        
        return reg, domain
