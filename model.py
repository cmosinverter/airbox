import torch    
import torch.nn as nn

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2 , 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2 , 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomposition(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    

class GRU(nn.Module):
    def __init__(self, input_features, hidden_size, win_len):
        super(GRU, self).__init__()
        
        self.long = win_len
        self.short = win_len // 4
        
        self.gru1 = nn.GRU(input_size=input_features, hidden_size=hidden_size, batch_first=True, num_layers=2)
        self.gru2 = nn.GRU(input_size=input_features, hidden_size=hidden_size, batch_first=True, num_layers=2)
        
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size//2, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        xl = x
        xs = x[:, -self.short:, :]
        
        outl, _ = self.gru1(xl)
        outs, _ = self.gru2(xs)
        
        out = self.fc1(torch.cat((outl[:, -1, :], outs[:, -1, :]), dim=1))
        
        out = self.dropout(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_sequence_length):
        super(PositionalEncoding, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class SelfAttn(nn.Module):
    def __init__(self, d_model, nhead):
        super(SelfAttn, self).__init__()
        
        # Multi-Head Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(d_model, nhead)
        
        # Feed-Forward Layer
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        
        
    
    def forward(self, x):
        # Self-Attention
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-Forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x




class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(Encoder, self).__init__()
        
        self.positional_encoding = PositionalEncoding(d_model, max_sequence_length=24)
        self.layers = nn.ModuleList([SelfAttn(d_model, nhead) for _ in range(num_layers)])
        self.reg_head = nn.Linear(d_model, 1)
        
    def forward(self, x):
        pe = self.positional_encoding.forward()
        x = x + pe.unsqueeze(0).to(x.device)
        
        for layer in self.layers:
            x = layer(x)
        x = self.reg_head(x)
        return x[:, -1, :]

