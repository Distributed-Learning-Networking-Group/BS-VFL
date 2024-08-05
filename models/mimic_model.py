import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmimic.models import StandardLSTM


class ClientNet(nn.Module):
    def __init__(self, n_dim):
        super(ClientNet, self).__init__()
        self.fc = nn.Linear(n_dim, 76)

    def forward(self, x):
        x = self.fc(x)
        x = torch.flatten(x, 1)
        return x

class ServerNet(nn.Module):
    def __init__(self):
        super(ServerNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=76,
            hidden_size=16,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True,
        )

        self.final_layer = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.shape[0],48,-1)
        out, _ = self.lstm(x)
        out = self.final_layer(out[:, -1, :])
        return out.squeeze(1)