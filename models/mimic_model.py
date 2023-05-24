import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ClientNet(nn.Module):
    def __init__(self, n_dim):
        super(ClientNet, self).__init__()
        self.fc = nn.Linear(n_dim, n_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)
        return x


class ServerNet(nn.Module):
    def __init__(self, hidden_dim=128, n_classes=2, num_layers=1, dropout_rate=0, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        if bidirectional:
            self.hidden_dim = hidden_dim // 2

        if num_layers == 1:
            dropout_rate = 0

        self.lstm_layer = nn.LSTM(48*76, self.hidden_dim, num_layers=num_layers,
                                  dropout=dropout_rate, bidirectional=bidirectional, batch_first=True)

        linear_input = self.hidden_dim
        if bidirectional:
            linear_input *= 2

        self.final_layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(linear_input, n_classes),
            nn.Sigmoid(),
        )

    def forward(self, data):
        seq = data.view(len(data), 1, -1)
        lens = torch.tensor([len(i) for i in seq]).to(torch.int64)
        # print("seq:", seq.size())
        # print("lens:", lens)
        packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
        # print("packed.data:", packed.data.size())
        # print("packed.len:", packed.batch_sizes)

        h_dim = 2 if self.bidirectional else 1

        z, (ht, ct) = self.lstm_layer(packed)

        seq_unpacked, lens_unpacked = pad_packed_sequence(z, batch_first=True)

        output = self.final_layer(
            torch.vstack(
                [seq_unpacked[i, int(l) - 1] for i, l in enumerate(lens_unpacked)]
            )
        )

        return output

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "bidirectional": self.bidirectional,
        }

'''
------------------------------------------------------------------------------------------------------------------------
'''

# class ClientNet(nn.Module):
#     def __init__(self, n_dim, hidden_dim=256, n_classes=2, num_layers=1, dropout_rate=0, bidirectional=False):
#         super(ClientNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.dropout_rate = dropout_rate
#         self.bidirectional = bidirectional

#         if bidirectional:
#             self.hidden_dim = hidden_dim // 2

#         # if num_layers == 1:
#         #     dropout_rate = 0

#         self.bi_lstm = nn.LSTM(n_dim, self.hidden_dim, num_layers=1, dropout=dropout_rate, bidirectional=True, batch_first=True)
#         # self.lstm = nn.LSTM(2*self.hidden_dim, self.hidden_dim, num_layers=num_layers, dropout=dropout_rate, bidirectional=False, batch_first=True)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         z, (ht, ct) = self.bi_lstm(x)
#         # output, (_, _) = self.lstm(z)
#         return z


# class ServerNet(nn.Module):
#     def __init__(self, n_input):
#         super().__init__()
#         self.final_layer = nn.Sequential(
#             # nn.Dropout(0),
#             nn.Linear(n_input*256*2, 2),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         output = self.final_layer(x)

#         return output

'''
------------------------------------------------------------------------------------------------------------------------
'''


# class ClientNet(nn.Module):
#     def __init__(self, n_dim, hidden_dim=128, n_classes=2, num_layers=2, dropout_rate=0, bidirectional=True):
#         super(ClientNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         # self.num_layers = num_layers
#         self.num_layers = 2
#         self.dropout_rate = dropout_rate
#         self.bidirectional = bidirectional
#
#         if bidirectional:
#             self.hidden_dim = hidden_dim // 2
#
#         if num_layers == 1:
#             dropout_rate = 0
#
#         self.lstm_layer = nn.LSTM(n_dim, self.hidden_dim, num_layers=num_layers,
#                                   dropout=dropout_rate, bidirectional=bidirectional, batch_first=True)
#
#         linear_input = self.hidden_dim
#         if bidirectional:
#             linear_input *= 2
#
#         self.final_layer = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             # nn.Linear(linear_input, n_classes),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         seq = x.view(len(x), 1, -1)
#         lens = torch.tensor([len(i) for i in seq]).to(torch.int64)
#         # print("seq:", seq.size())
#         # print("lens:", lens)
#         packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
#         # print("packed.data:", packed.data.size())
#         # print("packed.len:", packed.batch_sizes)
#
#         h_dim = 2 if self.bidirectional else 1
#
#         z, (ht, ct) = self.lstm_layer(packed)
#
#         seq_unpacked, lens_unpacked = pad_packed_sequence(z, batch_first=True)
#
#         output = self.final_layer(
#             torch.vstack(
#                 [seq_unpacked[i, int(l) - 1] for i, l in enumerate(lens_unpacked)]
#             )
#         )
#         return output
#
#
# class ServerNet(nn.Module):
#     def __init__(self, n_input, hidden_dim=256, n_classes=2, num_layers=1, dropout_rate=0, bidirectional=True):
#         super(ServerNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.dropout_rate = dropout_rate
#         self.bidirectional = bidirectional
#
#         self.final_layer = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.Linear(n_input*128, n_classes),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         output = self.final_layer(x)
#
#         return output
