import torch.nn as nn
import torch.nn.functional as F

class ClientNet(nn.Module):
    def __init__(self,n_dim):
        super(ClientNet, self).__init__()
        self.fc = nn.Linear(n_dim, n_dim)

    def forward(self,x):
        x = self.fc(x)
        x = F.relu(x)
        return x

class ServerNet(nn.Module):
    def __init__(self):
        super(ServerNet, self).__init__()
        self.fc1 = nn.Linear(123, 10)
        self.fc2 = nn.Linear(10,2)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

