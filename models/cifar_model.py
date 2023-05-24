import torch
import torch.nn as nn
from models.resnet import ResNet18

class ClientNet(nn.Module):
    def __init__(self):
        super(ClientNet, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self,x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return x

class ServerNet(nn.Module):
    def __init__(self):
        super(ServerNet, self).__init__()
        self.resnet18 = ResNet18(in_channel=3)
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        # self.fc1 = nn.Linear(4 * 4 * 64, 64)
        # self.fc2 = nn.Linear(64, 10)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self,x):
        x = x.view(x.shape[0],3,32,-1)
        x = self.resnet18(x)
        # x = x.view(x.shape[0],3,32,-1)
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = self.dropout(x)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = self.dropout(x)
        # x = self.conv3(x)
        # x = self.dropout(x)
        # x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        return x
    