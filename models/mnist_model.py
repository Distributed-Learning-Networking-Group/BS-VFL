import torch
import torch.nn as nn
from models.resnet import ResNet18

class ClientNet(nn.Module):
    def __init__(self):
        super(ClientNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self,x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return x

class ServerNet(nn.Module):
    def __init__(self):
        super(ServerNet, self).__init__()
        self.resnet18 = ResNet18(in_channel=1)
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.fc1 = nn.Linear(64*12*12, 128)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self,x):
        x = x.view(x.shape[0],1,28,-1)
        x = self.resnet18(x)
        # x = x.view(x.shape[0],1,28,-1)
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)
        return x