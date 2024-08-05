import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.resnet18 = ResNet18(in_channel=3,num_classes=10)

    def forward(self,x):
        x = x.view(x.shape[0],3,32,-1)
        x = self.resnet18(x)

        return x