import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

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
        mobilenet = mobilenet_v2(pretrained=True)
        mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 100)
        self.mobilenet = mobilenet

    def forward(self,x):
        x = x.view(x.shape[0],3,84,-1)
        x = self.mobilenet(x)
        return x