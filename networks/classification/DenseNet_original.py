import torch
import torch.nn as nn
import torch.nn.parallel
from torchvision import models

class DenseNet121_original(nn.Module):
    def __init__(self, net_config, num_classes=2):
        super(DenseNet121_original, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet_modules = list(models.densenet121(pretrained=True).children())
        self.hidden_modules = nn.Sequential(*self.densenet_modules[0][1:])
        self.gap = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, input):
        batch_size = input.size(0)
        input = input.unsqueeze(1)
        out = self.conv1(input)
        out = self.hidden_modules(out)
        out = self.gap(out)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out
    
class DenseNet169_original(nn.Module):
    def __init__(self, net_config, num_classes=2):
        super(DenseNet169_original, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet_modules = list(models.densenet169(pretrained=True).children())
        self.hidden_modules = nn.Sequential(*self.densenet_modules[0][1:])
        self.gap = nn.AvgPool2d(7)
        self.fc = nn.Linear(1664, num_classes)
    
    def forward(self, input):
        batch_size = input.size(0)
        input = input.unsqueeze(1)
        out = self.conv1(input)
        out = self.hidden_modules(out)
        out = self.gap(out)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out

class DenseNet201_original(nn.Module):
    def __init__(self, net_config, num_classes=2):
        super(DenseNet201_original, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet_modules = list(models.densenet201(pretrained=True).children())
        self.hidden_modules = nn.Sequential(*self.densenet_modules[0][1:])
        self.gap = nn.AvgPool2d(7)
        self.fc = nn.Linear(1920, num_classes)
    
    def forward(self, input):
        batch_size = input.size(0)
        input = input.unsqueeze(1)
        out = self.conv1(input)
        out = self.hidden_modules(out)
        out = self.gap(out)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out