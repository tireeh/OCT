import torch
import torch.nn as nn
import torch.nn.parallel
from torchvision import models
import torch.utils.model_zoo as model_zoo
import pdb

class ResNet18_original(nn.Module):
    def __init__(self, net_config, num_classes):
        super(ResNet18_original, self).__init__()
        self.net_config = net_config
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_modules = list(models.resnet18(pretrained=True).children())
        self.hidden_modules = nn.Sequential(*self.resnet_modules[1:-1])
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, input):
        batch_size = input.size(0)
        input = input.unsqueeze(1)
        out = self.conv1(input)
        out = self.hidden_modules(out).view(batch_size, -1)
        out = self.fc(out)
        return out

class ResNet34_original(nn.Module):
    def __init__(self, net_config, num_classes):
        super(ResNet34_original, self).__init__()
        self.net_config = net_config
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_modules = list(models.resnet34(pretrained=True).children())
        self.hidden_modules = nn.Sequential(*self.resnet_modules[1:-1])
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, input):
        batch_size = input.size(0)
        input = input.unsqueeze(1)
        out = self.conv1(input)
        out = self.hidden_modules(out).view(batch_size, -1)
        out = self.fc(out)
        return out

class ResNet50_original(nn.Module):
    def __init__(self, net_config, num_classes):
        super(ResNet50_original, self).__init__()
        self.net_config = net_config
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_modules = list(models.resnet50(pretrained=True).children())
        self.hidden_modules = nn.Sequential(*self.resnet_modules[1:-1])
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, input):
        batch_size = input.size(0)
        input = input.unsqueeze(1)
        out = self.conv1(input)
        out = self.hidden_modules(out).view(batch_size, -1)
        out = self.fc(out)
        return out