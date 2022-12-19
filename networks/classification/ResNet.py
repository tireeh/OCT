import torch
import torch.nn as nn
import torch.nn.parallel
from torchvision import models

class ResNet34(nn.Module):
    def __init__(self, net_config, num_classes):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 3, 3, 1, 1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU())
        
        self.resnet_modules = list(models.resnet34(pretrained=True).children())
        self.pre_module = nn.Sequential(*self.resnet_modules[:net_config["ensemble_idx"]])
        self.post_module = nn.Sequential(self.resnet_modules[net_config["ensemble_idx"]])
        self.gap = nn.AvgPool2d(net_config["feature_size"])
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, input):
        input = input.unsqueeze(2)
        batch_size = input.size(0)
        split_num = input.size(1)
        images = torch.chunk(input, split_num, 1)
        out = []
        for image in images:
            image = torch.squeeze(image, 1)
            image = self.conv1(image)
            image = self.pre_module(image)
            out.append(image)
        out = torch.stack(out, 1)
        [out, max_idx] = torch.max(out, 1)  
        out = self.post_module(out)
        out = self.gap(out).view(batch_size, -1)
        out = self.fc(out)
        return out

class ResNet50(nn.Module):
    def __init__(self, net_config, num_classes):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 3, 3, 1, 1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU())
        
        self.resnet_modules = list(models.resnet50(pretrained=True).children())
        self.pre_module = nn.Sequential(*self.resnet_modules[:net_config["ensemble_idx"]])
        self.post_module = nn.Sequential(self.resnet_modules[net_config["ensemble_idx"]])
        self.gap = nn.AvgPool2d(net_config["feature_size"])
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, input):
        input = input.unsqueeze(2)
        batch_size = input.size(0)
        split_num = input.size(1)
        images = torch.chunk(input, split_num, 1)
        out = []
        for image in images:
            image = torch.squeeze(image, 1)
            image = self.conv1(image)
            image = self.pre_module(image)
            out.append(image)
        out = torch.stack(out, 1)
        [out, max_idx] = torch.max(out, 1)  
        out = self.post_module(out)
        out = self.gap(out).view(batch_size, -1)
        out = self.fc(out)
        return out

class ResNet101(nn.Module):
    def __init__(self, net_config, num_classes):
        super(ResNet101, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 3, 3, 1, 1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU())
        
        self.resnet_modules = list(models.resnet101(pretrained=True).children())
        self.pre_module = nn.Sequential(*self.resnet_modules[:net_config["ensemble_idx"]])
        self.post_module = nn.Sequential(self.resnet_modules[net_config["ensemble_idx"]])
        self.gap = nn.AvgPool2d(net_config["feature_size"])
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, input):
        input = input.unsqueeze(2)
        batch_size = input.size(0)
        split_num = input.size(1)
        images = torch.chunk(input, split_num, 1)
        out = []
        for image in images:
            image = torch.squeeze(image, 1)
            image = self.conv1(image)
            image = self.pre_module(image)
            out.append(image)
        out = torch.stack(out, 1)
        [out, max_idx] = torch.max(out, 1)  
        out = self.post_module(out)
        out = self.gap(out).view(batch_size, -1)
        out = self.fc(out)
        return out