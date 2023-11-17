import numpy as np
import torch
import torch.nn as nn
import torchvision


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# model class
class VGG16_CBAM(torch.nn.Module):

  # init function
  def __init__(self, model, num_classes=1):
    super().__init__()
    
    self.ca = ChannelAttention(64)
    self.sa = SpatialAttention()
    # features
    self.features_1 = torch.nn.Sequential(*list(model.features.children())[:3])
    self.features_2 = torch.nn.Sequential(*list(model.features.children())[3:6])
    self.features_3 = torch.nn.Sequential(*list(model.features.children())[6:10])
    self.features_4 = torch.nn.Sequential(*list(model.features.children())[10:13])
    self.features_5 = torch.nn.Sequential(*list(model.features.children())[13:17])
    self.features_6 = torch.nn.Sequential(*list(model.features.children())[17:20])
    self.features_7 = torch.nn.Sequential(*list(model.features.children())[20:23])
    self.features_8 = torch.nn.Sequential(*list(model.features.children())[23:27])
    self.features_9 = torch.nn.Sequential(*list(model.features.children())[27:30])
    self.features_10 = torch.nn.Sequential(*list(model.features.children())[30:33])
    self.features_11 = torch.nn.Sequential(*list(model.features.children())[33:37])
    self.features_12 = torch.nn.Sequential(*list(model.features.children())[37:40])
    self.features_13 = torch.nn.Sequential(*list(model.features.children())[40:43])

    self.avgpool = nn.AdaptiveAvgPool2d(7)

    # classifier
    self.classifier = torch.nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(25088,1),
        nn.Sigmoid()
    )


  # forward
  def forward(self, x):
    
    x = self.features_1(x)
    x = self.ca(x) * x
    x = self.sa(x) * x    

    x = self.features_2(x)
    x = self.features_3(x)
    x = self.features_4(x)
    x = self.features_5(x)
    x = self.features_6(x)
    x = self.features_7(x)
    x = self.features_8(x)
    x = self.features_9(x)
    x = self.features_10(x)

    x = self.avgpool(x)
    x = x.view(x.shape[0], -1)
    x = self.classifier(x)
    return x
    
    
    
    
    



