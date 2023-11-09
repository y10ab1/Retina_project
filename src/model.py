import numpy as np
import torch
import torch.nn as nn
import torchvision



class MY_VGG16(torch.nn.Module):

  # init function
  def __init__(self, model, num_classes=1):
    super().__init__()

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