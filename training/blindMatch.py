from torchvision.models import resnet18
import os
import torch
import numpy as np
# from imgaug import augmenters as iaa
import torchvision
import random
import PIL.Image as Image
import cv2
import math

print("Congratulations on [BlindAuth] file import!")

class FingerNet(torch.nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # print(num_classes)
        self.basenet = resnet18(num_classes=num_classes) # feature 차원 갯수 -> 이후 작업 

    def forward(self, x):
        x = self.basenet(x)
        x = torch.nn.functional.normalize(x)
        return x
    
class FingerSTNNet(FingerNet):
    def __init__(self, num_classes=1):
        super().__init__(num_classes=num_classes)
        # Spatial transformer localization-network
        self.localization = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=7),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, 10, kernel_size=5),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = torch.nn.Sequential(
            torch.nn.Linear(10 * 52 * 52, 32),
            torch.nn.ReLU(True),
            torch.nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = torch.nn.functional.affine_grid(theta, x.size())
        x = torch.nn.functional.grid_sample(x, grid)

        return x

    def forward(self, x):
        return super().forward(self.stn(x))
    
class FingerCentroids(torch.nn.Module): # 무게중심 
    def __init__(self, n_ids, n_dim=16):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (n_ids, n_dim)))

    def forward(self, x):
        out = torch.nn.functional.linear(
            torch.nn.functional.normalize(x), 
            torch.nn.functional.normalize(self.weight))

        return out
    
class ArcFace(torch.nn.Module):
    def __init__(self, sp=64.0, sn=64.0, m=0.5, **kwargs):
        super(ArcFace, self).__init__()
        self.sp = sp / sn  # sn will be multiplied again
        self.sn = sn
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, cosine: torch.Tensor, label):
        cosine = cosine.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
       
        cos_theta = cosine[index]
        target_logit = cos_theta[torch.arange(0, cos_theta.size(0)), label[index]].view(
            -1, 1
        )
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = (target_logit * self.cos_m - sin_theta * self.sin_m).to(
            cosine.dtype
        )  # cos(target+margin)

        cosine[index] = cosine[index].scatter(
            1, label[index, None], cos_theta_m * self.sp
        )
        cosine.mul_(self.sn)

        return cosine
