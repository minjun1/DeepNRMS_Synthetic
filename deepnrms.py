import os
import torch
import numpy as np
import time
import copy

import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch import nn


class Deep_NRMS(nn.Module):
    def __init__(self):
        super(Deep_NRMS, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.dense1 = nn.Linear(16*16*32, 2048)
        self.dense2 = nn.Linear(2048, 512)
        self.dense3 = nn.Linear(512, 128)
        
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
   
    def forward(self, img):
        
        x = self.conv1(img)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.elu(x)
        
        x = x.view(-1, 16*16*32)
        
        x = self.dense1(x)
        x = F.dropout(x, training=self.training)
        x = F.elu(x)
        
        x = self.dense2(x)
        x = F.dropout(x, training=self.training)
        x = F.elu(x)
        
        x = self.dense3(x)
        x = F.elu(x)      
        return x

