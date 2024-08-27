import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import datetime
import logging
import time
import torch.nn.functional as F

from KKLayer import KKLayer



class SlidingProjectionNet(nn.Module):
    def __init__(self, input_channels = 1, num_classes=10):
        super(SlidingProjectionNet, self).__init__()
        self.layer1 = nn.Sequential(
            #ProjConv2D(in_channels=start_channels, out_channels=16, kernel_size=3, padding=1),
            #nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x14x14
            KKLayer(in_channels=input_channels, out_channels=16), # 16x14x14
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16x7x7
        )
        self.layer2 = nn.Sequential(
            #ProjConv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), #32x7x7
            #KKLayer(in_channels=16, out_channels=32), #64x7x7
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) #32x3x3
        )
        # Die
        self.layer3 = nn.Sequential(
            #ProjConv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), #64x3x3
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2) #64x1x1
        )
        
        self.layer4 = nn.Sequential(
            #ProjConv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            #ProjConv2D(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.single_fc = nn.Linear(64, num_classes)


        # Not used in current flow
        """ self.fc1 = nn.Sequential(
            #nn.Linear(256 * 1 * 1, 512), # MNIST
            nn.Linear(64 * 3 * 3, 512), # CIFAR10
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(256, 10) """
    
    def forward(self, x):
        x = self.layer1(x)  
        layer1_output = x.clone() # Return this to visualize FM
        
        x = self.layer2(x)
        layer2_output = x.clone() # Return this to visualize FM
        
        
        x = self.layer3(x)
        layer3_output = x.clone() # Return this to visualize FM
        
        #x = self.layer4(x)
        #x = self.layer5(x)

        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # GAP
        x = self.single_fc(x)
        return x, [layer1_output, layer2_output, layer3_output]
    
        # using GAP instead of FCs
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x, [layer1_output, layer2_output, layer3_output]