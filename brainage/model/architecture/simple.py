import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):

    def __init__(self, input_channels, output_channels, use_position=False):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.use_position = use_position

        self.conv1 = nn.Conv3d(self.input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(32 + self.input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

         # include patch position ?
        self.use_position = use_position
        add_pos_features = 3 if self.use_position else 0
        self.fc = nn.Linear(97 + add_pos_features, output_channels)

        # grad cam
        self.gradients = None

    def forward(self, x, pos=None, hook=False):
        residual = x
        y1 = self.conv1(x)
        y1 = self.conv2(y1)
        y1 = torch.cat([y1,residual], dim=1)
        y1 = self.maxpool(y1)
        residual = y1
        y2 = self.conv3(y1)
        y2 = self.conv4(y2)
        y2 = torch.cat([y2,residual], dim=1)

        # register the hook
        if hook:
            y2.register_hook(self.activations_hook)

        y3 = self.avgpool(y2)
        y3 = y3.view(y3.size(0), -1)
        if self.use_position: 
            y3 = torch.cat([y3, pos], dim=1)
        y4 = self.fc(y3)
        return y4

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        residual = x
        y1 = self.conv1(x)
        y1 = self.conv2(y1)
        y1 = torch.cat([y1,residual], dim=1)
        y1 = self.maxpool(y1)
        residual = y1
        y2 = self.conv3(y1)
        y2 = self.conv4(y2)
        y2 = torch.cat([y2,residual], dim=1)
        return y2
