import torch.nn as nn


class ConvNet3D(nn.Module):
    def __init__(self):
        super(ConvNet3D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16000, 1024),#19200
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU())
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # flatten
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.linear(out)
        return out