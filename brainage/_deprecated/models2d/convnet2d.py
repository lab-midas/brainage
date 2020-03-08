import torch.nn as nn


class ConvNet2D(nn.Module):

  def __init__(self, in_channels=1):
    super(ConvNet2D, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer3 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.drop_out = nn.Dropout()
    self.fc1 = nn.Linear(38912, 1000)
    self.fc2 = nn.Linear(1000, 1)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = out.reshape(out.size(0), -1)
    out = self.drop_out(out)
    out = self.fc1(out)
    out = self.fc2(out)
    return out

