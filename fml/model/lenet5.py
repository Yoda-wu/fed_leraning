import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        # 卷积层
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 全连接层
        # self.fc = nn.Linear(4*4*16,120)
        in_features = 4 * 4 * 16
        if self.in_channels == 1:
            in_features = 4 * 4 * 16
        else:
            in_features = 5 * 5 * 16
        self.fc = nn.Linear(in_features, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        # print(x.shape)
        # x = torch.unsqueeze(x, 1)
        if self.in_channels == 1:
            x = x.view(-1, 1, 28, 28)
        # print(x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
