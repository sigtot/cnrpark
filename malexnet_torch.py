import torch
import torch.nn as nn


class mAlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(mAlexNet, self).__init__()

        self.convol = nn.Sequential(

            # Layer 1
            nn.Conv2d(3, 16, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 2
            nn.Conv2d(16, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 3
            nn.Conv2d(20, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc = nn.Sequential(

            # Layer 4
            nn.Linear(30, 48),
            nn.ReLU(inplace=True),

            # Layer 5
            nn.Linear(48, num_classes),  # chk number
        )

    def forward(self, x):
        x = self.convol(x)
        x = x.view(x.size(0), 30)
        x = self.fc(x)
        return x


def mAlexnet(**kwargs):
    model = mAlexNet(**kwargs)
    return model
