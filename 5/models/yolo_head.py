import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, backbone):
        super(Head, self).__init__()

        self.backbone = backbone

        self.fc1 = nn.Linear(512 * 8 * 8, 4)
        self.fc2 = nn.Linear(512 * 8 * 8, 5)

        # self.fc1 = nn.Linear(64 * 16 * 16, 4)
        # self.fc2 = nn.Linear(64 * 16 * 16, 5)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.view(x.shape[0], -1)
        boxes = self.fc1(x)
        classes = self.fc2(x)
        return boxes, classes
