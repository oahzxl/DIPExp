import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNHead(nn.Module):
    def __init__(self, backbone):
        super(FPNHead, self).__init__()

        self.backbone = backbone

        self.fc1 = nn.Conv1d(128 * 15 * 15, 4, 1)
        self.fc2 = nn.Conv1d(128 * 15 * 15, 5, 1)

        # self.fc1 = nn.Linear(64 * 16 * 16, 4)
        # self.fc2 = nn.Linear(64 * 16 * 16, 5)
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(64 * (2 ** i), 256, 1) for i in range(4)]
        )
        self.bn = nn.BatchNorm2d(256)
        self.conv = nn.Conv2d(256, 128, 3, 2)

    def forward(self, x):
        x = self.backbone(x)

        laterals = [
            lateral_conv(F.dropout(x[i], p=0.2))
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape)

        x = laterals[0]
        x = F.relu(self.bn(x))
        x = self.conv(x)
        x = F.dropout(x, p=0.2)

        x = x.view(x.shape[0], -1).unsqueeze(-1)
        boxes = self.fc1(x).squeeze(-1)
        classes = self.fc2(x).squeeze(-1)
        return boxes, classes
