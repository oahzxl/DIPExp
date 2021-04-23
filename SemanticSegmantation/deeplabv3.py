import torch
import torch.nn as nn
from torch.nn import functional as F
from backbone_8stride import resnet18


class ASPPModule(nn.Module):
    def __init__(self, dilations, in_channels, channels):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels

        self.aspp = nn.ModuleList()
        for dilation in dilations:
            self.aspp.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        self.channels,
                        1 if dilation == 1 else 3,
                        dilation=dilation,
                        padding=0 if dilation == 1 else dilation),
                    nn.BatchNorm2d(channels),
                    nn.ReLU()))

    def forward(self, x):
        aspp_outs = []
        for aspp_module in self.aspp:
            aspp_outs.append(aspp_module(x))
        return aspp_outs


class DeepLabUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                1 if dilation == 1 else 3,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)


class DeepLabv3(nn.Module):
    def __init__(self, dilations=(1, 6, 12, 18)):
        super(DeepLabv3, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.in_channels = 512
        self.channels = 256

        self.feats = resnet18(pretrained=True)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels, self.channels, 1))
        self.aspp_modules = ASPPModule(
            dilations, self.in_channels, self.channels)
        self.bottleneck = nn.Conv2d((len(dilations) + 1) * self.channels,
                                    self.channels, 3, padding=1)
        self.final = nn.Sequential(
            nn.Conv2d(self.channels, 6, kernel_size=1),
        )
        self.cuda()

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.cuda()
        x = self.feats(x)[0]
        aspp_outs = [F.interpolate(self.image_pool(x), x.size()[2:],
                                   mode='bilinear', align_corners=True)]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        output = F.interpolate(output, (h, w), mode='bilinear', align_corners=True)
        return self.final(output)
