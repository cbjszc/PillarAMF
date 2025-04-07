import torch
import torch.nn as nn
# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, dilation):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, dilation=dilation, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.PReLU(num_parameters=in_ch)
        self.relu2 = nn.PReLU(num_parameters=out_ch)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

# 多尺度特征提取器
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels)
        )
        self.conv3x3_d1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv5x5_d2 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=5, padding=4, dilation=2)
        self.conv7x7_d3 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=7, padding=9, dilation=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_d1(x)
        x3 = self.conv5x5_d2(x)
        x4 = self.conv7x7_d3(x)
        x_concat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.fuse(x_concat)

# 金字塔融合颈部
class PyramidFusionNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.MultiScaleConv = MultiScaleFeatureExtractor(in_channels, out_channels)

        self.Upsample = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        self.upsamplex2 = nn.ModuleList()
        for i in range(2):
            Upsample = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2 ** (i + 1),
                                   stride=2 ** (i + 1), padding=0),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(num_parameters=out_channels)
            )
            conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(num_parameters=out_channels)
            )
            upsamplex2 = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(num_parameters=out_channels)
            )
            self.Upsample.append(Upsample)
            self.conv1x1.append(conv1x1)
            self.upsamplex2.append(upsamplex2)
        self.conv1x1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels)
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, multi_scale_x):
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)

        temp2 = self.MultiScaleConv(multi_scale_x[2])
        x2 = self.Upsample[1](temp2)
        temp1 = alpha * self.conv1x1[1](multi_scale_x[1]) + (1 - alpha) * self.upsamplex2[1](temp2)
        x1 = self.Upsample[0](temp1)
        temp0 = beta * self.conv1x1[0](multi_scale_x[0]) + (1 - beta) * self.upsamplex2[0](temp1)
        x0 = self.conv1x1_(temp0)

        x = torch.cat([x0, x1, x2], dim=1)
        return x
