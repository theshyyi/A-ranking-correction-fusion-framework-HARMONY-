import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享感知层
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 这里的权重表示：哪个降水产品在当前更重要
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 这里的权重表示：哪个像素点需要被重点修正
        return self.sigmoid(self.conv1(x))

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(7)

    def forward(self, x):
        out = x * self.ca(x) # 通道加权
        out = out * self.sa(out) # 空间加权
        return out

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.cbam = CBAMBlock(dim)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out) # 注意力机制
        out += residual
        out = self.relu(out)
        return out

class AttentionFusionNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(AttentionFusionNet, self).__init__()
        # 浅层特征提取
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 深层残差网络
        self.layer1 = ResBlock(64)
        self.layer2 = ResBlock(64)
        self.layer3 = ResBlock(64)
        self.layer4 = ResBlock(64)
        
        # 输出层
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)
        # Softplus 保证降水非负，且在 0 附近平滑
        self.act_out = nn.Softplus()

    def forward(self, x):
        x = self.conv_in(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.conv_out(x)
        return self.act_out(out)

class TweedieLoss(nn.Module):
    def __init__(self, p=1.5):
        super(TweedieLoss, self).__init__()
        self.p = p

    def forward(self, y_pred, y_true):
        epsilon = 1e-8
        y_pred = y_pred + epsilon
        a = y_true * (y_pred ** (1 - self.p)) / (1 - self.p)
        b = (y_pred ** (2 - self.p)) / (2 - self.p)
        loss = -a + b
        return loss.mean()