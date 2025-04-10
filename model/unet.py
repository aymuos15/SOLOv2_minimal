import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ConvModule


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, bilinear=True, base_c=64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        
        # UNet encoder (down path)
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        
        # UNet decoder (up path) - not used directly for feature extraction
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        
        # Store feature maps for FPN-like output
        self.features = []

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str) and pretrained:  # Only load if pretrained is a non-empty string
            self.load_state_dict(torch.load(pretrained, weights_only=True), strict=False)
        else:  # Handle both None and empty string cases
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)           # base_c channels, 1/1 scale
        x2 = self.down1(x1)        # base_c*2 channels, 1/2 scale
        x3 = self.down2(x2)        # base_c*4 channels, 1/4 scale
        x4 = self.down3(x3)        # base_c*8 channels, 1/8 scale
        x5 = self.down4(x4)        # base_c*16 channels, 1/16 scale
        
        # For SOLOv2, we need to return a tuple of feature maps at different scales
        features = (x2, x3, x4, x5)  # Features at 1/2, 1/4, 1/8, 1/16 scales
        
        return features


class UNetFPN(nn.Module):
    """
    UNet-based Feature Pyramid Network for SOLOv2
    
    This class takes UNet features and converts them to a format compatible with
    the SOLOv2 head, similar to FPN.
    """
    def __init__(self, 
                 in_channels,  # List of input channel dimensions from UNet features
                 out_channels=256,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 activation=None):
        super(UNetFPN, self).__init__()
        
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
            
        self.start_level = start_level
        self.end_level = end_level
        
        # Lateral convolutions - reduce channel dimensions
        self.lateral_convs = nn.ModuleList()
        # FPN convolutions - further process features
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=None,
                activation=self.activation,
                inplace=False)
            
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=None,
                activation=self.activation,
                inplace=False)
            
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        
        # Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path (feature fusion)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        
        # Build outputs
        # Part 1: From original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        
        # Part 2: Add extra levels if needed (using max pooling)
        if self.num_outs > len(outs):
            # Use max pool to get more levels on top of outputs
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        
        return tuple(outs)