"""
Baseline U-Net Architecture with ResNet-50 Encoder for Ilādṛṣṭi.
Supports 3-channel (RGB) and 4-channel (RGB + NIR) inputs with weight adaptation.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class ConvBlock(nn.Module):
    """Double 3x3 Convolution block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    Decoder stage with bilinear upsampling or transposed conv followed by
    skip-connection concatenation and double conv.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # Handle possible slight spatial discrepancies by padding if necessary
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class WildfireBaselineUNet(nn.Module):
    """
    Standard U-Net with ResNet-50 backbone for Wildfire Binary Segmentation.

    Attributes:
        in_channels (int): 3 or 4 input channels.
        num_classes (int): Number of segmentation classes (1 for binary).
        pretrained (bool): Whether to use ImageNet pretrained weights.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 1. Instantiate ResNet-50 Encoder
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # 2. Adapt first conv layer for 4 channels (RGB + NIR)
        if in_channels == 4:
            old_conv1 = resnet.conv1
            new_conv1 = nn.Conv2d(
                in_channels=4,
                out_channels=old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=old_conv1.bias is not None,
            )

            if pretrained:
                with torch.no_grad():
                    # Copy pretrained RGB weights
                    new_conv1.weight[:, :3, :, :] = old_conv1.weight
                    # Initialize 4th channel (NIR) as the mean of RGB channels
                    new_conv1.weight[:, 3:4, :, :] = old_conv1.weight.mean(
                        dim=1, keepdim=True
                    )
            resnet.conv1 = new_conv1
        elif in_channels != 3:
            raise ValueError(f"Supported in_channels are 3 or 4, got {in_channels}")

        # Encoder stages
        self.init_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )  # Output: 64 channels, 1/2 resolution (Skip 0)
        self.maxpool = resnet.maxpool  # 1/4 resolution
        self.layer1 = resnet.layer1  # 256 channels, 1/4 resolution (Skip 1)
        self.layer2 = resnet.layer2  # 512 channels, 1/8 resolution (Skip 2)
        self.layer3 = resnet.layer3  # 1024 channels, 1/16 resolution (Skip 3)
        self.layer4 = resnet.layer4  # 2048 channels, 1/32 resolution (Bottleneck)

        # 3. Decoder stages
        # Block 4: 1/32 -> 1/16
        self.dec4 = DecoderBlock(in_channels=2048, skip_channels=1024, out_channels=512)
        # Block 3: 1/16 -> 1/8
        self.dec3 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256)
        # Block 2: 1/8 -> 1/4
        self.dec2 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        # Block 1: 1/4 -> 1/2
        self.dec1 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        # Block 0: 1/2 -> 1/1
        self.dec0 = DecoderBlock(in_channels=64, skip_channels=0, out_channels=32)

        # 4. Final classification head
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input batch of shape (B, C, H, W).
            return_logits (bool): If True, returns unnormalized logits.
                                  If False, applies torch.sigmoid().

        Returns:
            torch.Tensor: Logit map or probability map of shape (B, 1, H, W).
        """
        # Encoder forward pass
        s0 = self.init_conv(x)  # (B, 64, H/2, W/2)
        p0 = self.maxpool(s0)  # (B, 64, H/4, W/4)
        s1 = self.layer1(p0)  # (B, 256, H/4, W/4)
        s2 = self.layer2(s1)  # (B, 512, H/8, W/8)
        s3 = self.layer3(s2)  # (B, 1024, H/16, W/16)
        b = self.layer4(s3)  # (B, 2048, H/32, W/32)

        # Decoder forward pass with skip connections
        d4 = self.dec4(b, s3)  # (B, 512, H/16, W/16)
        d3 = self.dec3(d4, s2)  # (B, 256, H/8, W/8)
        d2 = self.dec2(d3, s1)  # (B, 128, H/4, W/4)
        d1 = self.dec1(d2, s0)  # (B, 64, H/2, W/2)
        d0 = self.dec0(d1)  # (B, 32, H, W)

        logits = self.head(d0)  # (B, 1, H, W)

        if return_logits:
            return logits
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method returning sigmoid probabilities."""
        self.eval()
        return self.forward(x, return_logits=False)
