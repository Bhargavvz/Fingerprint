"""
Attention mechanisms for the hybrid deep learning model.
Implements CBAM (Convolutional Block Attention Module) and SE (Squeeze-and-Excitation) blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.
    
    Uses both max-pooling and average-pooling to capture channel-wise statistics,
    then applies a shared MLP to generate attention weights.
    
    Reference: https://arxiv.org/abs/1807.06521
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Args:
            in_channels: Number of input channels
            reduction: Reduction ratio for the MLP bottleneck
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Channel attention weights of shape (B, C, 1, 1)
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        attention = torch.sigmoid(avg_out + max_out)
        
        return attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.
    
    Uses channel-wise max and average pooling, concatenates them,
    and applies a convolution to generate spatial attention weights.
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: Kernel size for the convolution layer
        """
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Spatial attention weights of shape (B, 1, H, W)
        """
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(concat))
        
        return attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines channel and spatial attention mechanisms to refine feature maps.
    The attention is applied sequentially: first channel, then spatial.
    
    Reference: https://arxiv.org/abs/1807.06521
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 7
    ):
        """
        Args:
            in_channels: Number of input channels
            reduction: Reduction ratio for channel attention
            spatial_kernel_size: Kernel size for spatial attention convolution
        """
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-refined tensor of same shape
        """
        # Apply channel attention
        x = x * self.channel_attention(x)
        
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    
    An alternative to CBAM that only uses channel attention.
    Simpler but still effective for channel recalibration.
    
    Reference: https://arxiv.org/abs/1709.01507
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Args:
            in_channels: Number of input channels
            reduction: Reduction ratio for the bottleneck
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Recalibrated tensor of same shape
        """
        b, c, _, _ = x.size()
        
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale
        return x * y.expand_as(x)


class ECABlock(nn.Module):
    """
    Efficient Channel Attention Block.
    
    A lightweight alternative to SE with adaptive kernel size.
    Uses 1D convolution instead of fully connected layers.
    
    Reference: https://arxiv.org/abs/1910.03151
    """
    
    def __init__(self, in_channels: int, gamma: int = 2, b: int = 1):
        """
        Args:
            in_channels: Number of input channels
            gamma: Parameter for kernel size calculation
            b: Parameter for kernel size calculation
        """
        super().__init__()
        
        # Adaptive kernel size
        import math
        kernel_size = int(abs((math.log2(in_channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Recalibrated tensor of same shape
        """
        # Squeeze
        y = self.avg_pool(x)  # (B, C, 1, 1)
        
        # 1D conv for channel attention
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        
        # Scale
        y = torch.sigmoid(y)
        
        return x * y


if __name__ == "__main__":
    # Test attention modules
    batch_size = 4
    channels = 256
    height, width = 14, 14
    
    x = torch.randn(batch_size, channels, height, width)
    
    # Test CBAM
    cbam = CBAM(channels, reduction=16)
    out_cbam = cbam(x)
    print(f"CBAM: {x.shape} -> {out_cbam.shape}")
    
    # Test SE Block
    se = SEBlock(channels, reduction=16)
    out_se = se(x)
    print(f"SE Block: {x.shape} -> {out_se.shape}")
    
    # Test ECA Block
    eca = ECABlock(channels)
    out_eca = eca(x)
    print(f"ECA Block: {x.shape} -> {out_eca.shape}")
    
    # Parameter counts
    print(f"\nParameter counts:")
    print(f"  CBAM: {sum(p.numel() for p in cbam.parameters()):,}")
    print(f"  SE: {sum(p.numel() for p in se.parameters()):,}")
    print(f"  ECA: {sum(p.numel() for p in eca.parameters()):,}")
