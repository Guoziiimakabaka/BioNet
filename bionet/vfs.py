"""Visual Foreground Saliency modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterSurroundLayer(nn.Module):
    """
    Depthwise center-surround filtering for visual foreground enhancement.

    The layer initializes separate center and surround Gaussian filters and
    returns the sum of ON-center and OFF-center responses.
    """

    def __init__(self, channels, kernel_size_center=3, sigma_center=1.0, kernel_size_surround=7, sigma_surround=2.0):
        super().__init__()
        self.channels = channels
        self.center_conv = nn.Conv2d(
            channels, channels, kernel_size_center, padding=kernel_size_center // 2, bias=False, groups=channels
        )
        self.surround_conv = nn.Conv2d(
            channels, channels, kernel_size_surround, padding=kernel_size_surround // 2, bias=False, groups=channels
        )
        self._initialize_weights(kernel_size_center, sigma_center, kernel_size_surround, sigma_surround)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create a normalized 2D Gaussian kernel."""
        k = (kernel_size - 1) / 2.0
        x_cord = torch.arange(-k, k + 1)
        x_grid, y_grid = torch.meshgrid(x_cord, x_cord, indexing="ij")
        exponent = -(x_grid**2 + y_grid**2) / (2 * sigma**2)
        gaussian_kernel = torch.exp(exponent)
        return gaussian_kernel / gaussian_kernel.sum()

    def _initialize_weights(self, kernel_size_center, sigma_center, kernel_size_surround, sigma_surround):
        """Initialize the center and surround depthwise kernels."""
        center_kernel = self._create_gaussian_kernel(kernel_size_center, sigma_center)
        surround_kernel = self._create_gaussian_kernel(kernel_size_surround, sigma_surround)
        center_weight = center_kernel.expand(self.channels, 1, -1, -1)
        surround_weight = surround_kernel.expand(self.channels, 1, -1, -1)
        self.center_conv.weight.data.copy_(center_weight)
        self.surround_conv.weight.data.copy_(surround_weight)

    def forward(self, x):
        center_signal = self.center_conv(x)
        surround_signal = self.surround_conv(x)
        on_center_response = F.relu(center_signal - surround_signal)
        off_center_response = F.relu(surround_signal - center_signal)
        return on_center_response + off_center_response


class ChannelAttention(nn.Module):
    """Lightweight squeeze-excitation style channel reweighting."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_x = self.global_avg_pool(x)
        channel_weights = self.sigmoid(self.fc2(self.relu(self.fc1(pooled_x))))
        return x * channel_weights


class BrainInspiredBottleneck(nn.Module):
    """
    Visual foreground-saliency bottleneck.

    The active path applies center-surround filtering followed by channel
    reweighting, then adds the enhanced response back to the input.
    """

    def __init__(self, c1, c2, shortcut=True, reduction_ratio=16):
        super().__init__()
        assert c1 == c2, "BrainInspiredBottleneck requires input and output channels to be the same."
        self.channels = c1
        self.shortcut = shortcut
        self.retina_layer = CenterSurroundLayer(self.channels)
        self.channel_attention = ChannelAttention(in_channels=self.channels, reduction_ratio=reduction_ratio)

    def forward(self, x, x_top_down_prediction=None):
        """Forward pass with an unused top-down argument kept for API stability."""
        x_processed = self.retina_layer(x)
        x_processed = self.channel_attention(x_processed)
        return 2 * x_processed + x
