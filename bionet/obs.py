"""Olfactory Bulb-inspired Suppression modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class GlomerulusLayer(nn.Module):
    """Initial visual feature projection analogous to glomerular aggregation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class PeriglomerularCellLayer(nn.Module):
    """
    Local subtractive inhibition inspired by periglomerular cells.

    The convolution estimates a learnable local inhibitory field from the
    current feature map and subtracts its bounded magnitude from the input.
    """

    def __init__(self, channels, inhibition_kernel_size=3):
        super().__init__()
        self.local_inhibitor = nn.Conv2d(
            channels, channels, kernel_size=inhibition_kernel_size, padding=inhibition_kernel_size // 2, bias=False
        )
        nn.init.kaiming_normal_(self.local_inhibitor.weight, mode="fan_out")

        self.inhibition_strength_coeff = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x_mitral_tufted):
        local_inhibitory_signal = self.local_inhibitor(x_mitral_tufted)
        inhibitory_signal = torch.tanh(local_inhibitory_signal)
        return x_mitral_tufted - (self.inhibition_strength_coeff * inhibitory_signal).abs()


class GranuleCellLayer(nn.Module):
    """
    Global feedback inhibition inspired by granule cells.

    A pooled global descriptor is projected into a channel-wise inhibitory
    signal, encouraging sparse and less redundant feature responses.
    """

    def __init__(self, channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gc_integrator = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.SiLU(),
        )

        self.gc_inhibitory_projection = nn.Linear(channels // 2, channels)
        self.inhibition_strength_coeff = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x_mitral_tufted):
        original_mc_tc_input = x_mitral_tufted
        b, c, h, w = x_mitral_tufted.shape

        gc_input = self.global_pool(original_mc_tc_input).view(b, c)
        gc_activated = self.gc_integrator(gc_input)
        inhibitory_signal_val = self.gc_inhibitory_projection(gc_activated).view(b, c, 1, 1)

        output = original_mc_tc_input - self.inhibition_strength_coeff * inhibitory_signal_val

        return nn.functional.silu(output)


class MitralTuftedCellLayer(nn.Module):
    """Integrates glomerular input with local and global inhibitory pathways."""

    def __init__(self, channels, pgc_inhibition_kernel_size=3):
        super().__init__()
        self.pgc_inhibition = PeriglomerularCellLayer(channels, inhibition_kernel_size=pgc_inhibition_kernel_size)
        self.gc_inhibition = GranuleCellLayer(channels)

    def forward(self, x_glomeruli):
        x_mc_tc_initial = x_glomeruli

        x_post_pgc = self.pgc_inhibition(x_mc_tc_initial)

        x_final_mc_tc_output = self.gc_inhibition(x_post_pgc)

        return x_glomeruli + x_final_mc_tc_output


class OlfactoryBulbInspiredNet(nn.Module):
    """
    Olfactory-bulb-inspired module for visual feature denoising.

    The tufted pathway uses a 3x3 local-inhibition kernel, and the mitral
    pathway uses a 5x5 local-inhibition kernel.
    """

    def __init__(self, in_channels, num_glomeruli_channels, pgc_inhibition_kernel_size=3):
        super().__init__()

        self.glomerulus_layer = GlomerulusLayer(in_channels, num_glomeruli_channels)

        self.tufted_complex = MitralTuftedCellLayer(num_glomeruli_channels // 2, pgc_inhibition_kernel_size=3)
        self.mitral_complex = MitralTuftedCellLayer(num_glomeruli_channels // 2, pgc_inhibition_kernel_size=5)

        self.output_layer = nn.Conv2d(num_glomeruli_channels, num_glomeruli_channels, kernel_size=1)
        self.final_activation = nn.SiLU()

    def forward(self, x):
        x_glomeruli = self.glomerulus_layer(x)
        channels = x.shape[1]
        half_channels = channels // 2
        x1 = x_glomeruli[:, :half_channels, :, :]
        x2 = x_glomeruli[:, half_channels:, :, :]

        x_processed_features1 = self.tufted_complex(x1)
        x_processed_features2 = self.mitral_complex(x2)

        output = self.final_activation(self.output_layer(torch.cat((x_processed_features1, x_processed_features2), dim=1)))

        if x.shape == output.shape:
            return output + x
        else:
            return output
