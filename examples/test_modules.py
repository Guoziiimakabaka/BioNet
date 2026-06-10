"""Smoke tests for BioNet core modules."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bionet import BrainInspiredBottleneck, OlfactoryBulbInspiredNet


def test_vfs_shape():
    x = torch.randn(1, 32, 40, 40)
    module = BrainInspiredBottleneck(32, 32)
    y = module(x)
    assert y.shape == x.shape


def test_obs_shape():
    x = torch.randn(1, 32, 40, 40)
    module = OlfactoryBulbInspiredNet(32, 32)
    y = module(x)
    assert y.shape == x.shape


if __name__ == "__main__":
    test_vfs_shape()
    test_obs_shape()
    print("BioNet smoke tests passed.")
