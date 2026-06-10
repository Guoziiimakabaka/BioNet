"""Minimal forward-pass example for BioNet core modules."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bionet import BrainInspiredBottleneck, OlfactoryBulbInspiredNet


def main():
    x = torch.randn(2, 64, 80, 80)

    vfs = BrainInspiredBottleneck(64, 64)
    obs = OlfactoryBulbInspiredNet(64, 64)

    y_vfs = vfs(x)
    y_obs = obs(x)

    print("VFS output:", tuple(y_vfs.shape))
    print("OBS output:", tuple(y_obs.shape))


if __name__ == "__main__":
    main()
