# BioNet

BioNet provides the core implementation of two bio-inspired modules for object detection under visual interference:

- Visual Foreground Saliency (VFS): learnable center-surround feature enhancement.
- Olfactory Bulb-inspired Suppression (OBS): subtractive local and global inhibition.

The repository is intentionally lightweight. It contains the method modules, smoke-test examples, and YOLO integration YAMLs, but it does not vendor the full Ultralytics codebase.

## Repository Structure

```text
bionet/
├── bionet/
│   ├── vfs.py                      # VFS core modules
│   ├── obs.py                      # OBS core modules
│   └── __init__.py                 # Public exports
├── examples/
│   ├── minimal_forward.py          # Minimal forward-pass example
│   └── test_modules.py             # Shape smoke tests
├── integrations/
│   └── ultralytics/
│       ├── yolo11_bionet.yaml
│       ├── yolov8_bionet.yaml
│       ├── yolov3_bionet.yaml
│       ├── yolo12_bionet.yaml
│       └── README.md
├── CITATION.cff
├── requirements.txt
└── README.md
```

## Core Modules

### Visual Foreground Saliency

VFS is implemented by:

- `CenterSurroundLayer`
- `ChannelAttention`
- `BrainInspiredBottleneck`

It initializes depthwise center and surround Gaussian filters, forms ON-center and OFF-center responses, applies channel reweighting, and adds the enhanced response back to the input feature.

### Olfactory Bulb-inspired Suppression

OBS is implemented by:

- `GlomerulusLayer`
- `PeriglomerularCellLayer`
- `GranuleCellLayer`
- `MitralTuftedCellLayer`
- `OlfactoryBulbInspiredNet`

The paper configuration uses:

- Tufted pathway: 3x3 local-inhibition kernel.
- Mitral pathway: 5x5 local-inhibition kernel.

## Installation

```bash
conda create -n bionet python=3.10 -y
conda activate bionet
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA environment.

## Minimal Usage

```python
import torch

from bionet import BrainInspiredBottleneck, OlfactoryBulbInspiredNet

x = torch.randn(2, 64, 80, 80)

vfs = BrainInspiredBottleneck(64, 64)
obs = OlfactoryBulbInspiredNet(64, 64)

y_vfs = vfs(x)
y_obs = obs(x)
```

Run the included smoke tests:

```bash
python examples/test_modules.py
```

## YOLO Integration

YOLO integration files are provided in `integrations/ultralytics/`. They show the insertion points used for YOLO-style detectors, but they are not a standalone training framework.

To reproduce YOLO experiments, install or prepare a compatible Ultralytics codebase and register the BioNet modules in the model parser. See `integrations/ultralytics/README.md`.

## Reproducibility Notes

- Report the exact detector YAML, dataset YAML, seed, image size, batch size, optimizer settings, and checkpoint path for each experiment.
- The integration YAMLs document module placement, not dataset paths or trained weights.
- For manuscript reproduction, use the same Ultralytics version or fork used during the experiments.

## Citation

If you use this repository, please cite the associated manuscript after publication.

```bibtex
@article{bionet2026,
  title   = {BioNet: Bio-inspired object detection under visual interference},
  author  = {Xu, Junxiu and Guo, Zhen and Xia, Hualin and Xu, Kun and Bai, Xiuxiu},
  journal = {TBD},
  year    = {2026}
}
```

## License

The core BioNet modules are provided for research use. The integration YAMLs refer to Ultralytics-style model definitions; please follow the license terms of any external framework used for training or deployment.
