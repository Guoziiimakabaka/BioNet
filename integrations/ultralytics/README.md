# Ultralytics Integration

This directory contains model YAML examples that show where BioNet modules were inserted in YOLO-style detectors.

The full Ultralytics framework is not vendored in this repository. To reproduce YOLO experiments, install a compatible Ultralytics release or use the project-specific fork used for the paper, then register:

- `BrainInspiredBottleneck` or a YOLO wrapper such as `MyC3K2` in the backbone.
- `OlfactoryBulbInspiredNet` before the detection head on P3/P4/P5 features.

For YAML parsing, the detector framework must expose these classes to the model parser namespace. In an Ultralytics-style fork this usually means importing them in `ultralytics/nn/modules/__init__.py` and adding them to the module set handled by `parse_model`.

The core implementations are available in:

- `bionet.vfs`
- `bionet.obs`

The OBS assignment used by the paper is:

- Tufted pathway: 3x3 local-inhibition kernel.
- Mitral pathway: 5x5 local-inhibition kernel.
