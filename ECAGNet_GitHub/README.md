# ECAGNet

Official, code-only implementation of ECAGNet for report-guided lesion segmentation in chest images. This repository contains the model, data loader, training entry point, and evaluation entry point used by the revised manuscript. It intentionally excludes patient data, pre-trained weights, checkpoints, prediction masks, logs, and manuscript files.

## Environment

Create a Python environment, install a CUDA-compatible PyTorch build appropriate for the target machine, then install the remaining packages:

```bash
pip install -r requirements.txt
```

The text encoder is loaded from `microsoft/BiomedVLP-CXR-BERT-specialized` through Hugging Face on first use. Internet access or a locally cached Hugging Face model is therefore required. The code supports optional ConvNeXt-Tiny pretraining: set `MODEL.vision_pretrained_path` in the YAML file to the downloaded checkpoint. Leave it empty only when training the visual encoder from scratch.

## Data

Download QaTa-COV19 and/or MosMedData+ from their official sources. Data are not redistributed here. Arrange the image, mask, and report CSV files as documented in [`data/README.md`](data/README.md), then adjust the relative paths in the appropriate file under `configs/` if needed.

## Train

```bash
python train.py --config configs/qata_cov19.yaml --seed 42
```

The QaTa-COV19 template reproduces the 80/20 split applied to the training CSV by the original experimental protocol. The MosMedData+ template uses explicitly supplied train, validation, and test CSV files.

## Evaluate

```bash
python evaluate.py \
  --config configs/qata_cov19.yaml \
  --checkpoint outputs/qata_cov19/checkpoints/<checkpoint>.ckpt
```

The evaluation script reports mean Dice and mean IoU over the configured test split.

## Reproducibility notes

* Use the supplied YAML templates as the starting point and record the seed, hardware, PyTorch/CUDA build, and any data-preparation changes.
* `gps_mode: blurred` is the ECAGNet setting used for Gaussian-prior supervision. `hard` replaces the soft target with a binary mask, and `none` removes this auxiliary supervision.
* Published or third-party baseline values should be evaluated under their own documented protocols; they are not bundled or rerun by this repository.

## License and acknowledgements

This release retains the GNU GPL v3 license of the upstream implementation on which the codebase was built. It uses MONAI, PyTorch Lightning, Hugging Face Transformers, BiomedVLP CXR-BERT, and ConvNeXt. Please cite their respective papers and repositories when using those components.
