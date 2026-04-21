# CONA-Net

PyTorch implementation of **CONA-Net** for 3D cerebrovascular segmentation from **Time-of-Flight Magnetic Resonance Angiography (TOF-MRA)**.

> Paper: **Accurate Delineation of Cerebrovascular Structures from TOF-MRA with Connectivity-Reinforced Deep Learning**  
> Authors: **Shoujun Yu, Cheng Li, Yousuf Babiker Mohammed Osman, Shanshan Wang, Hairong Zheng**

---

## Overview

Segmenting cerebrovascular structures from TOF-MRA is difficult because vessels are:

- thin and tortuous,
- highly imbalanced relative to the background,
- prone to discontinuity or fragmentation in prediction maps,
- sensitive to topology damage during training and inference.

**CONA-Net** is proposed to improve vessel delineation while better preserving vascular continuity and connectivity.  
This repository contains the implementation of CONA-Net, several ablation variants, baseline models, training scripts, testing scripts, and a sample configuration file for the **IXI** dataset.

---

## Highlights

- **3D vessel segmentation** for TOF-MRA volumes
- **Connectivity-aware learning design** for preserving vessel continuity
- Includes **CONA-Net** and multiple **ablation models**
- Includes several **baseline networks** for comparison
- Supports **k-fold training**
- Reports common vessel-segmentation metrics:
  - Dice (DSC)
  - clDice
  - HD95
  - ASD
  - Sensitivity
  - Specificity

---

## Repository Structure

```text
CONA-Net/
├── configs/
│   └── IXI_CONANet_config.yml
├── models/
│   ├── CONANet.py
│   ├── CONANet_Base.py
│   ├── CONANet_WOCL.py
│   ├── CONANet_WOEDGE.py
│   ├── CONANet_WOFC.py
│   ├── CONANet_WOLOSS.py
│   ├── CONANet_WOCONAM.py
│   ├── CONANet_WOCONAM_FC.py
│   ├── CS2Net.py
│   ├── ERNet.py
│   ├── RENet.py
│   ├── UNet2Plus.py
│   ├── UNet3D.py
│   ├── Uception.py
│   ├── VNet.py
│   ├── __init__.py
│   └── init_weights.py
├── dataloader.py
├── losses.py
├── metrics.py
├── optimizers.py
├── sknw.py
├── train.py
├── test.py
├── utils.py
└── README.md
```

---

## Requirements

This repository does not currently provide an official `requirements.txt` or environment file.  
Based on the source code imports, you will likely need:

```bash
python >= 3.9
pytorch
torchvision
numpy
PyYAML
SimpleITK
nibabel
torchinfo
alive-progress
```

A typical setup may look like this:

```bash
conda create -n conanet python=3.10 -y
conda activate conanet

pip install torch torchvision
pip install numpy pyyaml SimpleITK nibabel torchinfo alive-progress
```

> Please adjust the PyTorch install command based on your CUDA version.

---

## Dataset Preparation

The current dataloader expects the dataset to be organized as follows:

```text
DATA_ROOT/
├── train/
│   ├── raw/
│   │   ├── case001.nii.gz
│   │   ├── case002.nii.gz
│   │   └── ...
│   └── gt/
│       ├── case001_GT.nii.gz
│       ├── case002_GT.nii.gz
│       └── ...
└── test/
    ├── raw/
    │   ├── case101.nii.gz
    │   ├── case102.nii.gz
    │   └── ...
    └── gt/
        ├── case101_GT.nii.gz
        ├── case102_GT.nii.gz
        └── ...
```

### Naming convention

For each input volume:

- raw image: `caseXXX.nii.gz`
- ground truth: `caseXXX_GT.nii.gz`

The test loader follows the same naming rule.

---

## Configuration

A sample config file is provided at:

```bash
configs/IXI_CONANet_config.yml
```

Before training or testing, update at least the following fields:

```yaml
checkpoint_dir: './checkpoint'
pred_dir: './predictions'

model:
  name: CONANet
  input_channels: 1
  output_channels: 1

train:
  num_fold: 5
  batch_size: 2
  epochs: NUM_EPOCHS
  validate_after_epochs: X
  max_num_iterations: MAX_ITERATIONS
  data_loader:
    num_workers: 32
    dataset_name: IXI
    data_path: "ABSOLUTE/PATH/TO/YOUR/TRAINING/DATA"
    patch_size: [W, H, Slice]
    patch_center: [W, H, Slice]

loss:
  name: AdaptiveRegionalEdgeDiceCLDiceLoss
  threshold: 0.8
  partition_size: 16

optimizer:
  name: Adam
  learning_rate: 0.001
  weight_decay: 0.0005

eval_metric:
  name: BinaryMetrics
  voxel_spacing: [Z, X, Y]

test:
  num_workers: 32
  data_path: "ABSOLUTE/PATH/TO/YOUR/TEST/DATA"
```

---

## Training

The current training script loads the config directly from:

```python
configs/IXI_CONANet_config.yml
```

You can start training with:

```bash
python train.py
```

### Notes

- Training currently uses **k-fold cross-validation**.
- The script is designed for **GPU execution** and explicitly does **not** support CPU-only use.
- Checkpoints are saved under `checkpoint_dir`.

---

## Testing / Inference

The test script requires you to manually edit two lines in `test.py`:

```python
config = load_config_ide('PATH/TO/CONFIGURATION .yml FILES')
stamp = 'PATH/TO/.pth FILES'
```

After editing them, run:

```bash
python test.py
```

### Output

The script computes and reports:

- DSC
- clDice
- HD95
- ASD
- SEN
- SPEC

There is also code for saving prediction volumes, but it is currently commented out in `test.py`.  
You can uncomment the relevant `sitk.WriteImage(...)` lines if you want to export raw, prediction, and ground-truth NIfTI files.

---

## Available Models

### Proposed model

- `CONANet`

### Ablation models

- `CONANet_Base`
- `CONANet_WOCL`
- `CONANet_WOEDGE`
- `CONANet_WOFC`
- `CONANet_WOLOSS`
- `CONANet_WOCONAM`
- `CONANet_WOCONAM_FC`

### Baseline / comparison models

- `UNet3D`
- `UNet2Plus`
- `VNet`
- `Uception`
- `ERNet`
- `RENet`
- `CS2Net`

To switch models, change the `model.name` field in the YAML configuration.

---

## Evaluation Metrics

This repository evaluates vessel segmentation using several metrics that are common in topology-sensitive segmentation tasks:

- **Dice (DSC)**: overlap between prediction and ground truth
- **clDice**: topology-aware overlap metric for tubular structures
- **HD95**: 95th percentile Hausdorff distance
- **ASD**: average surface distance
- **Sensitivity (SEN)**: true positive rate
- **Specificity (SPEC)**: true negative rate

---

## Practical Tips

- Make sure your `patch_size` and `patch_center` match the actual spatial size of your preprocessed TOF-MRA volumes.
- Start with a small subset of cases to verify:
  - data loading,
  - file naming,
  - crop settings,
  - checkpoint writing,
  - metric computation.
- If you want a cleaner workflow, consider refactoring:
  - `train.py` to accept a command-line `--config`,
  - `test.py` to accept both `--config` and `--checkpoint`,
  - a dedicated `requirements.txt`.

---

## Citation

If you use this repository in your research, please cite the original CONA-Net paper:

**Yu S, Li C, Osman YBM, Wang S, Zheng H.** Accurate Delineation of Cerebrovascular Structures from TOF-MRA with Connectivity-Reinforced Deep Learning. In: *Machine Learning in Medical Imaging (MLMI 2024)*, pp. 280-289. Springer, 2024. **DOI:** 10.1007/978-3-031-73284-3_28

### BibTeX

```bibtex
@incollection{yu2024conanet,
  title     = {Accurate Delineation of Cerebrovascular Structures from TOF-MRA with Connectivity-Reinforced Deep Learning},
  author    = {Yu, Shoujun and Li, Cheng and Osman, Yousuf Babiker Mohammed and Wang, Shanshan and Zheng, Hairong},
  booktitle = {Machine Learning in Medical Imaging},
  pages     = {280--289},
  year      = {2024},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-73284-3_28}
}
```
