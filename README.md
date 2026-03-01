# CS 7640 - Project Lego 2: MS Lesion Segmentation Failure Modes

**Author:** Xuyen Nguyen (u1252970)

## Project Overview

This project investigates the failure modes of deep learning models in medical image segmentation, specifically focusing on Multiple Sclerosis (MS) white matter lesions. It compares a **3D UNet** (representing a strong local structural assumption) against a **Swin UNETR** (representing a relaxed, global context assumption) to understand how architectural biases influence performance under distribution shifts.

## Experimental Setup

### Data

- **Training:** "Best" dataset (Simulating a limited, high-quality clinical dataset).
- **Evaluation/Shift:** "Ljubljana" dataset (Simulating a distribution shift).

### Metrics

- **Lesion-Scale F1 Score:** Detection capability regardless of size.
- **Predictive Entropy:** Model confidence.
- **Retention Area Under the Curve (R-AUC):** Robustness and uncertainty quality.
- **Normalized Dice (nDSC):** Voxel-level segmentation quality.

### Audits

1.  **Spatial Overlap Audit:** Calculates intersection of False Positives between models to see if they fail in the same locations.
2.  **Uncertainty Calibration Audit:** Compares predictive entropy at failure sites.

---

## How to Run

### 0. Run with `run.py`

```bash
python run.py --epochs 300 --seeds 1 2 3
# Optional flags: --skip_install --skip_train --skip_eval --skip_audit
```

### 1. Requirements

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. Data Setup

Ensure your data is organized as follows (see `DATA_README.txt` for details):

```text
data/
├── train/      (Best dataset)
│   ├── flair/
│   ├──  gt/
│   └──...
├── dev_in/     (Validation)
│   ├── flair/
│   ├──  gt/
│   └──...
└── dev_out/    (Ljubljana - Evaluation/Shift)
    ├── flair/
    ├── gt/
│   ├── fg_masks/
│   └──...
└──...
```

### 3. Training

#### Option A: Run in the Cloud (Linux/RunPod)

Use the provided shell script to train both models (UNet and Swin UNETR) sequentially or in parallel depending on GPU availability.

```bash
# Run for default 300 epochs
bash cloud_train.sh

# Or specify number of epochs (e.g., 100)
bash cloud_train.sh 100
```

_Note: This script automatically stop the pod once training is completed to save costs._

#### Option B: Run Locally (Windows)

Use the batch scripts to train the models separately.

**Train 3D UNet:**

```cmd
train_unet.bat
```

**Train Swin UNETR:**

```cmd
train_swin.bat
```

### 4. Evaluation

Once training is complete, use the test scripts to generate metrics on the validation/test set.

**Evaluate 3D UNet:**

```cmd
eval_unet.bat
```

**Evaluate Swin UNETR:**

```cmd
eval_swin.bat
```

### 5. Failure Mode Audits

To run the specific audits (Spatial Overlap and Uncertainty Calibration) comparing the two trained ensembles:

```cmd
audit.bat
```

---

## File Structure

- `data/`: The data used to train and evaluate the models.
- `src/`: Source code for models, training, and testing.
  - `train_unet.py`: Training script for 3D UNet.
  - `train_swin.py`: Training script for Swin UNETR.
  - `test_unet.py`: Evaluation script for UNet.
  - `test_swin.py`: Evaluation script for Swin UNETR.
  - `audit.py`: Script for failure mode comparison.
  - `data_load.py`: Data loading and augmentation pipeline.
  - `inference.py`: Script for generating predictions and uncertainty maps.
  - `metrics.py`: Implementation of evaluation metrics (Dice, nDSC, F1, etc.).
  - `retention_curves.py`: Script for generating retention curve plots.
  - `uncertainty.py`: Implementation of uncertainty measures.
- `cloud_train.sh`: Master script for cloud training.
- `*.bat`: Helper scripts for running on Windows.

---

## Acknowledgements

This project builds upon the baseline code provided by the **Shifts 2.0 Benchmark**.

- The following files are borrowed directly or adapted from the Shifts 2.0 baseline project: `src/data_load.py`, `src/inference.py`, `src/metrics.py`, `src/retention_curves.py`, `src/train_unet.py`, and `src/test_unet.py`.
- The Swin UNETR implementation (`src/train_swin.py` and `src/test_swin.py`) was written for this project, following the structure of the baseline and utilizing its utility functions (e.g., `data_load`).
- The Swin UNETR implementation (`src/train_swin.py` and `src/test_swin.py`) was written for this project, following the structure of the baseline and utilizing its utility functions (e.g., `data_load`).
