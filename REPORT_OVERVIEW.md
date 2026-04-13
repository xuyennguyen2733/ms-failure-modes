# Lego 3 Report — Running Notes

This file tracks changes made between Lego 2 and Lego 3 so they can be summarized
in a "Pre-Lego-3 Fix-ups" section at the top of the Lego 3 report. The goal of
that section is to set expectations: what was broken or weak in Lego 2, what was
fixed, and how those fixes shape the Lego 3 experiments that follow.

---

## Lego 2 Professor Feedback (Summary)

- Phase 2 results were not fully reported; hypotheses could not be evaluated against evidence.
- Comparison between 3D UNet and Swin UNETR was not controlled — architecture,
  capacity, and training behavior all changed simultaneously, confounding the
  "locality vs global context" question.
- The structural assumption ("local context is sufficient") was too absolute;
  should be refined to specify *when* global context helps.

---

## Pre-Lego-3 Fix-ups (Changes Made)

### 1. Training-data augmentation upgrade (acquisition-shift simulators)
- **Why:** Access to Shifts Part 1 was never granted, leaving only ~10 `best`
  training subjects. To expand the *effective* training distribution without
  new subjects, the augmentation pipeline was upgraded from generic
  geometric/intensity jitter to transforms that simulate **cross-scanner
  acquisition variability** — the exact gap between the `best` (source) and
  `ljubljana` (target) domains.
- **What changed** in `src/data_load.py` `get_train_transforms()`:
  - Added (pre-normalize, raw-intensity stage):
    - `RandBiasFieldd` — simulates MRI coil inhomogeneity
    - `RandAdjustContrastd` (gamma 0.7–1.5) — nonlinear protocol-driven contrast
    - `RandHistogramShiftd` — generic nonlinear intensity remapping ("different scanner")
  - Added (post-normalize stage):
    - `RandGaussianNoised` — sensor noise / SNR differences
    - `RandGaussianSmoothd` — PSF / resolution differences
    - `RandGibbsNoised` — k-space truncation ringing
  - All new transforms are **image-only** (labels untouched) and applied at
    `prob=0.2–0.3` so most patches remain clean.
- **Assumption being addressed:** the baseline silently assumed
  "training-distribution ≈ test-distribution at the acquisition level."
  The augmentation explicitly relaxes that assumption.

### Reframing: single-backbone locality study (primary), cross-model comparison (secondary)
- After adopting the patch-size sweep as the controlled locality knob, the
  *primary* analysis is now per-backbone — sweep `--patch_size` on a single
  architecture and study how its failure modes shift with the visible context.
- The original UNet-vs-Swin cross-model audit (FP-overlap IoU) is **demoted to
  a secondary add-on**: it still runs when both backbones happen to be trained,
  but it is not the headline result and can be turned off with
  `--skip_comparison` (run.py) / `--no_comparison` (audit.py).
- `src/audit.py` now accepts one OR both of `--path_unet` / `--path_swin`. The
  per-backbone uncertainty-calibration audit (entropy at FP/FN sites) runs for
  whichever ensemble is provided; the spatial-overlap comparison is gated on
  having both backbones AND `--no_comparison` not being set.
- `run.py` exposes `--models {unet,swin}` so the user can train/eval/audit a
  single backbone, and `--skip_comparison` to suppress the cross-model step
  even when both are present.

### 2. Controlled comparison: input patch-size sweep (IMPLEMENTED)
- **Why:** UNet-vs-Swin changes too many factors at once (capacity,
  optimizer dynamics, normalization, receptive field). The professor asked for
  an isolated locality knob.
- **Original plan:** Swin UNETR `window_size ∈ {3, 7, 12}` sweep.
- **Pivot:** MONAI 0.9.0's `SwinUNETR` does **not** expose `window_size` as a
  constructor argument (it is hardcoded to 7 inside `SwinTransformer`).
  Upgrading MONAI would risk breaking the new acquisition-shift augmentation
  transforms and the rest of the pinned pipeline. We keep MONAI at 0.9.0 and
  instead probe locality at the **input** level.
- **Chosen knob:** cubic **training/inference patch size** `P ∈ {64, 96, 128}`.
  This literally caps the spatial context each example exposes, applies
  identically to *both* backbones (no architectural surgery), and is the most
  operationally honest definition of "how much context the model can see".
  Both `P=64` and `P=128` are multiples of 32, satisfying SwinUNETR's
  downsampling constraint.
- **What stays fixed across the sweep:**
  - Architecture, channels/feature_size, optimizer, learning rate, loss,
    augmentation pipeline, seeds, data splits, sliding-window overlap mode.
  - Only `--patch_size` changes between runs.
- **Where it's wired:** a single `--patch_size` CLI flag was added to
  `src/train_unet.py`, `src/train_swin.py`, `src/test_unet.py`,
  `src/test_swin.py`, `src/inference.py`, `src/audit.py`, and
  `src/retention_curves.py`. In `src/data_load.py`, `get_train_transforms()`
  and `get_train_dataloader()` now accept `patch_size`, scaling the
  lesion-biased outer crop, `RandSpatialCropd`, and `RandAffined`
  simultaneously. `run.py` exposes `--patch_sizes` as a list so a full sweep
  runs from one entry point (experiment directories are tagged with
  `_p{P}` suffixes so runs don't collide).
- **Metrics:** same as Lego 2 (nDSC, Lesion-F1, R-AUC, mean entropy), reported
  per patch-size, plus stratification by lesion size to test the refined
  hypothesis (see §3).
- **Status:** implemented; runs pending.

### 3. Refined structural assumption (PARTIALLY IMPLEMENTED)
- **Old (Lego 2):** "Local spatial context is sufficient to identify MS lesions."
- **New (Lego 3):** "Local context is sufficient for typical, well-circumscribed
  lesions, but global context becomes necessary at periventricular boundaries,
  near scanner artifacts, and for ambiguous low-contrast lesions where local
  texture alone overlaps normal anatomy."
- **Lesion-size stratification implemented in `src/audit.py`:**
  - GT lesions are bucketed by voxel count: `small (<10)`, `medium (10-50)`,
    `large (>=50)`. Edges live in `LESION_SIZE_BINS` at the top of the file
    so they are easy to defend / change.
  - For each bucket and each backbone, the audit reports lesion **detection
    recall** (fraction of GT lesions for which any predicted voxel overlaps)
    and the **mean predictive entropy across missed-lesion voxels** (proxy
    for "did the model know it was missing this kind of lesion?").
  - Also writes a `stratified_recall.png` bar plot per backbone.
- **What this lets us test directly:**
  - If recall on `small` lesions improves with larger patch size (more global
    context), the refined hypothesis is supported.
  - If FN entropy on `small` lesions stays low while recall stays low, the
    model is *confidently missing* small lesions — a clinically dangerous
    failure mode worth flagging in the report.
- **Still planned:** anatomical-location stratification (periventricular vs
  juxtacortical vs deep WM) — requires segmentation atlases not currently
  in the data, so deferred.

### 4. Reporting improvements (planned)
- Per-subject distributions (boxplots) instead of mean ± std only.
- Retention curves rendered and included (already supported by
  `src/retention_curves.py`).
- Qualitative slice panel showing a UNet-only FP, a Swin-only FP, and a shared FN.
- Each results paragraph explicitly tagged *supports / contradicts / inconclusive*
  against the Phase-1 hypotheses.

---

## Lego 3 Plan Pointer

The supervision stress test for Lego 3 will be designed *on top of* the
fixed-up baseline above — not on top of the original Lego 2 code. The
supervision constraint (label scarcity / corruption / coarsening — TBD) will
be the only variable changed between the Lego-3 baseline and the Lego-3
constrained system.
