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

### 2. Controlled comparison: Swin UNETR window-size sweep (planned)
- **Why:** UNet-vs-Swin changes too many factors at once. To isolate the
  *locality* knob, keep **one backbone** (Swin UNETR) and vary only the
  attention `window_size`, which directly controls how much global context
  each token sees.
- **Plan:** train Swin UNETR with `window_size ∈ {3, 7, 12}` under otherwise
  identical settings (data, loss, optimizer, seeds, augmentation pipeline).
- **Status:** not yet implemented.

### 3. Refined structural assumption (planned)
- **Old (Lego 2):** "Local spatial context is sufficient to identify MS lesions."
- **New (Lego 3):** "Local context is sufficient for typical, well-circumscribed
  lesions, but global context becomes necessary at periventricular boundaries,
  near scanner artifacts, and for ambiguous low-contrast lesions where local
  texture alone overlaps normal anatomy."
- **Status:** to be reflected in Phase-1 write-up and in a lesion-stratified
  audit (by size and/or anatomical location) in `src/audit.py`.

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
