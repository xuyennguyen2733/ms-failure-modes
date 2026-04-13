"""
Qualitative visualization for the report.

Picks one subject deterministically from the eval set and renders a set of
report-ready PNGs using nilearn (as suggested in src/README.md):

  - <save_dir>/<subj>_flair_gt.png        FLAIR + GT overlay   (plot_roi)
  - <save_dir>/<subj>_flair_pred.png      FLAIR + prediction   (plot_roi)
  - <save_dir>/<subj>_gt_vs_pred.png      FLAIR with both masks contoured
  - <save_dir>/<subj>_uncertainty.png     FLAIR + RMI uncertainty heatmap
  - <save_dir>/<subj>_flair.png           FLAIR alone, axial/coronal/sagittal

The subject is picked with a seeded RNG so the same run produces the same
images on reruns — keeps the report reproducible.
"""

import argparse
import glob
import os
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting


def _sid_from_flair(path):
    m = re.match(r"(\d+)_FLAIR_isovox", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path).split(".")[0]


def _cut_coords_from_gt(gt_img):
    """MNI-space coordinates of the largest GT lesion's centroid. Falls
    back to nilearn's default 'auto' cut when the GT is empty."""
    arr = gt_img.get_fdata()
    if arr.sum() == 0:
        return None  # nilearn will pick its own
    from scipy import ndimage
    labels, n = ndimage.label(arr > 0)
    sizes = ndimage.sum(arr > 0, labels, range(1, n + 1))
    biggest = 1 + int(np.argmax(sizes))
    vox = np.round(np.mean(np.argwhere(labels == biggest), axis=0)).astype(int)
    # Voxel -> world coords through the affine
    homogeneous = np.append(vox, 1.0)
    world = gt_img.affine @ homogeneous
    return tuple(float(x) for x in world[:3])


def _binarize_img(img, threshold=0.5):
    """Return a nibabel image with a strictly binary {0,1} int8 array,
    which is what nilearn.plot_roi expects."""
    data = (img.get_fdata() >= threshold).astype(np.int8)
    return nib.Nifti1Image(data, img.affine, img.header)


def render_with_nilearn(flair_img, gt_img, pred_img, unc_img, sid, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cut_coords = _cut_coords_from_gt(gt_img)

    gt_bin = _binarize_img(gt_img)
    pred_bin = _binarize_img(pred_img)

    common_kw = dict(
        bg_img=flair_img,
        display_mode="ortho",
        cut_coords=cut_coords,
        black_bg=True,
        draw_cross=False,
    )

    # --- FLAIR alone, ortho cuts ---
    d = plotting.plot_img(flair_img, title=f"subj {sid} — FLAIR",
                          display_mode="ortho", cut_coords=cut_coords,
                          black_bg=True, draw_cross=False, cmap="gray",
                          colorbar=False)
    d.savefig(os.path.join(save_dir, f"{sid}_flair.png"), dpi=150)
    d.close()

    # --- FLAIR + GT overlay ---
    d = plotting.plot_roi(gt_bin, title=f"subj {sid} — FLAIR + GT",
                          cmap="autumn", alpha=0.55, **common_kw)
    d.savefig(os.path.join(save_dir, f"{sid}_flair_gt.png"), dpi=150)
    d.close()

    # --- FLAIR + prediction overlay ---
    d = plotting.plot_roi(pred_bin, title=f"subj {sid} — FLAIR + prediction",
                          cmap="autumn", alpha=0.55, **common_kw)
    d.savefig(os.path.join(save_dir, f"{sid}_flair_pred.png"), dpi=150)
    d.close()

    # --- GT vs prediction side-by-side (contours) ---
    d = plotting.plot_img(flair_img, title=f"subj {sid} — GT (green) vs pred (red)",
                          display_mode="ortho", cut_coords=cut_coords,
                          black_bg=True, draw_cross=False, cmap="gray",
                          colorbar=False)
    # GT contour in green, prediction contour in red.
    d.add_contours(gt_bin, levels=[0.5], colors="lime", linewidths=1.0)
    d.add_contours(pred_bin, levels=[0.5], colors="red", linewidths=1.0)
    d.savefig(os.path.join(save_dir, f"{sid}_gt_vs_pred.png"), dpi=150)
    d.close()

    # --- FLAIR + uncertainty heatmap ---
    if unc_img is not None:
        d = plotting.plot_stat_map(
            unc_img, bg_img=flair_img,
            title=f"subj {sid} — RMI uncertainty",
            display_mode="ortho", cut_coords=cut_coords,
            black_bg=True, draw_cross=False, cmap="hot",
            colorbar=True,
        )
        d.savefig(os.path.join(save_dir, f"{sid}_uncertainty.png"), dpi=150)
        d.close()


def pick_one_subject(flair_dir, seed):
    flair_files = sorted(glob.glob(os.path.join(flair_dir, "*FLAIR_isovox.nii.gz")))
    if not flair_files:
        raise FileNotFoundError(f"No FLAIR files under {flair_dir}")
    rng = random.Random(seed)
    return rng.choice(flair_files)


def main():
    ap = argparse.ArgumentParser(description="Qualitative nilearn panel for one random subject.")
    ap.add_argument("--flair_dir",   required=True, help="FLAIR volumes for the eval set.")
    ap.add_argument("--gt_dir",      required=True, help="Ground-truth masks for those subjects.")
    ap.add_argument("--pred_dir",    required=True,
                    help="Predictions dir (contains <sid>_pred_seg.nii.gz and <sid>_uncs_rmi.nii.gz).")
    ap.add_argument("--save_dir",    required=True, help="Output directory.")
    ap.add_argument("--pick_seed",   type=int, default=42,
                    help="RNG seed for picking the subject. Same seed = same subject on reruns.")
    ap.add_argument("--model_label", default="model",
                    help="Short label embedded in the save_dir name.")
    args = ap.parse_args()

    flair_path = pick_one_subject(args.flair_dir, args.pick_seed)
    sid = _sid_from_flair(flair_path)

    gt_path   = os.path.join(args.gt_dir, f"{sid}_gt_isovox.nii.gz")
    pred_path = os.path.join(args.pred_dir, f"{sid}_pred_seg.nii.gz")
    unc_path  = os.path.join(args.pred_dir, f"{sid}_uncs_rmi.nii.gz")

    for p in (flair_path, gt_path, pred_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing expected file: {p}")

    flair_img = nib.load(flair_path)
    gt_img    = nib.load(gt_path)
    pred_img  = nib.load(pred_path)
    unc_img   = nib.load(unc_path) if os.path.exists(unc_path) else None

    out_dir = os.path.join(args.save_dir, f"{args.model_label}_subj{sid}")
    render_with_nilearn(flair_img, gt_img, pred_img, unc_img, sid, out_dir)
    print(f"[qualitative_viz] wrote panel for subj {sid} -> {out_dir}")


if __name__ == "__main__":
    main()
