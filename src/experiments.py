"""Post-hoc failure-mode experiments. Reads saved predictions and produces
CSV/PNG outputs under <output_root>/run_aug-<profile>/experiments/<test>/.

All tests operate on the files dumped by run_inference() to
`predictions_{model}_p{P}/*_{pred_prob,pred_seg,uncs_rmi}.nii.gz` and the
matching ground-truth volumes under --data_root (default: data/eval_in).
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage


LESION_SIZE_BINS = [
    ("small (<10 vox)", 1, 10),
    ("medium (10-50)", 10, 50),
    ("large (>=50)", 50, float("inf")),
]

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
L_MINS = [0, 3, 9, 20, 50]
DEFAULT_THRESHOLD = 0.35
DEFAULT_LMIN = 9
TOP_N_PANELS = 3


# --------------------------- shared helpers ---------------------------

def _voxel_prf(seg, gt, mask=None):
    if mask is not None:
        seg = seg[mask == 1]
        gt = gt[mask == 1]
    tp = int(((seg == 1) & (gt == 1)).sum())
    fp = int(((seg == 1) & (gt == 0)).sum())
    fn = int(((seg == 0) & (gt == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1


def _remove_small(seg, l_min):
    if l_min <= 0:
        return seg.astype(np.uint8)
    labels, n = ndimage.label(seg)
    if n == 0:
        return seg.astype(np.uint8)
    sizes = ndimage.sum(seg, labels, range(1, n + 1))
    out = np.zeros_like(seg, dtype=np.uint8)
    for i, sz in enumerate(sizes, start=1):
        if sz > l_min:
            out[labels == i] = 1
    return out


def _stratify_gt(gt, seg, bm):
    labels, n = ndimage.label((gt == 1) & (bm == 1))
    out = {name: {"n_total": 0, "n_detected": 0}
           for name, _, _ in LESION_SIZE_BINS}
    for i in range(1, n + 1):
        mask = labels == i
        size = int(mask.sum())
        bucket = None
        for name, lo, hi in LESION_SIZE_BINS:
            if lo <= size < hi:
                bucket = name
                break
        if bucket is None:
            continue
        out[bucket]["n_total"] += 1
        if (seg[mask] == 1).any():
            out[bucket]["n_detected"] += 1
    return out


# Recognized splits -> (pred-dir suffix, default data_root)
# "" suffix = run_inference output (eval_in), "_devout" = test-script dump.
SPLITS = {
    "eval_in": ("",        os.path.join("data", "eval_in")),
    "dev_out": ("_devout", os.path.join("data", "dev_out")),
}


def _iter_predictions(profile_root, models, suffix):
    """Yield dicts with model/patch_size/sid/pred_prob_path/pred_dir.
    `suffix` selects which pred-dir flavor to walk (see SPLITS)."""
    for model in models:
        pattern = os.path.join(profile_root, f"predictions_{model}_p*{suffix}")
        for pred_dir in sorted(glob.glob(pattern)):
            m = re.search(r"p(\d+)" + re.escape(suffix) + r"$", pred_dir)
            if not m:
                continue
            patch = int(m.group(1))
            for pp in sorted(glob.glob(os.path.join(pred_dir, "*_pred_prob.nii.gz"))):
                sid = os.path.basename(pp).split("_pred_prob")[0]
                yield {
                    "model": model,
                    "patch_size": patch,
                    "sid": sid,
                    "pred_dir": pred_dir,
                    "pred_prob_path": pp,
                }


def _load_gt_and_bm(sid, data_root):
    gt_path = os.path.join(data_root, "gt", f"{sid}_gt_isovox.nii.gz")
    bm_path = os.path.join(data_root, "fg_mask", f"{sid}_isovox_fg_mask.nii.gz")
    if not os.path.exists(gt_path):
        return None, None
    gt = nib.load(gt_path).get_fdata().astype(np.uint8)
    if os.path.exists(bm_path):
        bm = nib.load(bm_path).get_fdata().astype(np.uint8)
    else:
        bm = np.ones_like(gt)
    return gt, bm


def _pick_lesion_slice(gt):
    labels, n = ndimage.label(gt > 0)
    if n == 0:
        return gt.shape[2] // 2
    sizes = ndimage.sum(gt > 0, labels, range(1, n + 1))
    biggest = 1 + int(np.argmax(sizes))
    zs = np.argwhere(labels == biggest)[:, 2]
    return int(np.round(zs.mean()))


def _overlay(ax, flair_slice, mask_slice, cmap, alpha=0.5):
    ax.imshow(flair_slice.T, cmap="gray", origin="lower")
    if mask_slice.any():
        m = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
        ax.imshow(m, cmap=cmap, alpha=alpha, origin="lower")
    ax.axis("off")


def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[experiments] wrote {path}")


# --------------------------- TEST A: threshold sweep ---------------------------

def test_threshold(profile_root, data_root, models, out_dir, suffix):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for pred in _iter_predictions(profile_root, models, suffix):
        gt, bm = _load_gt_and_bm(pred["sid"], data_root)
        if gt is None:
            continue
        prob = nib.load(pred["pred_prob_path"]).get_fdata()
        for th in THRESHOLDS:
            seg = _remove_small((prob >= th).astype(np.uint8), DEFAULT_LMIN)
            p, r, f1 = _voxel_prf(seg, gt, bm)
            rows.append({
                "model": pred["model"],
                "patch_size": pred["patch_size"],
                "subject": pred["sid"],
                "threshold": th,
                "precision": round(p, 6),
                "recall": round(r, 6),
                "f1": round(f1, 6),
            })

    _write_csv(rows, os.path.join(out_dir, "threshold_sweep.csv"))
    if not rows:
        return

    by_series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_series[(r["model"], r["patch_size"])][r["threshold"]].append(r["f1"])

    # F1 vs threshold plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for (model, patch), th_map in sorted(by_series.items()):
        xs = sorted(th_map.keys())
        ys = [float(np.mean(th_map[t])) for t in xs]
        ax.plot(xs, ys, marker="o", label=f"{model}_p{patch}")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Voxel F1 (mean across subjects)")
    ax.set_title("F1 vs binarization threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f1_vs_threshold.png"), dpi=150)
    plt.close(fig)
    print(f"[experiments] wrote f1_vs_threshold.png")

    # Best-threshold summary
    best_rows = []
    for (model, patch), th_map in sorted(by_series.items()):
        pairs = [(t, float(np.mean(th_map[t]))) for t in sorted(th_map.keys())]
        best_t, best_f1 = max(pairs, key=lambda x: x[1])
        best_rows.append({"model": model, "patch_size": patch,
                          "best_threshold": best_t, "best_f1": round(best_f1, 4)})
    _write_csv(best_rows, os.path.join(out_dir, "optimal_thresholds.csv"))


# --------------------------- TEST B: l_min sweep ---------------------------

def test_lmin(profile_root, data_root, models, out_dir, suffix):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for pred in _iter_predictions(profile_root, models, suffix):
        gt, bm = _load_gt_and_bm(pred["sid"], data_root)
        if gt is None:
            continue
        prob = nib.load(pred["pred_prob_path"]).get_fdata()
        seg_raw = (prob >= DEFAULT_THRESHOLD).astype(np.uint8)
        for lmin in L_MINS:
            seg = _remove_small(seg_raw, lmin)
            strata = _stratify_gt(gt, seg, bm)
            for bucket_name, _, _ in LESION_SIZE_BINS:
                s = strata[bucket_name]
                recall = (s["n_detected"] / s["n_total"]) if s["n_total"] > 0 else float("nan")
                rows.append({
                    "model": pred["model"],
                    "patch_size": pred["patch_size"],
                    "subject": pred["sid"],
                    "l_min": lmin,
                    "bucket": bucket_name,
                    "n_total": s["n_total"],
                    "n_detected": s["n_detected"],
                    "recall": round(recall, 6) if recall == recall else "",
                })

    _write_csv(rows, os.path.join(out_dir, "lmin_sweep.csv"))
    if not rows:
        return

    buckets = [n for n, _, _ in LESION_SIZE_BINS]
    by_series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["recall"] == "" or r["n_total"] == 0:
            continue
        by_series[(r["bucket"], r["model"], r["patch_size"])][r["l_min"]].append(r["recall"])

    fig, axes = plt.subplots(1, len(buckets), figsize=(4 * len(buckets), 4), sharey=True)
    if len(buckets) == 1:
        axes = [axes]
    for ax, bucket in zip(axes, buckets):
        for (b, model, patch), lmin_map in sorted(by_series.items()):
            if b != bucket:
                continue
            xs = sorted(lmin_map.keys())
            ys = [float(np.mean(lmin_map[x])) for x in xs]
            ax.plot(xs, ys, marker="o", label=f"{model}_p{patch}")
        ax.set_xlabel("l_min (min lesion voxels kept)")
        ax.set_title(bucket)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Lesion detection recall")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Recall vs post-processing l_min filter")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "recall_vs_lmin.png"), dpi=150)
    plt.close(fig)
    print(f"[experiments] wrote recall_vs_lmin.png")


# --------------------------- TEST C: patch-size disagreement ---------------------------

def test_disagreement(profile_root, data_root, models, out_dir, suffix):
    """Rank subjects by cross-patch-size F1 delta and render panels for
    the top `TOP_N_PANELS` disagreements."""
    os.makedirs(out_dir, exist_ok=True)

    # (model, sid) -> {patch_size: pred_prob_path}
    by_subj = defaultdict(dict)
    for pred in _iter_predictions(profile_root, models, suffix):
        by_subj[(pred["model"], pred["sid"])][pred["patch_size"]] = pred["pred_prob_path"]

    rows = []
    cache = []  # (delta, model, sid, patches_sorted, segs_dict, gt, bm)
    for (model, sid), patch_paths in by_subj.items():
        if len(patch_paths) < 2:
            continue
        gt, bm = _load_gt_and_bm(sid, data_root)
        if gt is None:
            continue

        segs, f1s = {}, {}
        for patch, path in patch_paths.items():
            prob = nib.load(path).get_fdata()
            seg = _remove_small((prob >= DEFAULT_THRESHOLD).astype(np.uint8), DEFAULT_LMIN)
            segs[patch] = seg
            _, _, f1 = _voxel_prf(seg, gt, bm)
            f1s[patch] = f1

        patches_sorted = sorted(f1s.keys())
        delta = max(f1s.values()) - min(f1s.values())
        rows.append({
            "model": model,
            "subject": sid,
            "patches": ",".join(str(p) for p in patches_sorted),
            "f1_per_patch": ",".join(f"{f1s[p]:.4f}" for p in patches_sorted),
            "delta_f1": round(delta, 6),
        })
        cache.append((delta, model, sid, patches_sorted, segs, gt, bm))

    rows.sort(key=lambda r: -r["delta_f1"])
    _write_csv(rows, os.path.join(out_dir, "disagreement_ranked.csv"))

    cache.sort(key=lambda t: -t[0])
    for rank, (delta, model, sid, patches, segs, gt, bm) in enumerate(cache[:TOP_N_PANELS], 1):
        flair_path = os.path.join(data_root, "flair", f"{sid}_FLAIR_isovox.nii.gz")
        if not os.path.exists(flair_path):
            continue
        flair = nib.load(flair_path).get_fdata()
        z = _pick_lesion_slice(gt)

        cols = 1 + len(patches)
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
        _overlay(axes[0], flair[:, :, z], gt[:, :, z], "autumn", alpha=0.5)
        axes[0].set_title("FLAIR + GT")
        for ax, p in zip(axes[1:], patches):
            _overlay(ax, flair[:, :, z], segs[p][:, :, z], "cool", alpha=0.5)
            ax.set_title(f"pred p{p}")
        fig.suptitle(f"rank{rank} {model} subj{sid}  ΔF1={delta:.3f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"rank{rank}_{model}_subj{sid}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[experiments] wrote rank{rank}_{model}_subj{sid}.png")


# --------------------------- TEST D: confident-wrong cases ---------------------------

def test_confident_wrong(profile_root, data_root, models, out_dir, suffix):
    """Rank subjects by 'confidently missing small lesions' score:
    small-lesion recall LOW + FN entropy LOW = confidently wrong."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for pred in _iter_predictions(profile_root, models, suffix):
        gt, bm = _load_gt_and_bm(pred["sid"], data_root)
        if gt is None:
            continue
        unc_path = pred["pred_prob_path"].replace("_pred_prob", "_uncs_rmi")
        if not os.path.exists(unc_path):
            continue

        prob = nib.load(pred["pred_prob_path"]).get_fdata()
        unc = nib.load(unc_path).get_fdata()
        seg = _remove_small((prob >= DEFAULT_THRESHOLD).astype(np.uint8), DEFAULT_LMIN)

        labels, n = ndimage.label((gt == 1) & (bm == 1))
        small_total = 0
        small_missed = 0
        missed_entropies = []
        for i in range(1, n + 1):
            mask = labels == i
            size = int(mask.sum())
            if not (1 <= size < 10):
                continue
            small_total += 1
            if not (seg[mask] == 1).any():
                small_missed += 1
                missed_entropies.append(float(unc[mask].mean()))
        if small_total == 0 or small_missed == 0:
            continue

        recall = 1.0 - (small_missed / small_total)
        mean_fn_ent = float(np.mean(missed_entropies))
        rows.append({
            "model": pred["model"],
            "patch_size": pred["patch_size"],
            "subject": pred["sid"],
            "small_total": small_total,
            "small_missed": small_missed,
            "small_recall": round(recall, 6),
            "mean_fn_entropy": round(mean_fn_ent, 6),
            "conf_wrong_score": round(mean_fn_ent + recall, 6),  # lower = worse
        })

    rows.sort(key=lambda r: r["conf_wrong_score"])
    _write_csv(rows, os.path.join(out_dir, "confident_wrong_ranked.csv"))

    for rank, r in enumerate(rows[:TOP_N_PANELS], 1):
        sid = r["subject"]
        model = r["model"]
        patch = r["patch_size"]
        flair_path = os.path.join(data_root, "flair", f"{sid}_FLAIR_isovox.nii.gz")
        if not os.path.exists(flair_path):
            continue
        flair = nib.load(flair_path).get_fdata()
        gt, _ = _load_gt_and_bm(sid, data_root)
        pred_dir = os.path.join(profile_root, f"predictions_{model}_p{patch}{suffix}")
        prob = nib.load(os.path.join(pred_dir, f"{sid}_pred_prob.nii.gz")).get_fdata()
        unc = nib.load(os.path.join(pred_dir, f"{sid}_uncs_rmi.nii.gz")).get_fdata()
        seg = _remove_small((prob >= DEFAULT_THRESHOLD).astype(np.uint8), DEFAULT_LMIN)

        z = _pick_lesion_slice(gt)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        _overlay(axes[0], flair[:, :, z], gt[:, :, z], "autumn", alpha=0.5)
        axes[0].set_title("FLAIR + GT")
        _overlay(axes[1], flair[:, :, z], seg[:, :, z], "cool", alpha=0.5)
        axes[1].set_title(f"pred {model}_p{patch}")
        axes[2].imshow(flair[:, :, z].T, cmap="gray", origin="lower")
        axes[2].imshow(unc[:, :, z].T, cmap="hot", alpha=0.55, origin="lower")
        axes[2].set_title("uncertainty (RMI)")
        axes[2].axis("off")

        fig.suptitle(
            f"rank{rank} {model}_p{patch} subj{sid}  "
            f"small-recall={r['small_recall']:.2f}  "
            f"mean FN ent={r['mean_fn_entropy']:.4f}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"rank{rank}_{model}_p{patch}_subj{sid}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        print(f"[experiments] wrote rank{rank}_{model}_p{patch}_subj{sid}.png")


# --------------------------- CLI ---------------------------

TESTS = {
    "threshold":       test_threshold,
    "lmin":            test_lmin,
    "disagreement":    test_disagreement,
    "confident_wrong": test_confident_wrong,
}


def main():
    ap = argparse.ArgumentParser(description="Post-hoc failure-mode experiments.")
    ap.add_argument("--test", choices=list(TESTS.keys()) + ["all"], default="all",
                    help="Which experiment to run (default: all).")
    ap.add_argument("--output_root", default="download",
                    help="Matches run.py --output_root.")
    ap.add_argument("--aug_profile", default="full",
                    help="Selects <output_root>/run_aug-<profile>/ as the "
                         "profile subtree to read from and write into.")
    ap.add_argument("--splits", nargs="+", default=["dev_out"],
                    choices=list(SPLITS.keys()),
                    help="Which prediction dumps to run against. 'dev_out' "
                         "reads predictions_*_p*_devout/ (written by the eval "
                         "stage). 'eval_in' reads predictions_*_p*/ (written "
                         "by the inference stage). Default: dev_out only.")
    ap.add_argument("--data_root", default=None,
                    help="Override the data_root for ALL splits. If unset, "
                         "each split uses its registered default "
                         "(eval_in -> data/eval_in, dev_out -> data/dev_out).")
    ap.add_argument("--models", nargs="+", default=["unet", "swin"],
                    choices=["unet", "swin"])
    args = ap.parse_args()

    profile_root = os.path.join(args.output_root, f"run_aug-{args.aug_profile}")
    if not os.path.isdir(profile_root):
        print(f"[experiments] profile root not found: {profile_root}")
        sys.exit(1)
    base_out = os.path.join(profile_root, "experiments")
    os.makedirs(base_out, exist_ok=True)

    tests_to_run = list(TESTS.keys()) if args.test == "all" else [args.test]
    for split in args.splits:
        suffix, default_data_root = SPLITS[split]
        data_root = args.data_root or default_data_root
        print(f"\n[experiments] === split={split}  suffix={suffix!r}  "
              f"data_root={data_root} ===")
        for t in tests_to_run:
            out_dir = os.path.join(base_out, split, t)
            print(f"[experiments] running: {t} -> {out_dir}")
            TESTS[t](profile_root, data_root, args.models, out_dir, suffix)

    print("\n[experiments] done")


if __name__ == "__main__":
    main()
