import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ======================== Parsing eval log files ==========================

_EVAL_LOG_NAME_RE = re.compile(
    r"(?P<model>unet|swin)_p(?P<patch>\d+)_eval_log_(?P<ts>\d{8}_\d{6})\.txt$"
)

_METRIC_LINE_RE = re.compile(
    r"\s*(?P<name>[A-Za-z0-9 ().%\-]+?)\s*:\s*"
    r"(?P<mean>-?\d+(?:\.\d+)?)\s*\+/-\s*(?P<std>-?\d+(?:\.\d+)?)"
)


def parse_eval_log(path):
    """Return dict with keys: model, patch_size, and one entry per metric
    (as (mean, std) tuples). Returns None if the file has no final summary."""
    m = _EVAL_LOG_NAME_RE.search(os.path.basename(path))
    if not m:
        return None
    entry = {
        "model":      m.group("model"),
        "patch_size": int(m.group("patch")),
        "timestamp":  m.group("ts"),
        "path":       path,
    }
    with open(path, "r", errors="replace") as f:
        text = f.read()

    # Find the final-summary section. If not present the run was interrupted.
    if "Final Summary" not in text:
        return None

    summary_block = text.split("Final Summary", 1)[1]
    for line in summary_block.splitlines():
        mm = _METRIC_LINE_RE.match(line)
        if mm:
            name = mm.group("name").strip()
            mean = float(mm.group("mean"))
            std = float(mm.group("std"))
            entry[name] = (mean, std)
    return entry


def collect_latest_eval_logs(eval_dir):
    """For each (model, patch_size) pair, keep the most recent .txt with a
    final summary."""
    latest = {}
    for path in glob.glob(os.path.join(eval_dir, "*.txt")):
        e = parse_eval_log(path)
        if e is None:
            continue
        key = (e["model"], e["patch_size"])
        if key not in latest or e["timestamp"] > latest[key]["timestamp"]:
            latest[key] = e
    return latest


# ============================ Writing outputs ==============================

METRIC_ORDER = [
    ("nDSC (%)",        "nDSC"),
    ("Lesion F1 (%)",   "F1"),
    ("nDSC R-AUC (%)",  "R-AUC"),
    ("Pred. Entropy",   "Entropy"),
]


def write_summary_csv_md(entries, out_dir):
    rows = []
    for (model, patch), e in sorted(entries.items()):
        row = {"model": model, "patch_size": patch}
        for key, short in METRIC_ORDER:
            mean, std = e.get(key, (float("nan"), float("nan")))
            row[f"{short}_mean"] = mean
            row[f"{short}_std"] = std
        rows.append(row)

    # CSV
    csv_path = os.path.join(out_dir, "summary_table.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[aggregate] wrote {csv_path}")

    # Markdown
    md_path = os.path.join(out_dir, "summary_table.md")
    with open(md_path, "w") as f:
        header = ["model", "patch"] + [short for _, short in METRIC_ORDER]
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in rows:
            cells = [row["model"], str(row["patch_size"])]
            for _, short in METRIC_ORDER:
                mean = row[f"{short}_mean"]
                std = row[f"{short}_std"]
                cells.append(f"{mean:.2f} ± {std:.2f}")
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"[aggregate] wrote {md_path}")
    return rows


# ============================ Stratified recall ============================

def collect_audit_raw(viz_dir):
    """Walk visualization/p{P}/audit_raw.json and return a dict
    keyed by patch_size."""
    out = {}
    for json_path in glob.glob(os.path.join(viz_dir, "p*", "audit_raw.json")):
        m = re.search(r"p(\d+)", os.path.basename(os.path.dirname(json_path)))
        if not m:
            continue
        with open(json_path, "r") as f:
            out[int(m.group(1))] = json.load(f)
    return out


def write_stratified_csv(audit_raw, out_dir):
    csv_path = os.path.join(out_dir, "stratified_recall.csv")
    rows = []
    for patch, data in sorted(audit_raw.items()):
        for model_key, strata_key in (("unet", "unet_strata"),
                                       ("swin", "swin_strata")):
            strata = data.get(strata_key, {})
            if not strata:
                continue
            for bucket_name, bucket in strata.items():
                n_total = bucket["n_total"]
                n_detected = bucket["n_detected"]
                recall = (n_detected / n_total) if n_total > 0 else float("nan")
                fn_ent = (float(np.mean(bucket["fn_entropies"]))
                          if bucket["fn_entropies"] else float("nan"))
                rows.append({
                    "model": model_key,
                    "patch_size": patch,
                    "bucket": bucket_name,
                    "n_total": n_total,
                    "n_detected": n_detected,
                    "recall": recall,
                    "mean_fn_entropy": fn_ent,
                })
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[aggregate] wrote {csv_path}")
    return rows


def plot_recall_by_patch_size(strat_rows, out_dir):
    """Line plot: recall vs patch_size, one line per (model, bucket).

    This is the direct visualization of the refined-hypothesis question:
    "Does growing the context (patch size) help most for small lesions?"
    """
    if not strat_rows:
        return
    by_series = defaultdict(list)
    for r in strat_rows:
        key = (r["model"], r["bucket"])
        by_series[key].append((r["patch_size"], r["recall"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, bucket), pts in sorted(by_series.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ls = "-" if model == "unet" else "--"
        ax.plot(xs, ys, marker="o", linestyle=ls, label=f"{model} / {bucket}")
    ax.set_xlabel("Training/inference patch size (voxels)")
    ax.set_ylabel("Lesion detection recall")
    ax.set_ylim(0, 1.0)
    ax.set_title("Lesion-size-stratified recall vs. spatial context")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    png_path = os.path.join(out_dir, "recall_by_patch_size.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[aggregate] wrote {png_path}")


def write_audit_scalars_csv(audit_raw, out_dir):
    csv_path = os.path.join(out_dir, "audit_summary.csv")
    rows = []
    for patch, data in sorted(audit_raw.items()):
        row = {"patch_size": patch}
        for k in ("unet_fp_entropy", "swin_fp_entropy",
                  "unet_fn_entropy", "swin_fn_entropy"):
            vals = data.get(k, [])
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{k}_n"] = len(vals)
        fp_ious = [x for x in data.get("fp_ious", []) if x is not None]
        row["fp_iou_mean"] = float(np.mean(fp_ious)) if fp_ious else float("nan")
        row["fp_iou_n"] = len(fp_ious)
        rows.append(row)
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[aggregate] wrote {csv_path}")


# ========================== Retention curves =============================

def plot_combined_retention(rc_root, out_dir):
    """Walk retention_curves/p*/nDSC_rc_*.npy and put everything on one axes."""
    fracs_paths = glob.glob(os.path.join(rc_root, "p*", "fracs_retained.npy"))
    if not fracs_paths:
        return
    # Grab any fracs_retained — they are all identical by construction.
    fracs = np.load(fracs_paths[0])

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0
    for p_dir in sorted(glob.glob(os.path.join(rc_root, "p*"))):
        for npy in sorted(glob.glob(os.path.join(p_dir, "nDSC_rc_*.npy"))):
            # Skip the per-subject matrix file
            name = os.path.basename(npy)
            if name.startswith("nDSC_rc_all_"):
                continue
            label = name.replace("nDSC_rc_", "").replace(".npy", "")
            y = np.load(npy)
            if y.ndim != 1 or len(y) != len(fracs):
                continue
            ax.plot(fracs, y, marker="", label=label)
            plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("Retention Fraction")
    ax.set_ylabel("nDSC")
    ax.set_xlim(0, 1.01)
    ax.set_title("nDSC Retention Curves — all runs")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    png_path = os.path.join(out_dir, "retention_curves_combined.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[aggregate] wrote {png_path}")


# ================================= main ==================================

def _discover_roots(root):
    """Return a list of (aug_profile, root_path) pairs.

    Handles two cases transparently:
      1. Nested leaf: `root` directly contains eval_reports/ etc. In this
         case we extract the profile from the basename (run_aug-<profile>)
         and return a single entry. Defaults to "full" if the basename
         does not match.
      2. Top-level parent: `root` contains one or more run_aug-*/ sub-
         directories. Return one entry per subdirectory. This is how the
         aggregator gets run manually when the user wants a combined
         baseline-vs-ablation view.
    """
    # Case 1: leaf
    if os.path.isdir(os.path.join(root, "eval_reports")) or \
       os.path.isdir(os.path.join(root, "visualization")):
        base = os.path.basename(os.path.normpath(root))
        m = re.match(r"run_aug-(.+)", base)
        profile = m.group(1) if m else "full"
        return [(profile, root)]

    # Case 2: parent containing nested profile dirs
    pairs = []
    for entry in sorted(os.listdir(root)):
        if not entry.startswith("run_aug-"):
            continue
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        profile = entry[len("run_aug-"):]
        pairs.append((profile, sub))
    return pairs


def _aggregate_one(root, profile):
    """Walk a single leaf root and return the collected data structures.
    Does not write anything on its own — the caller merges and writes."""
    per_profile = {
        "profile": profile,
        "entries": {},       # (model, patch) -> eval entry
        "audit_raw": {},     # patch -> audit_raw dict
    }

    eval_dir = os.path.join(root, "eval_reports")
    if os.path.isdir(eval_dir):
        per_profile["entries"] = collect_latest_eval_logs(eval_dir)
    else:
        print(f"[aggregate] [{profile}] missing {eval_dir} — skipping eval logs")

    viz_dir = os.path.join(root, "visualization")
    if os.path.isdir(viz_dir):
        per_profile["audit_raw"] = collect_audit_raw(viz_dir)
    else:
        print(f"[aggregate] [{profile}] missing {viz_dir} — skipping audit raw")

    return per_profile


def main():
    ap = argparse.ArgumentParser(description="Aggregate pipeline outputs.")
    ap.add_argument("--output_root", default="download",
                    help="Directory that run.py wrote its artifacts to. "
                         "Works both with a nested profile root "
                         "(e.g. download/run_aug-full) and with a top-level "
                         "parent that contains several run_aug-*/ subdirs.")
    args = ap.parse_args()

    root = args.output_root
    out_dir = os.path.join(root, "aggregated")
    os.makedirs(out_dir, exist_ok=True)

    profile_roots = _discover_roots(root)
    if not profile_roots:
        print(f"[aggregate] no profile-tagged subdirs or leaf artifacts found under {root}")
        return

    print(f"[aggregate] found {len(profile_roots)} profile root(s): "
          f"{[p for p, _ in profile_roots]}")

    # --- Collect per-profile ---
    per_profile_data = [_aggregate_one(r, p) for p, r in profile_roots]

    # --- Summary table (tagged with profile column) ---
    all_entries = {}  # (profile, model, patch) -> entry
    for pp in per_profile_data:
        for (model, patch), entry in pp["entries"].items():
            all_entries[(pp["profile"], model, patch)] = entry
    if all_entries:
        _write_summary_across_profiles(all_entries, out_dir)
    else:
        print("[aggregate] no eval-log entries found across any profile")

    # --- Stratified recall (tagged with profile column) ---
    all_strat_rows = []
    for pp in per_profile_data:
        if not pp["audit_raw"]:
            continue
        for row in _stratified_rows_from_raw(pp["audit_raw"]):
            row["profile"] = pp["profile"]
            all_strat_rows.append(row)
    if all_strat_rows:
        _write_stratified_across_profiles(all_strat_rows, out_dir)
        _plot_recall_by_patch_size_multi_profile(all_strat_rows, out_dir)
    else:
        print("[aggregate] no audit_raw.json found across any profile")

    # --- Audit scalars per profile (tagged) ---
    all_audit_rows = []
    for pp in per_profile_data:
        if not pp["audit_raw"]:
            continue
        for row in _audit_scalar_rows_from_raw(pp["audit_raw"]):
            row["profile"] = pp["profile"]
            all_audit_rows.append(row)
    if all_audit_rows:
        _write_audit_scalars_across_profiles(all_audit_rows, out_dir)

    # --- Combined retention curves (aggregated across all profile roots) ---
    # plot_combined_retention walks <root>/retention_curves/ which only
    # exists at the leaf level. Call it once per profile and merge.
    _plot_retention_across_profiles(profile_roots, out_dir)

    print(f"\n[aggregate] done — artifacts under: {os.path.abspath(out_dir)}")


# ------------------- helpers for the multi-profile path --------------------

def _write_summary_across_profiles(entries, out_dir):
    rows = []
    for (profile, model, patch), e in sorted(entries.items()):
        row = {"profile": profile, "model": model, "patch_size": patch}
        for key, short in METRIC_ORDER:
            mean, std = e.get(key, (float("nan"), float("nan")))
            row[f"{short}_mean"] = mean
            row[f"{short}_std"] = std
        rows.append(row)

    csv_path = os.path.join(out_dir, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[aggregate] wrote {csv_path}")

    md_path = os.path.join(out_dir, "summary_table.md")
    with open(md_path, "w") as f:
        header = ["profile", "model", "patch"] + [s for _, s in METRIC_ORDER]
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in rows:
            cells = [row["profile"], row["model"], str(row["patch_size"])]
            for _, short in METRIC_ORDER:
                mean = row[f"{short}_mean"]
                std = row[f"{short}_std"]
                cells.append(f"{mean:.2f} ± {std:.2f}")
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"[aggregate] wrote {md_path}")


def _stratified_rows_from_raw(audit_raw):
    rows = []
    for patch, data in sorted(audit_raw.items()):
        for model_key, strata_key in (("unet", "unet_strata"),
                                       ("swin", "swin_strata")):
            strata = data.get(strata_key, {})
            if not strata:
                continue
            for bucket_name, bucket in strata.items():
                n_total = bucket["n_total"]
                n_detected = bucket["n_detected"]
                recall = (n_detected / n_total) if n_total > 0 else float("nan")
                fn_ent = (float(np.mean(bucket["fn_entropies"]))
                          if bucket["fn_entropies"] else float("nan"))
                rows.append({
                    "model": model_key,
                    "patch_size": patch,
                    "bucket": bucket_name,
                    "n_total": n_total,
                    "n_detected": n_detected,
                    "recall": recall,
                    "mean_fn_entropy": fn_ent,
                })
    return rows


def _audit_scalar_rows_from_raw(audit_raw):
    rows = []
    for patch, data in sorted(audit_raw.items()):
        row = {"patch_size": patch}
        for k in ("unet_fp_entropy", "swin_fp_entropy",
                  "unet_fn_entropy", "swin_fn_entropy"):
            vals = data.get(k, [])
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{k}_n"] = len(vals)
        fp_ious = [x for x in data.get("fp_ious", []) if x is not None]
        row["fp_iou_mean"] = float(np.mean(fp_ious)) if fp_ious else float("nan")
        row["fp_iou_n"] = len(fp_ious)
        rows.append(row)
    return rows


def _write_stratified_across_profiles(rows, out_dir):
    csv_path = os.path.join(out_dir, "stratified_recall.csv")
    fieldnames = ["profile", "model", "patch_size", "bucket",
                  "n_total", "n_detected", "recall", "mean_fn_entropy"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[aggregate] wrote {csv_path}")


def _write_audit_scalars_across_profiles(rows, out_dir):
    csv_path = os.path.join(out_dir, "audit_summary.csv")
    fieldnames = ["profile"] + [k for k in rows[0].keys() if k != "profile"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[aggregate] wrote {csv_path}")


def _plot_recall_by_patch_size_multi_profile(rows, out_dir):
    """One line per (profile, model, bucket). When a single profile is
    present this collapses to the original plot layout."""
    if not rows:
        return
    by_series = defaultdict(list)
    for r in rows:
        key = (r["profile"], r["model"], r["bucket"])
        by_series[key].append((r["patch_size"], r["recall"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    for (profile, model, bucket), pts in sorted(by_series.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ls = "-" if model == "unet" else "--"
        ax.plot(xs, ys, marker="o", linestyle=ls,
                label=f"{profile}/{model}/{bucket}")
    ax.set_xlabel("Training/inference patch size (voxels)")
    ax.set_ylabel("Lesion detection recall")
    ax.set_ylim(0, 1.0)
    ax.set_title("Lesion-size-stratified recall vs. spatial context")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    png_path = os.path.join(out_dir, "recall_by_patch_size.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[aggregate] wrote {png_path}")


def _plot_retention_across_profiles(profile_roots, out_dir):
    """Walk each leaf's retention_curves/ and put every curve on one axes,
    labeled with profile+patch+model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    fracs_seen = None
    for profile, root in profile_roots:
        rc_root = os.path.join(root, "retention_curves")
        if not os.path.isdir(rc_root):
            continue
        for p_dir in sorted(glob.glob(os.path.join(rc_root, "p*"))):
            fracs_path = os.path.join(p_dir, "fracs_retained.npy")
            if not os.path.exists(fracs_path):
                continue
            fracs_seen = np.load(fracs_path)
            for npy in sorted(glob.glob(os.path.join(p_dir, "nDSC_rc_*.npy"))):
                name = os.path.basename(npy)
                if name.startswith("nDSC_rc_all_"):
                    continue
                label = name.replace("nDSC_rc_", "").replace(".npy", "")
                y = np.load(npy)
                if y.ndim != 1 or len(y) != len(fracs_seen):
                    continue
                ax.plot(fracs_seen, y, label=f"{profile}/{label}")
                plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("Retention Fraction")
    ax.set_ylabel("nDSC")
    ax.set_xlim(0, 1.01)
    ax.set_title("nDSC Retention Curves — all runs")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    png_path = os.path.join(out_dir, "retention_curves_combined.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[aggregate] wrote {png_path}")


if __name__ == "__main__":
    main()
