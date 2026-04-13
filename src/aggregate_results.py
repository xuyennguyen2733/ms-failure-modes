"""
Post-sweep aggregator for the Lego-3 pipeline.

Walks a completed `--output_root` directory and produces a single place
containing everything the report needs:

  <output_root>/aggregated/
    ├── summary_table.csv           one row per (model, patch_size) with mean/std
    ├── summary_table.md            markdown table to paste into the report
    ├── stratified_recall.csv       one row per (model, patch_size, bucket)
    ├── recall_by_patch_size.png    line plot — the refined-hypothesis figure
    ├── retention_curves_combined.png   all retention curves on one axes
    └── audit_summary.csv           per-patch-size audit scalars (FP IoU, entropies)

Inputs consumed (all produced by run.py):
  - eval_reports/<model>_p<P>_eval_log_*.txt   (eval metrics)
  - visualization/p<P>/audit_raw.json          (audit + stratification)
  - retention_curves/p<P>/nDSC_rc_<label>.npy  (+ fracs_retained.npy)

The aggregator only reads files; it never re-runs GPU work. It is safe to
call repeatedly — each call overwrites the files under `aggregated/`.
"""

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

def main():
    ap = argparse.ArgumentParser(description="Aggregate Lego-3 pipeline outputs.")
    ap.add_argument("--output_root", default="download",
                    help="Directory that run.py wrote its artifacts to.")
    args = ap.parse_args()

    root = args.output_root
    out_dir = os.path.join(root, "aggregated")
    os.makedirs(out_dir, exist_ok=True)

    # --- Summary table from eval logs ---
    eval_dir = os.path.join(root, "eval_reports")
    if os.path.isdir(eval_dir):
        entries = collect_latest_eval_logs(eval_dir)
        if entries:
            write_summary_csv_md(entries, out_dir)
        else:
            print(f"[aggregate] no completed eval logs found in {eval_dir}")
    else:
        print(f"[aggregate] missing {eval_dir} — skipping summary table")

    # --- Stratified recall + audit scalars ---
    viz_dir = os.path.join(root, "visualization")
    if os.path.isdir(viz_dir):
        audit_raw = collect_audit_raw(viz_dir)
        if audit_raw:
            strat_rows = write_stratified_csv(audit_raw, out_dir)
            plot_recall_by_patch_size(strat_rows, out_dir)
            write_audit_scalars_csv(audit_raw, out_dir)
        else:
            print(f"[aggregate] no audit_raw.json files found under {viz_dir}")
    else:
        print(f"[aggregate] missing {viz_dir} — skipping stratified/audit outputs")

    # --- Combined retention curves ---
    rc_dir = os.path.join(root, "retention_curves")
    if os.path.isdir(rc_dir):
        plot_combined_retention(rc_dir, out_dir)
    else:
        print(f"[aggregate] missing {rc_dir} — skipping combined retention plot")

    print(f"\n[aggregate] done — artifacts under: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
