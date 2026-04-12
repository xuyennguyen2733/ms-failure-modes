#!/bin/bash
#chmod +x run.py
#apt update -y && apt install -y python3.9 python3.9-venv

import argparse
import os
import subprocess
import sys


def install_requirements():
    print("\n>>> Installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)


def _exp_dir(output_root, base, patch_size):
    """Experiment directory tagged with the patch size so sweeps don't collide.
    Rooted under `output_root` so every generated artifact (checkpoints,
    predictions, visualizations) lives under one parent dir — makes it trivial
    to download all outputs from RunPod with a single command."""
    return os.path.join(output_root, f"{base}_p{patch_size}")


ALL_MODELS = ["unet", "swin"]


def _filter_models(specs, selected):
    """Keep only the model entries whose key is in `selected` (lowercase)."""
    return [s for s in specs if s[0] in selected]


def run_training(epochs, num_workers, patch_size, seeds, output_root, selected_models):
    print(f"\n>>> Training (epochs={epochs}, patch_size={patch_size}, models={selected_models})")

    train_data = os.path.join("data", "train", "flair")
    train_gts = os.path.join("data", "train", "gt")
    val_data = os.path.join("data", "dev_in", "flair")
    val_gts = os.path.join("data", "dev_in", "gt")

    models = _filter_models([
        ("unet", "src/train_unet.py", _exp_dir(output_root, "experiments_unet", patch_size)),
        ("swin", "src/train_swin.py", _exp_dir(output_root, "experiments_swin", patch_size)),
    ], selected_models)

    for model_name, script, save_base in models:
        os.makedirs(save_base, exist_ok=True)
        for seed in seeds:
            print(f"--- Training {model_name} | seed {seed} | patch {patch_size} ---")
            save_path = os.path.join(save_base, f"seed{seed}")
            os.makedirs(save_path, exist_ok=True)
            cmd = [
                sys.executable, script,
                "--seed", str(seed),
                "--n_epochs", str(epochs),
                "--path_train_data", train_data,
                "--path_train_gts", train_gts,
                "--path_val_data", val_data,
                "--path_val_gts", val_gts,
                "--path_save", save_path,
                "--num_workers", str(num_workers),
                "--patch_size", str(patch_size),
            ]
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                print(f"Training failed for {model_name} seed {seed} patch {patch_size}: {e}")
                sys.exit(1)


def run_evaluation(num_workers, patch_size, seeds, output_root, selected_models):
    print(f"\n>>> Evaluation (patch_size={patch_size}, models={selected_models})")

    test_data = os.path.join("data", "dev_out", "flair")
    test_gts = os.path.join("data", "dev_out", "gt")
    test_bm = os.path.join("data", "dev_out", "fg_mask")

    evals = _filter_models([
        ("unet", "src/test_unet.py", _exp_dir(output_root, "experiments_unet", patch_size)),
        ("swin", "src/test_swin.py", _exp_dir(output_root, "experiments_swin", patch_size)),
    ], selected_models)

    for model_name, script, model_dir in evals:
        print(f"--- Evaluating {model_name} | patch {patch_size} ---")
        cmd = [
            sys.executable, script,
            "--path_model", model_dir,
            "--path_data", test_data,
            "--path_gts", test_gts,
            "--path_bm", test_bm,
            "--threshold", "0.35",
            "--num_workers", str(num_workers),
            "--patch_size", str(patch_size),
            "--seeds",
        ] + [str(s) for s in seeds]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed for {model_name} patch {patch_size}: {e}")
            sys.exit(1)


def run_inference(num_workers, patch_size, seeds, output_root, selected_models):
    print(f"\n>>> Inference (patch_size={patch_size}, models={selected_models})")

    test_data = os.path.join("data", "eval_in", "flair")
    test_bm = os.path.join("data", "eval_in", "fg_mask")

    # tuple: (key, model_name_for_inference_script, model_dir, output_dir)
    configs = [
        ("unet", "UNet",      _exp_dir(output_root, "experiments_unet", patch_size), _exp_dir(output_root, "predictions_unet", patch_size)),
        ("swin", "SwinUNETR", _exp_dir(output_root, "experiments_swin", patch_size), _exp_dir(output_root, "predictions_swin", patch_size)),
    ]
    configs = [(name, mdir, odir) for key, name, mdir, odir in configs if key in selected_models]

    for model_name, model_dir, output_dir in configs:
        print(f"--- Generating segmentations for {model_name} | patch {patch_size} ---")
        cmd = [
            sys.executable, "src/inference.py",
            "--model_name", model_name,
            "--path_model", model_dir,
            "--path_data", test_data,
            "--path_bm", test_bm,
            "--path_pred", output_dir,
            "--num_models", str(len(seeds)),
            "--num_workers", str(num_workers),
            "--patch_size", str(patch_size),
        ]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Inference failed for {model_name} patch {patch_size}: {e}")
            sys.exit(1)


def run_audit(num_workers, patch_size, output_root, selected_models, skip_comparison):
    print(f"\n>>> Audit (patch_size={patch_size}, models={selected_models}, "
          f"skip_comparison={skip_comparison})")

    audit_data = os.path.join("data", "dev_out", "flair")
    audit_gts = os.path.join("data", "dev_out", "gt")
    audit_bm = os.path.join("data", "dev_out", "fg_mask")

    cmd = [
        sys.executable, "src/audit.py",
        "--path_data", audit_data,
        "--path_gts", audit_gts,
        "--path_bm", audit_bm,
        "--num_workers", str(num_workers),
        "--patch_size", str(patch_size),
        "--path_save", os.path.join(output_root, "visualization", f"p{patch_size}"),
    ]
    if "unet" in selected_models:
        cmd += ["--path_unet", _exp_dir(output_root, "experiments_unet", patch_size)]
    if "swin" in selected_models:
        cmd += ["--path_swin", _exp_dir(output_root, "experiments_swin", patch_size)]
    if skip_comparison:
        cmd += ["--no_comparison"]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Audit failed (patch {patch_size}): {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full MS Lesion Segmentation pipeline.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data-loader workers")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3], help="Ensemble seeds")
    parser.add_argument("--patch_sizes", nargs="+", type=int, default=[96],
                        help="Cubic training/inference patch sizes to sweep (single knob for "
                             "the Lego-3 locality comparison). For SwinUNETR, each value must "
                             "be a multiple of 32. Example: --patch_sizes 64 96 128")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
                        help="Which model(s) to train/eval/infer. Defaults to both. "
                             "Audit step requires BOTH models and is skipped otherwise. "
                             "Example: --models unet")
    parser.add_argument("--output_root", type=str, default=".",
                        help="Parent directory for all generated artifacts "
                             "(experiments_*, predictions_*, visualization/). "
                             "Defaults to the repo root. On RunPod, set this to "
                             "a single folder like 'download' so every output "
                             "can be fetched with one command.")
    parser.add_argument("--skip_install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip_train", action="store_true", help="Skip training phase")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation phase")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference phase")
    parser.add_argument("--skip_audit", action="store_true", help="Skip audit phase entirely")
    parser.add_argument("--skip_comparison", action="store_true",
                        help="Run the audit but skip the secondary cross-model "
                             "FP-overlap comparison (UNet vs Swin). The primary "
                             "per-backbone uncertainty audit still runs.")
    args = parser.parse_args()

    if not args.skip_install:
        install_requirements()

    os.makedirs(args.output_root, exist_ok=True)
    print(f"[run.py] All outputs will be written under: {os.path.abspath(args.output_root)}")

    for ps in args.patch_sizes:
        if ps % 32 != 0:
            print(f"[run.py] WARNING: patch_size={ps} is not a multiple of 32; "
                  f"SwinUNETR will refuse to build. Skipping.")
            continue
        print(f"\n{'='*60}\n=== PATCH SIZE {ps} ===\n{'='*60}")

        if not args.skip_train:
            run_training(args.epochs, args.num_workers, ps, args.seeds, args.output_root, args.models)
        if not args.skip_eval:
            run_evaluation(args.num_workers, ps, args.seeds, args.output_root, args.models)
        if not args.skip_inference:
            run_inference(args.num_workers, ps, args.seeds, args.output_root, args.models)
        if not args.skip_audit:
            run_audit(args.num_workers, ps, args.output_root, args.models, args.skip_comparison)
