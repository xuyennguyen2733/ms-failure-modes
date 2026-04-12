#!/bin/bash
#chmod +x run.py
#apt update -y && apt install -y python3.9 python3.9-venv

import argparse
import os
import subprocess
import sys


def _detect_gpu_count():
    """Best-effort detection of available CUDA devices.
    Returns 0 if torch is unavailable or CUDA is not present, so the rest of
    the pipeline can transparently fall back to single-process execution."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _launch_per_model_parallel(jobs, gpu_ids, label):
    """Launch per-model jobs concurrently, pinning each to one GPU.

    Parameters
    ----------
    jobs : list[tuple[str, list[str]]]
        List of (job_name, command) pairs. Each `command` is the argv list
        passed to subprocess; we do NOT add CUDA flags to it — we instead
        scope GPU visibility via the CUDA_VISIBLE_DEVICES environment
        variable so the child sees only "its" GPU as cuda:0 and no code
        change is needed inside the train/test/inference scripts.
    gpu_ids : list[int] | None
        Which GPU IDs are available. If None or shorter than `jobs`, jobs
        are run sequentially on the available GPU(s) — never bypassing the
        single-GPU machine assumption.
    label : str
        Stage label used in error messages.
    """
    if not jobs:
        return

    # Decide a launch plan: how many we can fan out at once.
    n_jobs = len(jobs)
    n_gpus = len(gpu_ids) if gpu_ids else 0

    if n_gpus >= n_jobs and n_jobs > 1:
        print(f"[run.py] {label}: launching {n_jobs} jobs in parallel "
              f"across GPUs {gpu_ids[:n_jobs]}")
        procs = []
        for (name, cmd), gpu in zip(jobs, gpu_ids[:n_jobs]):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[run.py]   -> [{name}] on GPU {gpu}")
            procs.append((name, gpu, subprocess.Popen(cmd, env=env)))
        # Wait for all and collect failures
        failed = []
        for name, gpu, p in procs:
            ret = p.wait()
            if ret != 0:
                failed.append((name, gpu, ret))
        if failed:
            for name, gpu, ret in failed:
                print(f"[run.py] {label} FAILED for {name} on GPU {gpu} (rc={ret})")
            sys.exit(1)
    else:
        # Sequential fallback. Either single GPU, single job, or no GPUs.
        # When 1 GPU is available, pin every job to it for consistency;
        # when 0 GPUs, leave the env alone and let the script run on CPU.
        for (name, cmd) in jobs:
            env = os.environ.copy()
            if n_gpus >= 1:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
                print(f"[run.py] {label}: [{name}] on GPU {gpu_ids[0]} (sequential)")
            else:
                print(f"[run.py] {label}: [{name}] on CPU (sequential, no CUDA)")
            ret = subprocess.call(cmd, env=env)
            if ret != 0:
                print(f"[run.py] {label} FAILED for {name} (rc={ret})")
                sys.exit(1)


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


def _build_train_cmd(script, seed, epochs, paths, save_path, num_workers, patch_size):
    train_data, train_gts, val_data, val_gts = paths
    return [
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


def run_training(epochs, num_workers, patch_size, seeds, output_root,
                 selected_models, gpu_ids):
    print(f"\n>>> Training (epochs={epochs}, patch_size={patch_size}, models={selected_models})")

    paths = (
        os.path.join("data", "train", "flair"),
        os.path.join("data", "train", "gt"),
        os.path.join("data", "dev_in", "flair"),
        os.path.join("data", "dev_in", "gt"),
    )

    models = _filter_models([
        ("unet", "src/train_unet.py", _exp_dir(output_root, "experiments_unet", patch_size)),
        ("swin", "src/train_swin.py", _exp_dir(output_root, "experiments_swin", patch_size)),
    ], selected_models)

    for model_name, script, save_base in models:
        os.makedirs(save_base, exist_ok=True)
        for seed in seeds:
            os.makedirs(os.path.join(save_base, f"seed{seed}"), exist_ok=True)

    # Parallelize per-model at each seed: at seed S, unet and swin run
    # simultaneously on different GPUs. Seeds are still sequential so that
    # RAM footprint stays bounded. This mirrors how the user described it:
    # "everything that can be done separately for each model should also be
    # run separately on 2 different GPUs if possible."
    for seed in seeds:
        jobs = []
        for model_name, script, save_base in models:
            save_path = os.path.join(save_base, f"seed{seed}")
            jobs.append((
                f"train/{model_name}/seed{seed}/p{patch_size}",
                _build_train_cmd(script, seed, epochs, paths, save_path, num_workers, patch_size),
            ))
        _launch_per_model_parallel(jobs, gpu_ids, f"Training seed {seed}")


def run_evaluation(num_workers, patch_size, seeds, output_root,
                   selected_models, gpu_ids):
    print(f"\n>>> Evaluation (patch_size={patch_size}, models={selected_models})")

    test_data = os.path.join("data", "dev_out", "flair")
    test_gts = os.path.join("data", "dev_out", "gt")
    test_bm = os.path.join("data", "dev_out", "fg_mask")

    evals = _filter_models([
        ("unet", "src/test_unet.py", _exp_dir(output_root, "experiments_unet", patch_size)),
        ("swin", "src/test_swin.py", _exp_dir(output_root, "experiments_swin", patch_size)),
    ], selected_models)

    jobs = []
    for model_name, script, model_dir in evals:
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
        jobs.append((f"eval/{model_name}/p{patch_size}", cmd))
    _launch_per_model_parallel(jobs, gpu_ids, f"Evaluation p={patch_size}")


def run_inference(num_workers, patch_size, seeds, output_root,
                  selected_models, gpu_ids):
    print(f"\n>>> Inference (patch_size={patch_size}, models={selected_models})")

    test_data = os.path.join("data", "eval_in", "flair")
    test_bm = os.path.join("data", "eval_in", "fg_mask")

    # tuple: (key, model_name_for_inference_script, model_dir, output_dir)
    configs = [
        ("unet", "UNet",      _exp_dir(output_root, "experiments_unet", patch_size), _exp_dir(output_root, "predictions_unet", patch_size)),
        ("swin", "SwinUNETR", _exp_dir(output_root, "experiments_swin", patch_size), _exp_dir(output_root, "predictions_swin", patch_size)),
    ]
    configs = [(key, name, mdir, odir) for key, name, mdir, odir in configs
               if key in selected_models]

    jobs = []
    for key, model_name, model_dir, output_dir in configs:
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
        jobs.append((f"infer/{key}/p{patch_size}", cmd))
    _launch_per_model_parallel(jobs, gpu_ids, f"Inference p={patch_size}")


def run_audit(num_workers, patch_size, output_root, selected_models,
              skip_comparison, gpu_ids):
    """Audit is intentionally single-process — when the joined cross-model
    comparison runs, both ensembles have to live in the same Python process.
    We pin the whole thing to ONE GPU (the first available) so the comparison
    can never accidentally span two devices."""
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

    env = os.environ.copy()
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        print(f"[run.py] Audit pinned to GPU {gpu_ids[0]} (single-GPU by design)")

    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Audit failed (patch {patch_size}): {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full MS Lesion Segmentation pipeline.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data-loader workers")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3], help="Ensemble seeds")
    parser.add_argument("--patch_sizes", nargs="+", type=int, default=[96],
                        help="Cubic training/inference patch sizes to sweep")
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
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=None,
                        help="Explicit list of GPU IDs to use (e.g. --gpu_ids 0 1). "
                             "If omitted, all visible GPUs are auto-detected via "
                             "torch.cuda.device_count(). Per-model stages "
                             "(train/eval/infer) fan out one model per GPU when "
                             "2+ GPUs and 2+ models are present; the audit always "
                             "runs on a single GPU.")
    args = parser.parse_args()

    if not args.skip_install:
        install_requirements()

    os.makedirs(args.output_root, exist_ok=True)
    print(f"[run.py] All outputs will be written under: {os.path.abspath(args.output_root)}")

    # GPU plan. Explicit --gpu_ids always wins; otherwise auto-detect.
    if args.gpu_ids is not None:
        gpu_ids = list(args.gpu_ids)
    else:
        n_gpus = _detect_gpu_count()
        gpu_ids = list(range(n_gpus)) if n_gpus > 0 else []
    if gpu_ids:
        print(f"[run.py] Detected/selected GPUs: {gpu_ids} "
              f"(parallel-per-model = {len(gpu_ids) >= 2 and len(args.models) >= 2})")
    else:
        print("[run.py] No CUDA GPUs visible — falling back to CPU / sequential execution.")

    for ps in args.patch_sizes:
        if ps % 32 != 0:
            print(f"[run.py] WARNING: patch_size={ps} is not a multiple of 32; "
                  f"SwinUNETR will refuse to build. Skipping.")
            continue
        print(f"\n{'='*60}\n=== PATCH SIZE {ps} ===\n{'='*60}")

        if not args.skip_train:
            run_training(args.epochs, args.num_workers, ps, args.seeds,
                         args.output_root, args.models, gpu_ids)
        if not args.skip_eval:
            run_evaluation(args.num_workers, ps, args.seeds,
                           args.output_root, args.models, gpu_ids)
        if not args.skip_inference:
            run_inference(args.num_workers, ps, args.seeds,
                          args.output_root, args.models, gpu_ids)
        if not args.skip_audit:
            run_audit(args.num_workers, ps, args.output_root, args.models,
                      args.skip_comparison, gpu_ids)
