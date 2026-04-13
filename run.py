#!/bin/bash
#chmod +x run.py
#apt update -y && apt install -y python3.9 python3.9-venv

import argparse
import os
import subprocess
import sys


def _stop_runpod_if_requested(reason):
    """Best-effort RunPod shutdown. Only called from the outer try/finally
    block — and only when the reason is 'normal finish' or a non-interactive
    crash. We specifically do NOT call this on KeyboardInterrupt so the user
    can Ctrl+C into the tmux session without losing their pod.

    Requires `runpodctl` on PATH and the `RUNPOD_POD_ID` env var (RunPod
    sets this automatically). Silently no-ops if either is missing (e.g. on
    a local machine) so this is always safe to call.
    """
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        print(f"[run.py] stop_pod requested ({reason}) but RUNPOD_POD_ID not set — skipping.")
        return
    runpodctl = "runpodctl"
    try:
        print(f"[run.py] stop_pod ({reason}): calling `{runpodctl} stop pod {pod_id}`")
        subprocess.call([runpodctl, "stop", "pod", pod_id])
    except FileNotFoundError:
        print(f"[run.py] `{runpodctl}` not found on PATH; cannot stop pod {pod_id}.")
    except Exception as e:
        print(f"[run.py] Failed to stop pod {pod_id}: {e}")


def _detect_gpu_count():
    """Best-effort detection of available CUDA devices.
    Returns 0 if torch is unavailable or CUDA is not present, so the rest of
    the pipeline can transparently fall back to single-process execution."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _launch_parallel(jobs, gpu_ids, label):
    """Schedule N jobs across M GPUs using a simple wave scheduler.

    Each wave launches `min(pending, M)` subprocesses, one per free GPU.
    When all subprocesses in a wave finish, the next wave starts. This
    keeps GPU 1 busy even when only one model is being investigated
    (by fanning out *seeds* across GPUs) — a strict generalization of
    the old "one job per GPU, single shot" launcher.

    jobs : list[tuple[str, list[str]]]
        (job_name, argv) pairs. CUDA_VISIBLE_DEVICES is set per-process
        so each child sees its assigned GPU as cuda:0. No code change is
        needed in train/test/inference scripts.
    gpu_ids : list[int] | None
        Available GPU IDs. None/empty -> CPU fallback, fully sequential.
    label : str
        Stage label used in log messages.
    """
    if not jobs:
        return
    n_gpus = len(gpu_ids) if gpu_ids else 0

    # CPU fallback — sequential, no env tweaking.
    if n_gpus == 0:
        for name, cmd in jobs:
            print(f"[run.py] {label}: [{name}] on CPU (sequential, no CUDA)")
            ret = subprocess.call(cmd)
            if ret != 0:
                print(f"[run.py] {label} FAILED for {name} (rc={ret})")
                sys.exit(1)
        return

    width = min(n_gpus, len(jobs))
    print(f"[run.py] {label}: scheduling {len(jobs)} jobs across "
          f"{n_gpus} GPU(s) (wave width={width})")

    failed = []
    for wave_start in range(0, len(jobs), width):
        wave = jobs[wave_start : wave_start + width]
        procs = []
        for i, (name, cmd) in enumerate(wave):
            gpu = gpu_ids[i % n_gpus]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[run.py]   -> [{name}] on GPU {gpu}")
            procs.append((name, gpu, subprocess.Popen(cmd, env=env)))
        for name, gpu, p in procs:
            ret = p.wait()
            if ret != 0:
                failed.append((name, gpu, ret))

    if failed:
        for name, gpu, ret in failed:
            print(f"[run.py] {label} FAILED for {name} on GPU {gpu} (rc={ret})")
        sys.exit(1)


# Backwards-compat alias so existing call sites keep working.
_launch_per_model_parallel = _launch_parallel


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

    # Build one flat (model × seed) job list and let the wave scheduler fan
    # it out across all available GPUs. This means:
    #   - both models + 2 GPUs → UNet seed1 and Swin seed1 run in parallel
    #   - single model + 2 GPUs → seed1 and seed2 run in parallel (then seed3 solo)
    # Both cases keep *every* GPU busy whenever there's more than one job
    # left, regardless of whether the axis filling the second GPU is model
    # or seed.
    jobs = []
    for model_name, script, save_base in models:
        for seed in seeds:
            save_path = os.path.join(save_base, f"seed{seed}")
            jobs.append((
                f"train/{model_name}/seed{seed}/p{patch_size}",
                _build_train_cmd(script, seed, epochs, paths, save_path, num_workers, patch_size),
            ))
    _launch_parallel(jobs, gpu_ids, f"Training p={patch_size}")


def run_evaluation(num_workers, patch_size, seeds, output_root,
                   selected_models, gpu_ids, epochs, sw_batch_size, n_jobs):
    print(f"\n>>> Evaluation (patch_size={patch_size}, models={selected_models})")

    test_data = os.path.join("data", "dev_out", "flair")
    test_gts = os.path.join("data", "dev_out", "gt")
    test_bm = os.path.join("data", "dev_out", "fg_mask")

    log_dir = os.path.join(output_root, "eval_reports")
    os.makedirs(log_dir, exist_ok=True)

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
            "--sw_batch_size", str(sw_batch_size),
            "--n_jobs", str(n_jobs),
            "--log_dir", log_dir,
            "--model_label", f"{model_name}_p{patch_size}",
            "--train_epochs", str(epochs),
            "--seeds",
        ] + [str(s) for s in seeds]
        jobs.append((f"eval/{model_name}/p{patch_size}", cmd))
    _launch_per_model_parallel(jobs, gpu_ids, f"Evaluation p={patch_size}")


def run_inference(num_workers, patch_size, seeds, output_root,
                  selected_models, gpu_ids, sw_batch_size):
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
            "--sw_batch_size", str(sw_batch_size),
        ]
        jobs.append((f"infer/{key}/p{patch_size}", cmd))
    _launch_per_model_parallel(jobs, gpu_ids, f"Inference p={patch_size}")


def run_retention(num_workers, patch_size, seeds, output_root,
                  selected_models, gpu_ids, sw_batch_size, n_jobs):
    """Generate nDSC retention curves per backbone. Parallelized across GPUs
    the same way as training/eval — one backbone per GPU when possible."""
    print(f"\n>>> Retention curves (patch_size={patch_size}, models={selected_models})")

    rc_data = os.path.join("data", "dev_out", "flair")
    rc_gts = os.path.join("data", "dev_out", "gt")
    rc_bm = os.path.join("data", "dev_out", "fg_mask")

    save_dir = os.path.join(output_root, "retention_curves", f"p{patch_size}")
    os.makedirs(save_dir, exist_ok=True)

    configs = [
        ("unet", "UNet",      _exp_dir(output_root, "experiments_unet", patch_size)),
        ("swin", "SwinUNETR", _exp_dir(output_root, "experiments_swin", patch_size)),
    ]
    configs = [(key, name, mdir) for key, name, mdir in configs if key in selected_models]

    jobs = []
    for key, model_name, model_dir in configs:
        cmd = [
            sys.executable, "src/retention_curves.py",
            "--model_name", model_name,
            "--path_model", model_dir,
            "--path_data", rc_data,
            "--path_gts", rc_gts,
            "--path_bm", rc_bm,
            "--num_models", str(len(seeds)),
            "--num_workers", str(num_workers),
            "--n_jobs", str(n_jobs),
            "--patch_size", str(patch_size),
            "--sw_batch_size", str(sw_batch_size),
            "--path_save", save_dir,
            "--curve_label", f"{key}_p{patch_size}",
        ]
        jobs.append((f"retention/{key}/p{patch_size}", cmd))
    _launch_per_model_parallel(jobs, gpu_ids, f"Retention curves p={patch_size}")


def run_audit(num_workers, patch_size, output_root, selected_models,
              skip_comparison, gpu_ids, sw_batch_size):
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
        "--sw_batch_size", str(sw_batch_size),
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
                             "Defaults to the repo root.")
    parser.add_argument("--skip_install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip_train", action="store_true", help="Skip training phase")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation phase")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference phase")
    parser.add_argument("--skip_retention", action="store_true",
                        help="Skip retention-curve generation (Stage 5).")
    parser.add_argument("--skip_aggregate", action="store_true",
                        help="Skip the post-sweep aggregator (Stage 6) that "
                             "collates eval logs, audit JSON, and retention curves "
                             "into <output_root>/aggregated/.")
    parser.add_argument("--skip_audit", action="store_true", help="Skip audit phase entirely")
    parser.add_argument("--skip_comparison", action="store_true",
                        help="Run the audit but skip the secondary cross-model "
                             "FP-overlap comparison (UNet vs Swin). The primary "
                             "per-backbone uncertainty audit still runs.")
    parser.add_argument("--sw_batch_size", type=int, default=4,
                        help="Sliding-window batch size used at eval/inference/"
                             "audit time (patches per forward pass). Higher is "
                             "faster if GPU memory allows. Does not affect training.")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of CPU workers for lesion-F1 and nDSC R-AUC "
                             "metric computation (joblib).")
    parser.add_argument("--stop_pod_on_finish", action="store_true",
                        help="If set AND RUNPOD_POD_ID is present in the environment, "
                             "call `runpodctl stop pod` after the pipeline finishes "
                             "(either normally or via an uncaught exception). "
                             "Ctrl+C (KeyboardInterrupt) is EXCLUDED — if you interrupt "
                             "the run yourself, the pod is left running so you don't "
                             "get logged out of your tmux session.")
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

    # Run the full pipeline guarded so we can decide whether to stop the pod
    # at the end. Ctrl+C is EXPLICITLY excluded from the stop-pod path so an
    # interactive interruption (e.g. inside tmux on RunPod) keeps the pod up.
    pipeline_ok = False
    crash_reason = None
    try:
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
                               args.output_root, args.models, gpu_ids, args.epochs,
                               args.sw_batch_size, args.n_jobs)
            if not args.skip_inference:
                run_inference(args.num_workers, ps, args.seeds,
                              args.output_root, args.models, gpu_ids,
                              args.sw_batch_size)
            if not args.skip_retention:
                run_retention(args.num_workers, ps, args.seeds,
                              args.output_root, args.models, gpu_ids,
                              args.sw_batch_size, args.n_jobs)
            if not args.skip_audit:
                run_audit(args.num_workers, ps, args.output_root, args.models,
                          args.skip_comparison, gpu_ids, args.sw_batch_size)

        # Stage 6: aggregate across the entire sweep. Runs once, after all
        # patch sizes are processed. Non-fatal — failures only print a warning.
        if not args.skip_aggregate:
            print(f"\n>>> Aggregating results under {args.output_root}/aggregated/")
            try:
                subprocess.check_call([
                    sys.executable, "src/aggregate_results.py",
                    "--output_root", args.output_root,
                ])
            except subprocess.CalledProcessError as e:
                print(f"[run.py] Aggregation step failed (non-fatal): {e}")

        pipeline_ok = True
    except KeyboardInterrupt:
        # User pressed Ctrl+C. Keep the pod up no matter what.
        print("\n[run.py] KeyboardInterrupt received — pipeline aborted by user. "
              "Pod will NOT be stopped even if --stop_pod_on_finish was set.")
        sys.exit(130)
    except SystemExit:
        # A child subprocess failed and one of run_* called sys.exit(1).
        # Treat that as a crash for stop-pod purposes — we want to release
        # the pod rather than burn money on a failed long run.
        crash_reason = "stage failed (SystemExit from subprocess)"
        raise
    except Exception as e:
        crash_reason = f"uncaught exception: {type(e).__name__}: {e}"
        raise
    finally:
        if args.stop_pod_on_finish:
            if pipeline_ok:
                _stop_runpod_if_requested("normal finish")
            elif crash_reason is not None:
                _stop_runpod_if_requested(f"crash — {crash_reason}")
