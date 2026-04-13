"""
Developer entry point for run.py.

Wraps `python run.py ...` with three named presets and an optional tmux
launcher so you can fire off a long RunPod job without having to remember
the full flag set every time.

Modes
-----
- dummy : 1 epoch, 1 seed, 1 patch size, single model. Fastest possible
          end-to-end run — used for smoke tests / sanity checks.
- lite  : 50 epochs, 2 seeds, 2 patch sizes, both models. Mid-length run
          for iterating on hypotheses without paying for a full sweep.
- full  : 150 epochs, 3 seeds, 3 patch sizes, both models. 

Tmux
----
If --run-on-tmux NAME is given, the wrapper starts a *detached* tmux session
named NAME and runs run.py inside it. The pipeline gets --stop_pod_on_finish
automatically so the pod releases itself when the run completes (or crashes
on an unrecoverable stage). Ctrl+C inside the tmux session is still safe —
run.py's KeyboardInterrupt handler skips the pod-stop path.
"""

import argparse
import os
import shlex
import subprocess
import sys


# Preset configurations. Keep these in sync with the run.py CLI.
MODES = {
    # UNet-only locality sweep. For testing the full pipeline with a single run per model, patch size, and seed.
    "dummy": {
        "epochs":      1,
        "seeds":       [1],
        "patch_sizes": [96],
        "models":      ["unet"],          # single model -> audit comparison auto-skipped
        "num_workers": 1,
        "n_jobs":      1,
        "sw_batch_size": 4,
    },
    # UNet-only locality sweep.
    "lite": {
        "epochs":      50,
        "seeds":       [1, 2],
        "patch_sizes": [64, 96],
        "models":      ["unet"],
        "num_workers": 4,
        "n_jobs":      4,
        "sw_batch_size": 4,
    },
    # UNet-only locality sweep.
    "full": {
        "epochs":      150,
        "seeds":       [1, 2, 3],
        "patch_sizes": [64, 96, 128],
        "models":      ["unet"],
        "num_workers": 8,
        "n_jobs":      8,
        "sw_batch_size": 8,
    },
    # Swin-only locality sweep. 
    "swin": {
        "epochs":      150,
        "seeds":       [1, 2, 3],
        "patch_sizes": [64, 128],
        "models":      ["swin"],
        "num_workers": 8,
        "n_jobs":      8,
        "sw_batch_size": 8,
    },
    # Both models.
    "extra": {
        "epochs":      150,
        "seeds":       [1, 2],
        "patch_sizes": [64, 128],
        "models":      ["unet", "swin"],
        "num_workers": 8,
        "n_jobs":      8,
        "sw_batch_size": 8,
    },
}


def build_run_cmd(mode_cfg, stop_pod_on_finish, output_root):
    """Build the argv for `python run.py ...` from a preset dict."""
    cmd = [
        sys.executable, "run.py",
        "--epochs", str(mode_cfg["epochs"]),
        "--seeds", *[str(s) for s in mode_cfg["seeds"]],
        "--patch_sizes", *[str(p) for p in mode_cfg["patch_sizes"]],
        "--models", *mode_cfg["models"],
        "--num_workers", str(mode_cfg["num_workers"]),
        "--n_jobs", str(mode_cfg["n_jobs"]),
        "--sw_batch_size", str(mode_cfg["sw_batch_size"]),
        "--output_root", output_root,
        "--skip_install",  # devs install once; don't waste time on every dev run
    ]
    if stop_pod_on_finish:
        cmd.append("--stop_pod_on_finish")
    return cmd


def launch_in_tmux(session_name, cmd):
    """Start a detached tmux session and run `cmd` inside it."""
    if subprocess.call(["which", "tmux"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL) != 0:
        print("[run_dev] ERROR: tmux is not installed on this machine.")
        sys.exit(1)

    # If the session already exists, refuse to clobber it.
    rc = subprocess.call(["tmux", "has-session", "-t", session_name],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc == 0:
        print(f"[run_dev] ERROR: tmux session '{session_name}' already exists. "
              f"Attach with `tmux attach -t {session_name}` or pick a new name.")
        sys.exit(1)

    quoted = " ".join(shlex.quote(c) for c in cmd)
    print(f"[run_dev] Launching tmux session '{session_name}' with:")
    print(f"          {quoted}")
    subprocess.check_call(["tmux", "new-session", "-d", "-s", session_name, quoted])
    print(f"[run_dev] Session started. Attach with: tmux attach -t {session_name}")
    print(f"[run_dev] Detach from inside the session with: Ctrl+b then d")


def run_inline(cmd):
    """Run cmd in the current shell, propagating SIGINT to run.py."""
    quoted = " ".join(shlex.quote(c) for c in cmd)
    print(f"[run_dev] Running inline:\n          {quoted}\n")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\n[run_dev] KeyboardInterrupt forwarded to run.py.")
        return 130


def main():
    parser = argparse.ArgumentParser(
        description="Developer wrapper around run.py with named presets.")
    parser.add_argument("--mode", choices=list(MODES.keys()), default="dummy",
                        help="Preset training/eval scale (default: dummy).")
    parser.add_argument("--run-on-tmux", dest="tmux_session", metavar="NAME",
                        default=None,
                        help="If set, launches the run inside a detached tmux "
                             "session with this name and adds --stop_pod_on_finish "
                             "to release the RunPod when done.")
    parser.add_argument("--output_root", default="download",
                        help="Output root passed through to run.py "
                             "(default: download — single-folder for RunPod).")
    args = parser.parse_args()

    cfg = MODES[args.mode]
    print(f"[run_dev] mode={args.mode}  config={cfg}")

    cmd = build_run_cmd(cfg,
                        stop_pod_on_finish=bool(args.tmux_session),
                        output_root=args.output_root)

    if args.tmux_session:
        launch_in_tmux(args.tmux_session, cmd)
        return 0
    else:
        return run_inline(cmd)


if __name__ == "__main__":
    sys.exit(main())
