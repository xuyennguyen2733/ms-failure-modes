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


# Exit code that run.py uses for KeyboardInterrupt (POSIX SIGINT convention).
# run_dev.py treats this as "user abort" and refuses to stop the pod, so a
# Ctrl+C inside the tmux session never costs you the pod by accident.
RUNPY_KEYBOARD_INTERRUPT_RC = 130


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
        "patch_sizes": [64, 96, 128],
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


def build_run_cmd(mode_cfg, output_root):
    """Build the argv for `python run.py ...` from a preset dict.

    Pod-stop logic lives in run_dev.py (this file) — run.py is just a
    pipeline runner now and knows nothing about RunPod.
    """
    return [
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


def stop_pod_now(reason):
    """Best-effort RunPod shutdown via runpodctl. Silently no-ops outside
    of RunPod (no RUNPOD_POD_ID env var) so this is safe to call locally.

    Called from run_dev.py *after* run.py has fully exited — never from
    inside the pipeline, so we know every stage has had its chance to
    write artifacts to disk before the pod disappears.
    """
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        print(f"[run_dev] stop_pod requested ({reason}) but RUNPOD_POD_ID not set — skipping.")
        return
    try:
        print(f"[run_dev] stop_pod ({reason}): runpodctl stop pod {pod_id}")
        subprocess.call(["runpodctl", "stop", "pod", pod_id])
    except FileNotFoundError:
        print(f"[run_dev] runpodctl not on PATH; cannot stop pod {pod_id}.")
    except Exception as e:
        print(f"[run_dev] Failed to stop pod {pod_id}: {e}")


def _build_tmux_bash_snippet(run_py_cmd, stop_pod):
    """Wrap the `python run.py ...` command in a small bash snippet that:
      1. Runs run.py and captures its exit code.
      2. Prints a summary line (so the tmux scrollback is self-contained).
      3. If `stop_pod` is True AND run.py did not exit with 130 (Ctrl+C),
         calls `runpodctl stop pod $RUNPOD_POD_ID`.

    The snippet runs ENTIRELY inside the tmux session, so even if the user
    detaches and runpod nukes the run_dev.py parent, the cleanup still
    happens after the pipeline genuinely finishes.
    """
    inner = " ".join(shlex.quote(c) for c in run_py_cmd)
    if not stop_pod:
        # Just run the pipeline and exit — no cleanup, no pod-stop.
        return inner

    # Use $RUNPOD_POD_ID lazily inside the tmux session — it is set by
    # RunPod automatically. If runpodctl is missing or the var is unset we
    # print a notice instead of failing.
    return (
        f'{inner}; '
        f'rc=$?; '
        f'echo ""; echo "[run_dev/tmux] run.py exited with rc=$rc"; '
        f'if [ "$rc" -eq {RUNPY_KEYBOARD_INTERRUPT_RC} ]; then '
        f'  echo "[run_dev/tmux] Ctrl+C detected — leaving pod up."; '
        f'elif [ -z "${{RUNPOD_POD_ID:-}}" ]; then '
        f'  echo "[run_dev/tmux] RUNPOD_POD_ID unset — not on RunPod, leaving pod up."; '
        f'elif ! command -v runpodctl >/dev/null 2>&1; then '
        f'  echo "[run_dev/tmux] runpodctl not on PATH — cannot stop pod."; '
        f'else '
        f'  echo "[run_dev/tmux] Stopping pod $RUNPOD_POD_ID (rc=$rc)..."; '
        f'  runpodctl stop pod "$RUNPOD_POD_ID"; '
        f'fi'
    )


def launch_in_tmux(session_name, cmd, stop_pod):
    """Start a detached tmux session and run `cmd` inside it.

    If `stop_pod` is True, the tmux session also runs `runpodctl stop pod`
    once `cmd` finishes — but only if `cmd` did NOT exit on Ctrl+C (130).
    The pod-stop is part of the tmux command itself, NOT executed by
    run_dev.py, so it survives run_dev.py exiting.
    """
    if subprocess.call(["which", "tmux"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL) != 0:
        print("[run_dev] ERROR: tmux is not installed on this machine.")
        sys.exit(1)

    rc = subprocess.call(["tmux", "has-session", "-t", session_name],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc == 0:
        print(f"[run_dev] ERROR: tmux session '{session_name}' already exists. "
              f"Attach with `tmux attach -t {session_name}` or pick a new name.")
        sys.exit(1)

    snippet = _build_tmux_bash_snippet(cmd, stop_pod)
    # Force a real bash so set -u-style guards and `command -v` work.
    bash_cmd = ["bash", "-c", snippet]

    print(f"[run_dev] Launching tmux session '{session_name}' with:")
    print(f"          {snippet}")
    subprocess.check_call(["tmux", "new-session", "-d", "-s", session_name, *bash_cmd])
    print(f"[run_dev] Session started. Attach with: tmux attach -t {session_name}")
    print(f"[run_dev] Detach from inside the session with: Ctrl+b then d")
    if stop_pod:
        print(f"[run_dev] Pod will auto-stop after run.py exits "
              f"(unless aborted with Ctrl+C).")


def run_inline(cmd, stop_pod):
    """Run cmd in the current shell, propagating SIGINT to run.py.

    On normal completion (rc != 130), optionally calls runpodctl stop pod.
    On Ctrl+C, leaves the pod up.
    """
    quoted = " ".join(shlex.quote(c) for c in cmd)
    print(f"[run_dev] Running inline:\n          {quoted}\n")
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\n[run_dev] KeyboardInterrupt forwarded to run.py.")
        return RUNPY_KEYBOARD_INTERRUPT_RC

    print(f"\n[run_dev] run.py exited with rc={rc}")
    if stop_pod:
        if rc == RUNPY_KEYBOARD_INTERRUPT_RC:
            print("[run_dev] Ctrl+C detected — leaving pod up.")
        else:
            stop_pod_now(reason=f"run.py rc={rc}")
    return rc


def main():
    parser = argparse.ArgumentParser(
        description="Developer wrapper around run.py with named presets.")
    parser.add_argument("--mode", choices=list(MODES.keys()), default="dummy",
                        help="Preset training/eval scale (default: dummy).")
    parser.add_argument("--run-on-tmux", dest="tmux_session", metavar="NAME",
                        default=None,
                        help="If set, launches the run inside a detached tmux "
                             "session with this name. Implicitly enables "
                             "--stop_pod (the whole point of tmux mode is "
                             "long-running RunPod jobs).")
    parser.add_argument("--stop_pod", action="store_true",
                        help="After run.py finishes, call `runpodctl stop pod "
                             "$RUNPOD_POD_ID`. Auto-on when --run-on-tmux is set. "
                             "Skipped on Ctrl+C so an interactive interrupt never "
                             "costs you the pod by accident.")
    parser.add_argument("--output_root", default="download",
                        help="Output root passed through to run.py "
                             "(default: download — single-folder for RunPod).")
    args = parser.parse_args()

    cfg = MODES[args.mode]
    print(f"[run_dev] mode={args.mode}  config={cfg}")

    # Tmux mode implies stop_pod (otherwise the pod sits idle after the
    # detached run finishes — exactly what we're trying to avoid).
    stop_pod = args.stop_pod or bool(args.tmux_session)

    cmd = build_run_cmd(cfg, output_root=args.output_root)

    if args.tmux_session:
        launch_in_tmux(args.tmux_session, cmd, stop_pod=stop_pod)
        return 0
    else:
        return run_inline(cmd, stop_pod=stop_pod)


if __name__ == "__main__":
    sys.exit(main())
