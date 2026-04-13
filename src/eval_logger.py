"""
Incremental evaluation logger.

Writes a plain-text report to `eval_reports/<model>_eval_log_<MMDDYYYY>_<HHMMSS>.txt`
while evaluation is running. The file is line-buffered and flushed after every
write so partial results survive a SIGINT / OOM / pod termination.
"""

import os
import sys
from datetime import datetime


class EvalLogger:
    def __init__(self, model_label, log_dir="eval_reports", also_stdout=True):
        self.model_label = model_label
        self.also_stdout = also_stdout

        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%m%d%Y_%H%M%S")
        self.path = os.path.join(log_dir, f"{model_label}_eval_log_{ts}.txt")

        # Line-buffered (buffering=1) so each print flushes to disk immediately.
        self._fh = open(self.path, "w", buffering=1)
        self.write_line(f"=== Evaluation log: {model_label} ===")
        self.write_line(f"Started at: {datetime.now().isoformat(timespec='seconds')}")
        self.write_line(f"Log path:   {os.path.abspath(self.path)}")
        self.write_line("")

    # ---------- Primitive writers ----------
    def write_line(self, s=""):
        self._fh.write(s + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())  # best-effort durability for pod restarts
        except (OSError, ValueError):
            pass
        if self.also_stdout:
            print(s, flush=True)

    # ---------- Sections ----------
    def section(self, title):
        self.write_line("")
        self.write_line("=" * 60)
        self.write_line(title)
        self.write_line("=" * 60)

    def config(self, config_dict):
        """Render a key-value config block."""
        self.section("Model Configuration")
        width = max(len(k) for k in config_dict.keys())
        for k, v in config_dict.items():
            self.write_line(f"  {k.ljust(width)} : {v}")

    def per_subject_header(self, metric_names):
        self.section("Per-Subject Metrics")
        cols = ["idx"] + list(metric_names)
        self.write_line("  " + "  ".join(c.rjust(14) for c in cols))

    def per_subject_row(self, idx, values):
        """values: list of floats, same order as metric_names passed to header."""
        row = [str(idx)] + [f"{v:.4f}" for v in values]
        self.write_line("  " + "  ".join(c.rjust(14) for c in row))

    def summary(self, summary_rows):
        """summary_rows: list of (metric_name, mean, std) tuples."""
        self.section("Final Summary")
        name_w = max(len(r[0]) for r in summary_rows)
        for name, mean, std in summary_rows:
            self.write_line(f"  {name.ljust(name_w)} : {mean:>10.4f}  +/-  {std:.4f}")
        self.write_line("")
        self.write_line(f"Finished at: {datetime.now().isoformat(timespec='seconds')}")

    def error(self, msg):
        self.write_line("")
        self.write_line(f"!!! ERROR: {msg}")

    def close(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
