# -*- coding: utf-8 -*-
"""V17 Test 集 global hard Dice：调用 evaluate_stage2_test_metrics.py 并重算/汇总。"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from typing import Optional, Tuple

METRICS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(METRICS_DIR)
OUT_DIR = os.path.join(METRICS_DIR, "output")
OUT_CSV = os.path.join(OUT_DIR, "global_hard_dice_test_v17.csv")
OUT_TXT = os.path.join(OUT_DIR, "global_hard_dice_test_v17.txt")

EVAL_SCRIPT = os.path.join(CODE_ROOT, "evaluate_stage2_test_metrics.py")
SUMMARY_FILE = os.path.join(CODE_ROOT, "logs", "stage2_test_eval_metrics.txt")
DICE_KEYS = ("dice", "hard_dice")


def parse_global_hard_dice(summary_path: str) -> Optional[float]:
    if not os.path.isfile(summary_path):
        return None
    keys_lower = {k.lower() for k in DICE_KEYS}
    with open(summary_path, encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\t+", line.strip())
            if len(parts) != 2:
                continue
            key, val = parts[0].strip().lower(), parts[1].strip()
            if key in keys_lower:
                try:
                    return float(val)
                except ValueError:
                    continue
    return None


def parse_stdout_dice(stdout: str) -> Optional[float]:
    for line in stdout.splitlines():
        for pat in (
            r"hard_dice\s*[:=]\s*([0-9.]+)",
            r"Dice\s*\(global\)\s*:\s*([0-9.]+)",
            r"^Dice\s*:\s*([0-9.]+)",
        ):
            m = re.search(pat, line.strip(), re.I)
            if m:
                return float(m.group(1))
    return None


def run_eval() -> Tuple[Optional[float], str]:
    if not os.path.isfile(EVAL_SCRIPT):
        return None, f"缺少脚本: {EVAL_SCRIPT}"

    print(f"运行: {EVAL_SCRIPT}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-u", EVAL_SCRIPT],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-2000:]
        return None, f"运行失败 (code={proc.returncode}): {err}"

    dice = parse_global_hard_dice(SUMMARY_FILE)
    if dice is None:
        dice = parse_stdout_dice(proc.stdout or "")
    if dice is None:
        return None, f"未解析到 global hard Dice: {SUMMARY_FILE}"
    return dice, "ok"


def main():
    parse_only = "--parse-only" in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)

    if parse_only:
        dice = parse_global_hard_dice(SUMMARY_FILE)
        status = "ok" if dice is not None else f"missing in {SUMMARY_FILE}"
    else:
        dice, status = run_eval()

    row = {
        "version": "V17",
        "model": "Ours",
        "global_hard_dice": dice,
        "status": status,
        "summary_file": os.path.relpath(SUMMARY_FILE, CODE_ROOT),
    }

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["version", "model", "global_hard_dice", "status", "summary_file"],
        )
        w.writeheader()
        w.writerow(row)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("V17 Test global hard Dice\n")
        f.write("formula\t(2*inter+eps)/(pred_sum+gt_sum+eps), binary @0.5, micro over all Test lakes\n")
        f.write(f"eval_script\t{os.path.relpath(EVAL_SCRIPT, CODE_ROOT)}\n\n")
        d = f"{dice:.6f}" if dice is not None else "N/A"
        f.write(f"global_hard_dice\t{d}\n")
        f.write(f"status\t{status}\n")

    if dice is not None:
        print(f"global hard Dice = {dice:.6f}", flush=True)
    else:
        print(f"FAILED: {status}", flush=True)
    print(f"Saved:\n  {OUT_CSV}\n  {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
