#!/usr/bin/env python3
"""Re-sweep prefix_sum at BLOCK_SIZE=512 to reach N=262144 (256K).
Writes a new CSV; the existing BS=256 results stay in sweep_n_results.csv.

Note: DICE Blelloch at BLOCK_THREADS=256 has a known compiler-side bug
(passes at BLOCK_THREADS=128 but mis-renames registers at 256). We still
run it and record the MISMATCH so the issue is visible, but the plot
will mark it accordingly.
"""
import sys
sys.path.insert(0, "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval")
from sweep_n import (run_one, GPU_RODINIA, DICE_RODINIA,
                     GPU_SETUP, DICE_SETUP)
import csv, os

NS = [1024, 4096, 16384, 65536, 262144]
CSV = "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/sweep_n_bs512_results.csv"
FIELDNAMES = ["benchmark", "backend", "N", "block_size", "correct",
              "total_cycles", "kernels_cy", "sm_dyn_energy_uJ",
              "chip_energy_uJ", "wall_s"]

# (bench_label, backend_label, dir, binary, power_log, setup)
runs = [
    ("prefix_sum_large", "gpu",
     f"{GPU_RODINIA}/prefix_sum_large",      "prefix_sum_large",
     "accelwattch_power_report.log",         GPU_SETUP),
    ("prefix_sum_large", "dice",
     f"{DICE_RODINIA}/prefix_sum_large",     "prefix_sum_large",
     "gpgpusim_dice_power_report.log",       DICE_SETUP),
    ("prefix_sum_large", "dice_acc",
     f"{DICE_RODINIA}/prefix_sum_large_acc", "prefix_sum_large_acc",
     "gpgpusim_dice_power_report.log",       DICE_SETUP),
    ("prefix_sum_large_blelloch", "gpu",
     f"{GPU_RODINIA}/prefix_sum_large_blelloch",  "prefix_sum_large_blelloch",
     "accelwattch_power_report.log",              GPU_SETUP),
    ("prefix_sum_large_blelloch", "dice",
     f"{DICE_RODINIA}/prefix_sum_large_blelloch", "prefix_sum_large_blelloch",
     "gpgpusim_dice_power_report.log",            DICE_SETUP),
]

with open(CSV, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

for N in NS:
    for bench, backend, wd, b, pwr, setup in runs:
        if not os.path.exists(os.path.join(wd, b)):
            print(f"[SKIP] {bench}/{backend} N={N}: binary missing", flush=True)
            continue
        label = f"{bench}/{backend} N={N}"
        print(f"[RUN ] {label}", flush=True)
        row = run_one(label, wd, b, N, pwr, setup)
        row["benchmark"] = bench
        row["backend"] = backend
        row["block_size"] = 512
        with open(CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
                {k: row.get(k, "") for k in FIELDNAMES})
        print(f"      cy={row['total_cycles']} "
              f"SM_E={row['sm_dyn_energy_uJ']}uJ "
              f"correct={row['correct']} "
              f"wall={row['wall_s']}s", flush=True)
print("\nDONE")
