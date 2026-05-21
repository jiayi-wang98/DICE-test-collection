#!/usr/bin/env python3
"""Extend sweep_n_results.csv with prefix_sum_large_blelloch GPU+DICE at the
same N grid as the HS prefix_sum (1K, 4K, 16K, 65K)."""

import sys
sys.path.insert(0, "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval")
from sweep_n import (run_one, GPU_RODINIA, DICE_RODINIA,
                     GPU_SETUP, DICE_SETUP, CSV_PATH)
import csv, os

BENCH = "prefix_sum_large_blelloch"
NS    = [1024, 4096, 16384, 65536]

FIELDNAMES = ["benchmark", "backend", "N", "correct", "total_cycles",
              "kernels_cy", "sm_dyn_energy_uJ", "chip_energy_uJ", "wall_s"]

backends = [
    ("gpu",  f"{GPU_RODINIA}/{BENCH}",  BENCH, "accelwattch_power_report.log",
     GPU_SETUP),
    ("dice", f"{DICE_RODINIA}/{BENCH}", BENCH, "gpgpusim_dice_power_report.log",
     DICE_SETUP),
]

# Sanity: binaries exist
for tag, wd, b, _, _ in backends:
    if not os.path.exists(os.path.join(wd, b)):
        print(f"[ABORT] missing binary: {wd}/{b}")
        sys.exit(1)

with open(CSV_PATH, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    for N in NS:
        for tag, wd, b, pwr, setup in backends:
            label = f"{BENCH}/{tag} N={N}"
            print(f"[RUN ] {label}", flush=True)
            row = run_one(label, wd, b, N, pwr, setup)
            row["benchmark"] = BENCH
            row["backend"] = tag
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})
            f.flush()
            print(f"      cy={row['total_cycles']} "
                  f"SM_E={row['sm_dyn_energy_uJ']}uJ "
                  f"correct={row['correct']} "
                  f"wall={row['wall_s']}s", flush=True)
print("\nDONE")
