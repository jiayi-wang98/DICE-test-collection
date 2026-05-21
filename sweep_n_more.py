#!/usr/bin/env python3
"""N-sweep for the 5 new state-PE benchmarks (dot_product, l2norm,
variance, histo_k16, block_max). Writes sweep_n_more_results.csv."""

import sys
sys.path.insert(0, "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval")
from sweep_n import (run_one, GPU_RODINIA, DICE_RODINIA,
                     GPU_SETUP, DICE_SETUP)
import csv, os

CSV = "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/sweep_n_more_results.csv"
NS  = [1024, 4096, 16384, 65536, 262144, 1048576]
FIELDNAMES = ["benchmark", "backend", "N", "correct", "total_cycles",
              "kernels_cy", "sm_dyn_energy_uJ", "chip_energy_uJ", "wall_s"]

BENCHES = ["dot_product", "l2norm", "variance", "histo_k16", "block_max"]

with open(CSV, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

for bench in BENCHES:
    for N in NS:
        for backend, (wd, b, pwr, setup) in [
            ("gpu",      (f"{GPU_RODINIA}/{bench}",      bench,           "accelwattch_power_report.log", GPU_SETUP)),
            ("dice",     (f"{DICE_RODINIA}/{bench}",     bench,           "gpgpusim_dice_power_report.log", DICE_SETUP)),
            ("dice_acc", (f"{DICE_RODINIA}/{bench}_acc", f"{bench}_acc",  "gpgpusim_dice_power_report.log", DICE_SETUP)),
        ]:
            if not os.path.exists(os.path.join(wd, b)):
                print(f"[SKIP] {bench}/{backend} N={N}: missing", flush=True)
                continue
            label = f"{bench}/{backend} N={N}"
            print(f"[RUN ] {label}", flush=True)
            try:
                row = run_one(label, wd, b, N, pwr, setup)
            except Exception as e:
                print(f"[FAIL] {label}: {e}", flush=True)
                continue
            row["benchmark"] = bench
            row["backend"] = backend
            with open(CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(
                    {k: row.get(k, "") for k in FIELDNAMES})
            print(f"      cy={row['total_cycles']} SM_E={row['sm_dyn_energy_uJ']}uJ "
                  f"correct={row['correct']} wall={row['wall_s']}s", flush=True)
print("\nDONE")
