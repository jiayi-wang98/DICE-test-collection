#!/usr/bin/env python3
"""Re-do the GPU-side rows of sweep_n_more_results.csv (initial run
failed because gpgpusim.config was never copied into the GPU benchmark
dirs). Replaces those rows in place."""

import sys, csv, os
sys.path.insert(0, "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval")
from sweep_n import run_one, GPU_RODINIA, GPU_SETUP

CSV = "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/sweep_n_more_results.csv"
NS  = [1024, 4096, 16384, 65536, 262144, 1048576]
FIELDNAMES = ["benchmark", "backend", "N", "correct", "total_cycles",
              "kernels_cy", "sm_dyn_energy_uJ", "chip_energy_uJ", "wall_s"]
BENCHES = ["dot_product", "l2norm", "variance", "histo_k16", "block_max"]

# Read existing CSV, drop GPU rows for these benchmarks.
rows = list(csv.DictReader(open(CSV)))
keep = [r for r in rows if not (r["backend"] == "gpu" and r["benchmark"] in BENCHES)]

# Run GPU rows fresh.
new_gpu_rows = []
for bench in BENCHES:
    for N in NS:
        wd = f"{GPU_RODINIA}/{bench}"
        b = bench
        pwr = "accelwattch_power_report.log"
        if not os.path.exists(os.path.join(wd, b)):
            continue
        label = f"{bench}/gpu N={N}"
        print(f"[RUN ] {label}", flush=True)
        try:
            row = run_one(label, wd, b, N, pwr, GPU_SETUP)
        except Exception as e:
            print(f"[FAIL] {label}: {e}")
            continue
        row["benchmark"] = bench
        row["backend"] = "gpu"
        new_gpu_rows.append(row)
        print(f"      cy={row['total_cycles']} SM_E={row['sm_dyn_energy_uJ']}uJ "
              f"correct={row['correct']} wall={row['wall_s']}s", flush=True)

# Write merged CSV.
with open(CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    for r in keep + new_gpu_rows:
        w.writerow({k: r.get(k, "") for k in FIELDNAMES})
print(f"\nDONE. {len(new_gpu_rows)} new GPU rows; {len(keep)} kept.")
