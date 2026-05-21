#!/usr/bin/env python3
"""
N-sweep harness for prefix_sum and reduce_large benchmarks.

Per (benchmark, backend, N): run the binary, parse cycles from sim stdout,
parse SM-only dynamic power from the power report, write a CSV row.

Backends:
  gpu      - unmodified gpgpu-sim + AccelWattch (gpu-rodinia/cuda)
  dice     - DICE backend with SMEM-tree variant (dice-test-gpu-rodinia/cuda)
  dice_acc - DICE backend with state-PE variant (..._acc dir)

SM-only dynamic = all "*P" components minus
  {L2CP, MCP, NOCP, DRAMP, IDLE_COREP, CONST_DYNAMICP, CONSTP, STATICP}
"""

import csv
import os
import re
import subprocess
import sys
import time

ROOT          = "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval"
GPU_RODINIA   = f"{ROOT}/gpu-rodinia/cuda"
DICE_RODINIA  = f"{ROOT}/dice-test-gpu-rodinia/cuda"
GPU_SETUP     = f"{ROOT}/gpgpu-sim_distribution/setup_environment"
DICE_SETUP    = f"{ROOT}/dice_gpgpu-sim/setup_environment"
FREQ_GHZ      = 1.47
EXCLUDE       = {"L2CP", "MCP", "NOCP", "DRAMP", "IDLE_COREP",
                 "CONST_DYNAMICP", "CONSTP", "STATICP"}

# Per-benchmark configs. nblock_cap: K2-single-CTA constraint for prefix_sum.
BENCHES = {
    "prefix_sum_large": {
        "gpu_dir":      f"{GPU_RODINIA}/prefix_sum_large",
        "gpu_bin":      "prefix_sum_large",
        "gpu_power":    "accelwattch_power_report.log",
        "dice_dir":     f"{DICE_RODINIA}/prefix_sum_large",
        "dice_bin":     "prefix_sum_large",
        "dice_power":   "gpgpusim_dice_power_report.log",
        "acc_dir":      f"{DICE_RODINIA}/prefix_sum_large_acc",
        "acc_bin":      "prefix_sum_large_acc",
        "acc_power":    "gpgpusim_dice_power_report.log",
        "Ns":           [1024, 4096, 16384, 65536],
    },
    "reduce_large": {
        "gpu_dir":      f"{GPU_RODINIA}/reduce_large",
        "gpu_bin":      "reduce_large",
        "gpu_power":    "accelwattch_power_report.log",
        "dice_dir":     f"{DICE_RODINIA}/reduce_large",
        "dice_bin":     "reduce_large",
        "dice_power":   "gpgpusim_dice_power_report.log",
        "acc_dir":      f"{DICE_RODINIA}/reduce_large_acc",
        "acc_bin":      "reduce_large_acc",
        "acc_power":    "gpgpusim_dice_power_report.log",
        "Ns":           [1024, 4096, 16384, 65536, 262144, 1048576],
    },
}

CYCLE_RE = re.compile(r"gpu_sim_cycle\s*=\s*(\d+)")
KAVG_RE  = re.compile(r"kernel_avg_power\s*=\s*([\d.eE+-]+)")
COMP_RE  = re.compile(r"gpu_avg_(\w+P|STATICP),?\s*=\s*([\d.eE+-]+)")
MATCH_RE = re.compile(r"results match|MISMATCH")

CSV_PATH = f"{ROOT}/sweep_n_results.csv"

def shell(cmd, cwd=None, timeout=3600, env=None):
    """Run shell command, return (rc, stdout, stderr)."""
    p = subprocess.run(["bash", "-c", cmd], cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr

def env_with_setup(setup_path):
    """Source setup_environment and return resulting env."""
    rc, out, _ = shell(f". {setup_path} > /dev/null 2>&1 && env -0",
                       timeout=60)
    env = dict(os.environ)
    if rc == 0:
        for line in out.split('\0'):
            if '=' in line:
                k, v = line.split('=', 1)
                env[k] = v
    return env

def parse_power_log(path):
    """Return list of (kernel_avg_power, sm_dyn_power_W) per kernel."""
    if not os.path.exists(path):
        return []
    data = open(path).read()
    sections = re.split(r"kernel_avg_power = ", data)
    out = []
    for s in sections[1:]:
        lines = s.split('\n')
        try:
            kavg = float(lines[0])
        except ValueError:
            continue
        sm = 0.0
        for line in lines[1:]:
            if "Kernel Maximum" in line or "Accumulative" in line:
                break
            m = COMP_RE.match(line.strip())
            if not m:
                continue
            name, val = m.group(1), float(m.group(2))
            if name in EXCLUDE:
                continue
            sm += val
        out.append((kavg, sm))
    return out

def run_one(label, work_dir, binary, N, power_log, setup_path):
    """Run one (backend, benchmark, N) point. Returns dict row or None."""
    env = env_with_setup(setup_path)
    # Clean prior outputs.
    for f in [power_log, "sim.out", "_app_cuda_version_*",
              "_cuobjdump_list_ptx_*"]:
        shell(f"rm -f {f}", cwd=work_dir)
    t0 = time.time()
    rc, out, err = shell(f"./{binary} {N} > sim.out 2>&1",
                         cwd=work_dir, timeout=3600, env=env)
    elapsed = time.time() - t0
    # Parse cycles per kernel.
    sim_out = open(os.path.join(work_dir, "sim.out")).read()
    cycles = [int(m.group(1)) for m in CYCLE_RE.finditer(sim_out)]
    # Correctness check.
    match = MATCH_RE.search(sim_out)
    correct = bool(match and "MISMATCH" not in match.group(0))
    # Per-kernel power.
    power_path = os.path.join(work_dir, power_log)
    per_k = parse_power_log(power_path)
    # Aggregate: total cycles, integrated SM energy (sum of P_k * cy_k / f).
    total_cy = sum(cycles)
    sm_e_uJ = 0.0
    chip_e_uJ = 0.0
    for k, cy in enumerate(cycles):
        if k < len(per_k):
            kavg, sm = per_k[k]
            sm_e_uJ   += sm   * cy / (FREQ_GHZ * 1e9) * 1e6  # W*cy/(GHz*1e9)*1e6 = uJ
            chip_e_uJ += kavg * cy / (FREQ_GHZ * 1e9) * 1e6
    return {
        "label": label, "N": N, "correct": correct,
        "total_cycles": total_cy,
        "kernels_cy": "|".join(str(c) for c in cycles),
        "sm_dyn_energy_uJ": round(sm_e_uJ, 3),
        "chip_energy_uJ":  round(chip_e_uJ, 3),
        "wall_s": round(elapsed, 1),
    }

def main():
    rows = []
    fieldnames = ["benchmark", "backend", "N", "correct", "total_cycles",
                  "kernels_cy", "sm_dyn_energy_uJ", "chip_energy_uJ",
                  "wall_s"]

    # Write header immediately so partial runs are visible.
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for bench, cfg in BENCHES.items():
        for N in cfg["Ns"]:
            for backend, (dir_key, bin_key, pwr_key, setup) in [
                ("gpu",      ("gpu_dir",  "gpu_bin",  "gpu_power",  GPU_SETUP)),
                ("dice",     ("dice_dir", "dice_bin", "dice_power", DICE_SETUP)),
                ("dice_acc", ("acc_dir",  "acc_bin",  "acc_power",  DICE_SETUP)),
            ]:
                wd = cfg[dir_key]
                if not os.path.exists(os.path.join(wd, cfg[bin_key])):
                    print(f"[SKIP] {bench}/{backend} N={N}: binary missing")
                    continue
                tag = f"{bench}/{backend} N={N}"
                print(f"[RUN ] {tag}", flush=True)
                try:
                    row = run_one(tag, wd, cfg[bin_key], N,
                                  cfg[pwr_key], setup)
                except subprocess.TimeoutExpired:
                    print(f"[TIMEOUT] {tag}")
                    continue
                row["benchmark"] = bench
                row["backend"] = backend
                rows.append(row)
                with open(CSV_PATH, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(
                        {k: row.get(k, "") for k in fieldnames})
                print(f"      cy={row['total_cycles']} "
                      f"SM_E={row['sm_dyn_energy_uJ']}uJ "
                      f"chip_E={row['chip_energy_uJ']}uJ "
                      f"correct={row['correct']} "
                      f"wall={row['wall_s']}s",
                      flush=True)

    print(f"\nDONE. Wrote {len(rows)} rows to {CSV_PATH}")

if __name__ == "__main__":
    main()
