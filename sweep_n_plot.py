#!/usr/bin/env python3
"""Combined N-sweep plot.

prefix_sum panel: BS=512 sweep from sweep_n_bs512_results.csv (N up to 262K).
                  5 lines (GPU HS / GPU Blelloch / DICE HS / DICE Blelloch /
                  DICE+state-PE). DICE Blelloch is plotted but lightly
                  styled — currently MISMATCH at BS=512 (compiler edge case
                  in the DICE-ILP register-renaming pass for Blelloch's
                  shrinking-active-thread pattern at larger block sizes).
reduce_large panel: original BS=256 sweep from sweep_n_results.csv,
                    N up to 1M, 3 backends.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_BS512  = "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/sweep_n_bs512_results.csv"
CSV_ORIG   = "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/sweep_n_results.csv"
OUT        = "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/sweep_n_plot.png"

rows_bs512 = list(csv.DictReader(open(CSV_BS512)))
rows_orig  = list(csv.DictReader(open(CSV_ORIG)))

def select(rows, bench, backend):
    return sorted([r for r in rows
                   if r["benchmark"] == bench and r["backend"] == backend],
                  key=lambda r: int(r["N"]))

# Series: (rows_src, bench, backend, label, color, marker, linestyle, alpha)
prefix_series = [
    (rows_bs512, "prefix_sum_large",          "gpu",      "GPU HS",          "#d62728", "o", "-",  1.0),
    (rows_bs512, "prefix_sum_large_blelloch", "gpu",      "GPU Blelloch",    "#d62728", "o", "--", 1.0),
    (rows_bs512, "prefix_sum_large",          "dice",     "DICE HS",         "#1f77b4", "s", "-",  1.0),
    (rows_bs512, "prefix_sum_large_blelloch", "dice",     "DICE Blelloch (MISMATCH)", "#1f77b4", "s", ":",  0.4),
    (rows_bs512, "prefix_sum_large",          "dice_acc", "DICE + state-PE", "#2ca02c", "^", "-",  1.0),
]

reduce_series = [
    (rows_orig, "reduce_large", "gpu",      "GPU SIMT",        "#d62728", "o", "-", 1.0),
    (rows_orig, "reduce_large", "dice",     "DICE (SMEM tree)","#1f77b4", "s", "-", 1.0),
    (rows_orig, "reduce_large", "dice_acc", "DICE + state-PE", "#2ca02c", "^", "-", 1.0),
]

panels = [
    ("prefix_sum (BLOCK_SIZE=512, inclusive scan)", prefix_series),
    ("reduce_large (BLOCK_SIZE=256, sum reduction)", reduce_series),
]

fig, axes = plt.subplots(2, len(panels), figsize=(7*len(panels), 9),
                         sharex='col')
for bi, (title, series) in enumerate(panels):
    for src, bench, backend, label, color, marker, ls, alpha in series:
        sub = select(src, bench, backend)
        if not sub: continue
        Ns = [int(r["N"]) for r in sub]
        cy = [int(r["total_cycles"]) for r in sub]
        en = [float(r["sm_dyn_energy_uJ"]) for r in sub]
        axes[0][bi].loglog(Ns, cy, marker=marker, color=color, linestyle=ls,
                           label=label, linewidth=1.6, markersize=8,
                           alpha=alpha)
        axes[1][bi].loglog(Ns, en, marker=marker, color=color, linestyle=ls,
                           label=label, linewidth=1.6, markersize=8,
                           alpha=alpha)
    axes[0][bi].set_title(title, fontsize=12, fontweight='bold')
    axes[0][bi].set_ylabel("Total cycles")
    axes[0][bi].grid(True, which='both', alpha=0.3)
    axes[0][bi].legend(fontsize=9, loc='upper left')
    axes[1][bi].set_ylabel("SM-only dynamic energy (μJ)")
    axes[1][bi].set_xlabel("N (input size)")
    axes[1][bi].grid(True, which='both', alpha=0.3)
    axes[1][bi].legend(fontsize=9, loc='upper left')

fig.suptitle("N-sweep — GPU SIMT vs DICE vs DICE+state-PE",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT, dpi=120, bbox_inches='tight')
print(f"wrote {OUT}")
