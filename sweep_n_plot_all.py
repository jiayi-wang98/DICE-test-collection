#!/usr/bin/env python3
"""Combined N-sweep plot for all 7 state-PE benchmarks.

Sources:
  sweep_n_results.csv       — prefix_sum_large (BS=256) + reduce_large
  sweep_n_bs512_results.csv — prefix_sum_large at BS=512 (up to N=262K)
  sweep_n_more_results.csv  — dot_product, l2norm, variance, histo_k16,
                              block_max (BS=256, up to N=1M)

For prefix_sum we use BS=512 to reach N=262K. For the others, BS=256 (the
natural per-CTA size that fits the per-kernel reduction pattern).
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_ORIG  = "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/sweep_n_results.csv"
CSV_BS512 = "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/sweep_n_bs512_results.csv"
CSV_MORE  = "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/sweep_n_more_results.csv"
OUT       = "/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/sweep_n_plot_all.png"

rows_orig  = list(csv.DictReader(open(CSV_ORIG)))
rows_bs512 = list(csv.DictReader(open(CSV_BS512)))
rows_more  = list(csv.DictReader(open(CSV_MORE)))

def sel(rows, bench, backend):
    sub = [r for r in rows if r["benchmark"] == bench
                              and r["backend"] == backend
                              and r["correct"].lower() == "true"]
    return sorted(sub, key=lambda r: int(r["N"]))

# Per-panel config: (title, source_rows, bench_name, backends_to_show)
panels = [
    ("prefix_sum (BS=512)",          rows_bs512, "prefix_sum_large", ["gpu", "dice", "dice_acc"]),
    ("reduce_large",                 rows_orig,  "reduce_large",     ["gpu", "dice", "dice_acc"]),
    ("dot_product",                  rows_more,  "dot_product",      ["gpu", "dice", "dice_acc"]),
    ("l2norm",                       rows_more,  "l2norm",           ["gpu", "dice", "dice_acc"]),
    ("variance (2-slot)",            rows_more,  "variance",         ["gpu", "dice", "dice_acc"]),
    ("histo_k16 (16-slot)",          rows_more,  "histo_k16",        ["gpu", "dice", "dice_acc"]),
    ("block_max (MAX-mode)",         rows_more,  "block_max",        ["gpu", "dice", "dice_acc"]),
]

styles = {
    "gpu":      ("GPU SIMT",        "#d62728", "o", "-"),
    "dice":     ("DICE baseline",   "#1f77b4", "s", "-"),
    "dice_acc": ("DICE + state-PE", "#2ca02c", "^", "-"),
}

ncols = 4
nrows = 2 * ((len(panels) + ncols - 1) // ncols)  # cycles row + energy row per group
# Actually simpler: ncols panels, 2 rows (cycles top, energy bottom).
ncols = len(panels)
fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 7),
                         sharex='col')

for bi, (title, src, bench, backends) in enumerate(panels):
    for backend in backends:
        sub = sel(src, bench, backend)
        if not sub: continue
        Ns = [int(r["N"]) for r in sub]
        cy = [int(r["total_cycles"]) for r in sub]
        en = [float(r["sm_dyn_energy_uJ"]) for r in sub]
        label, color, marker, ls = styles[backend]
        axes[0][bi].loglog(Ns, cy, marker=marker, color=color, linestyle=ls,
                           label=label, linewidth=1.4, markersize=6)
        axes[1][bi].loglog(Ns, en, marker=marker, color=color, linestyle=ls,
                           label=label, linewidth=1.4, markersize=6)
    axes[0][bi].set_title(title, fontsize=10, fontweight='bold')
    axes[0][bi].grid(True, which='both', alpha=0.3)
    axes[1][bi].grid(True, which='both', alpha=0.3)
    if bi == 0:
        axes[0][bi].set_ylabel("Cycles")
        axes[1][bi].set_ylabel("SM_dyn E (μJ)")
    axes[1][bi].set_xlabel("N")
    if bi == 0:
        axes[0][bi].legend(fontsize=8, loc='upper left')

fig.suptitle("State-PE evaluation — 7 benchmarks, N-sweep, "
             "GPU SIMT vs DICE vs DICE+state-PE",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT, dpi=120, bbox_inches='tight')
print(f"wrote {OUT}")
