#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_rf_access import compute_launch_deltas, group_by_kernel, parse_kernel_entries
from plot_scale_up_common import (
    BENCHMARK_ORDER,
    DICE_BASE_MODE,
    DICE_U_MODE,
    geomean,
    kernel_label_map_from_mode,
    latest_log_for_mode,
    write_csv,
)
from plot_speedup import safe_ratio


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "pdf.fonttype": 42,
    }
)

COLORS = ["#f49845", "#e66d6f", "#86c2c0", "#6aac61"]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    output_dir = script_dir / "generated_scale_up"

    parser = argparse.ArgumentParser(
        description="Compare DICE-U RF access against baseline DICE-full and generate CSV/plot outputs."
    )
    parser.add_argument(
        "--dice-root",
        type=Path,
        default=repo_root / "dice-test-gpu-rodinia/cuda/dice_test/result_summary",
        help="Path to the DICE result_summary directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir,
        help="Directory for generated CSV/plot outputs",
    )
    parser.add_argument(
        "--launch-csv",
        type=Path,
        default=None,
        help="Output CSV with per-launch DICE-U vs DICE RF access ratios",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output CSV with per-kernel and overall normalized RF access",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Output PNG plot path",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Output PDF plot path",
    )
    args = parser.parse_args()

    if args.launch_csv is None:
        args.launch_csv = args.output_dir / "dice_u_vs_dice_rf_launch_metrics.csv"
    if args.summary_csv is None:
        args.summary_csv = args.output_dir / "dice_u_vs_dice_rf_summary.csv"
    if args.png is None:
        args.png = args.output_dir / "40PE_vs_20PE_rf.png"
    if args.pdf is None:
        args.pdf = args.output_dir / "40PE_vs_20PE_rf.pdf"

    return args


def build_outputs(dice_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    label_map, ordered_kernels = kernel_label_map_from_mode(dice_root, DICE_BASE_MODE)
    launch_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for app in BENCHMARK_ORDER:
        dice_log = latest_log_for_mode(dice_root / app, DICE_BASE_MODE)
        dice_u_log = latest_log_for_mode(dice_root / app, DICE_U_MODE)

        dice_launches = compute_launch_deltas(parse_kernel_entries(dice_log))
        dice_u_launches = compute_launch_deltas(parse_kernel_entries(dice_u_log))

        dice_by_kernel = group_by_kernel(dice_launches)
        dice_u_by_kernel = group_by_kernel(dice_u_launches)

        for kernel_name, kernel_label in label_map[app].items():
            dice_runs = dice_by_kernel[kernel_name]
            dice_u_runs = dice_u_by_kernel.get(kernel_name, [])
            if len(dice_runs) != len(dice_u_runs):
                raise ValueError(
                    f"{app} {kernel_name}: DICE has {len(dice_runs)} runs, "
                    f"DICE-U has {len(dice_u_runs)} runs"
                )

            dice_total_sum = 0
            dice_u_total_sum = 0

            for run_index, (dice_row, dice_u_row) in enumerate(zip(dice_runs, dice_u_runs), start=1):
                dice_total = int(dice_row["reg_total_delta"])
                dice_u_total = int(dice_u_row["reg_total_delta"])
                ratio = safe_ratio(dice_u_total, dice_total)

                dice_total_sum += dice_total
                dice_u_total_sum += dice_u_total

                launch_rows.append(
                    {
                        "app": app,
                        "kernel_label": kernel_label,
                        "kernel_name": kernel_name,
                        "run_index": run_index,
                        "dice_kernel_launch_uid": dice_row["kernel_launch_uid"],
                        "dice_u_kernel_launch_uid": dice_u_row["kernel_launch_uid"],
                        "dice_reg_total_delta": dice_total,
                        "dice_u_reg_total_delta": dice_u_total,
                        "normalized_rf_access": ratio,
                    }
                )

            summary_rows.append(
                {
                    "app": app,
                    "kernel_label": kernel_label,
                    "kernel_name": kernel_name,
                    "run_count": len(dice_runs),
                    "dice_reg_total_sum": dice_total_sum,
                    "dice_u_reg_total_sum": dice_u_total_sum,
                    "normalized_rf_access": safe_ratio(dice_u_total_sum, dice_total_sum),
                }
            )

    ordered_labels = [label for _, _, label in ordered_kernels]
    rows_by_label = {row["kernel_label"]: row for row in summary_rows}
    ordered_summary_rows = [rows_by_label[label] for label in ordered_labels]
    ordered_summary_rows.append(
        {
            "app": "OVERALL",
            "kernel_label": "GEOMEAN",
            "kernel_name": "GEOMEAN",
            "run_count": len(ordered_summary_rows),
            "dice_reg_total_sum": "",
            "dice_u_reg_total_sum": "",
            "normalized_rf_access": geomean([row["normalized_rf_access"] for row in ordered_summary_rows]),
        }
    )

    return launch_rows, ordered_summary_rows, ordered_labels + ["GEOMEAN"]


def make_plot(summary_rows: list[dict[str, object]], labels: list[str], png_path: Path, pdf_path: Path) -> None:
    lookup = {row["kernel_label"]: row for row in summary_rows}
    rf_all = np.array([lookup[label]["normalized_rf_access"] for label in labels], dtype=float)
    rf_ref_all = np.ones_like(rf_all, dtype=float)

    x = np.arange(len(labels), dtype=float)
    x[-1] += 0.6
    width = 0.36

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(
        x - width / 2,
        rf_all,
        width,
        color=COLORS[0],
        edgecolor="black",
        linewidth=0.8,
        label="DICE-U",
    )
    ax.bar(
        x + width / 2,
        rf_ref_all,
        width,
        color="white",
        edgecolor="black",
        linewidth=0.8,
        hatch="///",
        label="DICE",
    )

    ymin = min(0.4, float(np.floor((min(rf_all.min(), 1.0) - 0.05) * 10.0) / 10.0))
    ymax = max(1.2, float(np.ceil((max(rf_all.max(), 1.0) + 0.05) * 10.0) / 10.0))

    ax.set_ylabel("Normalized # of RF Access (x)")
    ax.yaxis.set_label_coords(-0.15, 0.37)
    ax.set_ylim(ymin, ymax)
    ax.axhline(1.0, linestyle="--", linewidth=1, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axvline(x[-1] - 0.7, linestyle=":", linewidth=1, color="black")

    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.57, 1.02),
        ncol=2,
        frameon=True,
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    launch_rows, summary_rows, labels = build_outputs(args.dice_root)

    write_csv(
        args.launch_csv,
        launch_rows,
        [
            "app",
            "kernel_label",
            "kernel_name",
            "run_index",
            "dice_kernel_launch_uid",
            "dice_u_kernel_launch_uid",
            "dice_reg_total_delta",
            "dice_u_reg_total_delta",
            "normalized_rf_access",
        ],
    )
    write_csv(
        args.summary_csv,
        summary_rows,
        [
            "app",
            "kernel_label",
            "kernel_name",
            "run_count",
            "dice_reg_total_sum",
            "dice_u_reg_total_sum",
            "normalized_rf_access",
        ],
    )
    make_plot(summary_rows, labels, args.png, args.pdf)

    print(f"Wrote {len(launch_rows)} launch rows to {args.launch_csv}")
    print(f"Wrote {len(summary_rows)} summary rows to {args.summary_csv}")
    print(f"Wrote plot to {args.png}")
    print(f"Wrote plot to {args.pdf}")


if __name__ == "__main__":
    main()
