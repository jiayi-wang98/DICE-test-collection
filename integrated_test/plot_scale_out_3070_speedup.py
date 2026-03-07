#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_speedup import BENCHMARK_ORDER, as_int, geomean, read_csv_rows, safe_ratio, write_csv


SERIES = [
    ("RTX3070", 1.0),
    ("DICE-3070", None),
]

PLOT_COLORS = ["#4e79a7", "#f28e2c"]
PLOT_HATCHES = ["", "///"]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    output_dir = script_dir / "generated_scale_out"

    parser = argparse.ArgumentParser(
        description=(
            "Collect scale-out DICE/GPU summary data, normalize DICE-3070 "
            "against RTX3070, and generate CSV/plot outputs."
        )
    )
    parser.add_argument(
        "--dice-root",
        type=Path,
        default=repo_root / "dice-test-gpu-rodinia/cuda/dice_test/result_summary",
        help="Path to the DICE result_summary directory",
    )
    parser.add_argument(
        "--gpu-root",
        type=Path,
        default=repo_root / "gpu-rodinia/cuda/gpu_test/result_summary",
        help="Path to the GPU result_summary directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir,
        help="Directory for generated CSV/plot outputs",
    )
    parser.add_argument("--launch-csv", type=Path, default=None, help="Output CSV with per-launch metrics")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Output CSV with per-kernel speedups")
    parser.add_argument("--png", type=Path, default=None, help="Output PNG plot path")
    parser.add_argument("--pdf", type=Path, default=None, help="Output PDF plot path")
    args = parser.parse_args()

    if args.launch_csv is None:
        args.launch_csv = args.output_dir / "scale_out_3070_speedup_launch_metrics.csv"
    if args.summary_csv is None:
        args.summary_csv = args.output_dir / "scale_out_3070_speedup_summary.csv"
    if args.png is None:
        args.png = args.output_dir / "scale_out_3070_speedup.png"
    if args.pdf is None:
        args.pdf = args.output_dir / "scale_out_3070_speedup.pdf"
    return args


def grouped_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["date_time"]].append(row)
    return grouped


def latest_scale_out_date_map(rows: list[dict[str, str]]) -> dict[str, str]:
    dates = sorted({row["date_time"] for row in rows})
    if len(dates) < 3:
        raise ValueError("Need at least three collected date groups to identify scale-out results")
    return {"rtx5000": dates[-3], "rtx6000": dates[-2], "rtx3070": dates[-1]}


def skip_launch_uids(app: str, kernel_name: str, rows: list[dict[str, str]]) -> set[int]:
    sorted_rows = sorted(rows, key=lambda item: int(item["kernel_launch_uid"]))
    if len(sorted_rows) < 2:
        return set()

    launch_uids = [int(row["kernel_launch_uid"]) for row in sorted_rows]
    positive_diffs = [curr - prev for prev, curr in zip(launch_uids, launch_uids[1:]) if curr > prev]
    if not positive_diffs:
        return set()

    expected_step = min(positive_diffs)
    skipped: set[int] = set()
    previous_uid: int | None = None

    for row in sorted_rows:
        launch_uid = int(row["kernel_launch_uid"])
        if previous_uid is not None and launch_uid - previous_uid > expected_step:
            print(
                f"Warning: {app} {kernel_name}: detected UID gap {previous_uid}->{launch_uid}; "
                f"skipping launch {launch_uid} per combined-run rule"
            )
            skipped.add(launch_uid)
        previous_uid = launch_uid

    return skipped


def rows_by_launch_uid(rows: list[dict[str, str]], skipped_uids: set[int]) -> dict[int, dict[str, str]]:
    return {
        int(row["kernel_launch_uid"]): row
        for row in sorted(rows, key=lambda item: int(item["kernel_launch_uid"]))
        if int(row["kernel_launch_uid"]) not in skipped_uids
    }


def kernel_label_map(gpu_root: Path) -> tuple[dict[str, dict[str, str]], list[tuple[str, str, str]]]:
    per_app: dict[str, dict[str, str]] = {}
    ordered: list[tuple[str, str, str]] = []

    for app in BENCHMARK_ORDER:
        rows = read_csv_rows(gpu_root / app / f"{app}.result.csv")
        date_map = latest_scale_out_date_map(rows)
        rtx3070_rows = grouped_rows(rows)[date_map["rtx3070"]]

        kernels_in_order: list[str] = []
        for row in sorted(rtx3070_rows, key=lambda item: int(item["kernel_launch_uid"])):
            kernel_name = row["kernel_name"]
            if kernel_name not in kernels_in_order:
                kernels_in_order.append(kernel_name)

        per_app[app] = {}
        multi_kernel = len(kernels_in_order) > 1
        for index, kernel_name in enumerate(kernels_in_order, start=1):
            label = f"{app}-{index}" if multi_kernel else app
            per_app[app][kernel_name] = label
            ordered.append((app, kernel_name, label))

    return per_app, ordered


def build_outputs(
    dice_root: Path,
    gpu_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[tuple[str, str, str]]]:
    label_map, ordered_kernels = kernel_label_map(gpu_root)

    launch_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for app in BENCHMARK_ORDER:
        gpu_rows = read_csv_rows(gpu_root / app / f"{app}.result.csv")
        gpu_date = latest_scale_out_date_map(gpu_rows)["rtx3070"]
        gpu_group = grouped_rows(gpu_rows)[gpu_date]

        dice_rows = read_csv_rows(dice_root / app / f"{app}.result.csv")
        dice_date = latest_scale_out_date_map(dice_rows)["rtx3070"]
        dice_group = grouped_rows(dice_rows)[dice_date]

        gpu_by_kernel: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in gpu_group:
            gpu_by_kernel[row["kernel_name"]].append(row)
        dice_by_kernel: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in dice_group:
            dice_by_kernel[row["kernel_name"]].append(row)

        for kernel_name, kernel_label in label_map[app].items():
            gpu_kernel_rows = gpu_by_kernel[kernel_name]
            dice_kernel_rows = dice_by_kernel[kernel_name]

            gpu_skipped = skip_launch_uids(app, kernel_name, gpu_kernel_rows)
            dice_skipped = skip_launch_uids(app, kernel_name, dice_kernel_rows)

            gpu_by_uid = rows_by_launch_uid(gpu_kernel_rows, gpu_skipped)
            dice_by_uid = rows_by_launch_uid(dice_kernel_rows, dice_skipped)

            common_uids = sorted(set(gpu_by_uid) & set(dice_by_uid))
            if not common_uids:
                raise ValueError(f"{app} {kernel_name}: no common launch IDs across selected RTX3070 groups")

            max_count = max(len(gpu_by_uid), len(dice_by_uid))
            if len(common_uids) != max_count:
                print(
                    f"Warning: {app} {kernel_name}: using {len(common_uids)} common launch IDs "
                    f"(gpu3070={len(gpu_by_uid)}, dice3070={len(dice_by_uid)})"
                )

            kernel_speedups: list[float | None] = []
            for run_index, launch_uid in enumerate(common_uids, start=1):
                gpu_row = gpu_by_uid[launch_uid]
                dice_row = dice_by_uid[launch_uid]

                baseline_cycle = as_int(gpu_row["gpu_sim_cycle"])
                dice_cycle = as_int(dice_row["gpu_sim_cycle"])
                speedup = safe_ratio(baseline_cycle, dice_cycle)
                kernel_speedups.append(speedup)

                launch_rows.append(
                    {
                        "app": app,
                        "kernel_label": kernel_label,
                        "kernel_name": kernel_name,
                        "run_index": run_index,
                        "matched_kernel_launch_uid": launch_uid,
                        "gpu_rtx3070_date_time": gpu_date,
                        "dice_rtx3070_date_time": dice_date,
                        "gpu_rtx3070_cycle": baseline_cycle,
                        "dice_rtx3070_cycle": dice_cycle,
                        "dice_rtx3070_speedup_vs_rtx3070": speedup,
                    }
                )

            summary_rows.append(
                {
                    "app": app,
                    "kernel_label": kernel_label,
                    "kernel_name": kernel_name,
                    "run_count": len(common_uids),
                    "RTX3070": 1.0,
                    "DICE-3070": geomean(kernel_speedups),
                }
            )

    summary_rows.append(
        {
            "app": "OVERALL",
            "kernel_label": "GEOMEAN",
            "kernel_name": "GEOMEAN",
            "run_count": len(summary_rows),
            "RTX3070": 1.0,
            "DICE-3070": geomean([row["DICE-3070"] for row in summary_rows]),
        }
    )

    return launch_rows, summary_rows, ordered_kernels


def make_plot(
    summary_rows: list[dict[str, object]],
    ordered_kernels: list[tuple[str, str, str]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    labels = [label for _, _, label in ordered_kernels] + ["GEOMEAN"]
    summary_by_label = {str(row["kernel_label"]): row for row in summary_rows}

    speedups = {
        series_label: np.array([summary_by_label[kernel_label][series_label] for kernel_label in labels], dtype=float)
        for series_label, _ in SERIES
    }

    fig, ax = plt.subplots(figsize=(8, 3))
    x_positions = np.arange(len(labels), dtype=float)
    gap_size = 0.5
    x_positions[-1] += gap_size
    total_width = 0.75
    bar_width = total_width / len(SERIES)

    containers = []
    for index, (series_label, _) in enumerate(SERIES):
        container = ax.bar(
            x_positions - total_width / 2 + index * bar_width,
            speedups[series_label],
            width=bar_width,
            color=PLOT_COLORS[index],
            edgecolor="black",
            linewidth=0.8,
            hatch=PLOT_HATCHES[index],
            label=series_label,
        )
        containers.append(container)

    separator_x = (x_positions[-2] + x_positions[-1]) / 2 - gap_size / 2
    ax.axvline(x=separator_x, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1.5)

    ax.set_ylabel("Normalized Speedup (x)\n(vs. RTX3070)", fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    handles = []
    for index, (series_label, _) in enumerate(SERIES):
        patch = plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=PLOT_COLORS[index],
            edgecolor="black",
            hatch=PLOT_HATCHES[index],
            linewidth=0.8,
            label=series_label,
        )
        handles.append(patch)
    ax.legend(
        handles,
        [series_label for series_label, _ in SERIES],
        fontsize=15,
        loc="lower center",
        bbox_to_anchor=(0.48, 1.0),
        ncol=2,
        frameon=True,
        fancybox=False,
        columnspacing=0.5,
        handlelength=1.5,
        handletextpad=0.3,
    )

    ax.text(
        1.0,
        0.91,
        "*: geometric mean",
        transform=ax.transAxes,
        fontsize=13,
        style="italic",
        ha="right",
        va="bottom",
    )

    ax.set_xlim(-0.5, x_positions[-1] + 0.5)
    ymax = max(series.max() for series in speedups.values())
    ax.set_ylim(0, max(1.6, ymax * 1.1))

    for index, container in enumerate(containers):
        geomean_bar = container[-1]
        height = geomean_bar.get_height()
        ax.text(
            geomean_bar.get_x() + geomean_bar.get_width() / 2 + 0.06 * index,
            height + 0.03 * index,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    launch_rows, summary_rows, ordered_kernels = build_outputs(args.dice_root, args.gpu_root)

    launch_fieldnames = [
        "app",
        "kernel_label",
        "kernel_name",
        "run_index",
        "matched_kernel_launch_uid",
        "gpu_rtx3070_date_time",
        "dice_rtx3070_date_time",
        "gpu_rtx3070_cycle",
        "dice_rtx3070_cycle",
        "dice_rtx3070_speedup_vs_rtx3070",
    ]
    summary_fieldnames = [
        "app",
        "kernel_label",
        "kernel_name",
        "run_count",
        "RTX3070",
        "DICE-3070",
    ]

    args.launch_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.launch_csv, launch_rows, launch_fieldnames)
    write_csv(args.summary_csv, summary_rows, summary_fieldnames)
    make_plot(summary_rows, ordered_kernels, args.png, args.pdf)

    print(f"Wrote {len(launch_rows)} launch rows to {args.launch_csv}")
    print(f"Wrote {len(summary_rows)} summary rows to {args.summary_csv}")
    print(f"Wrote plot to {args.png}")
    print(f"Wrote plot to {args.pdf}")


if __name__ == "__main__":
    main()
