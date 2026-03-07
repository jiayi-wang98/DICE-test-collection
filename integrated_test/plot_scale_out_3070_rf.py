#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_speedup import BENCHMARK_ORDER, as_int, geomean, read_csv_rows, write_csv


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    output_dir = script_dir / "generated_scale_out"

    parser = argparse.ArgumentParser(
        description=(
            "Collect scale-out DICE/GPU summary data, compute normalized RF access "
            "for DICE-3070 vs RTX3070, and generate CSV/plot outputs."
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
    parser.add_argument("--launch-csv", type=Path, default=None, help="Output CSV for per-launch RF deltas")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Output CSV for per-kernel RF summary")
    parser.add_argument("--png", type=Path, default=None, help="Output PNG plot path")
    parser.add_argument("--pdf", type=Path, default=None, help="Output PDF plot path")
    args = parser.parse_args()

    if args.launch_csv is None:
        args.launch_csv = args.output_dir / "scale_out_3070_rf_launch_metrics.csv"
    if args.summary_csv is None:
        args.summary_csv = args.output_dir / "scale_out_3070_rf_summary.csv"
    if args.png is None:
        args.png = args.output_dir / "scale_out_3070_rf_reduction.png"
    if args.pdf is None:
        args.pdf = args.output_dir / "scale_out_3070_rf_reduction.pdf"
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


def safe_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        raise ValueError("Encountered zero GPU regfile total while normalizing RF access")
    return numerator / denominator * 100.0


def scale_pair_to_total(first: float | None, second: float | None, total: float | None) -> tuple[float | None, float | None]:
    if first is None or second is None or total is None:
        return first, second
    pair_sum = first + second
    if pair_sum <= 0:
        return first, second
    scale = total / pair_sum
    return first * scale, second * scale


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


def build_delta_rows(app: str, kernel_name: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    sorted_rows = sorted(rows, key=lambda item: int(item["kernel_launch_uid"]))
    skipped_uids = skip_launch_uids(app, kernel_name, sorted_rows)

    prev_total = 0
    prev_read = 0
    prev_write = 0
    run_index = 0
    delta_rows: list[dict[str, object]] = []

    for row in sorted_rows:
        launch_uid = int(row["kernel_launch_uid"])
        total = as_int(row["gpgpu_n_tot_regfile_acesses"])
        read = as_int(row["gpgpu_n_tot_regfile_read_acesses"])
        write = as_int(row["gpgpu_n_tot_regfile_write_acesses"])
        if total is None or read is None or write is None:
            raise ValueError(f"{app} {kernel_name} launch {launch_uid}: missing cumulative regfile counters")

        delta_total = total - prev_total
        delta_read = read - prev_read
        delta_write = write - prev_write
        if delta_total < 0 or delta_read < 0 or delta_write < 0:
            raise ValueError(f"{app} {kernel_name} launch {launch_uid}: encountered negative regfile delta")

        if launch_uid not in skipped_uids:
            run_index += 1
            delta_rows.append(
                {
                    "kernel_name": kernel_name,
                    "kernel_launch_uid": launch_uid,
                    "run_index": run_index,
                    "reg_total_delta": delta_total,
                    "reg_read_delta": delta_read,
                    "reg_write_delta": delta_write,
                }
            )

        prev_total = total
        prev_read = read
        prev_write = write

    return delta_rows


def rows_by_launch_uid(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(row["kernel_launch_uid"]): row for row in rows}


def build_outputs(
    dice_root: Path,
    gpu_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
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
            gpu_delta_rows = build_delta_rows(app, kernel_name, gpu_by_kernel[kernel_name])
            dice_delta_rows = build_delta_rows(app, kernel_name, dice_by_kernel[kernel_name])

            gpu_by_uid = rows_by_launch_uid(gpu_delta_rows)
            dice_by_uid = rows_by_launch_uid(dice_delta_rows)
            common_uids = sorted(set(gpu_by_uid) & set(dice_by_uid))
            if not common_uids:
                raise ValueError(f"{app} {kernel_name}: no common launch IDs across selected RTX3070 groups")

            max_count = max(len(gpu_by_uid), len(dice_by_uid))
            if len(common_uids) != max_count:
                print(
                    f"Warning: {app} {kernel_name}: using {len(common_uids)} common launch IDs "
                    f"(gpu3070={len(gpu_by_uid)}, dice3070={len(dice_by_uid)})"
                )

            gpu_total_sum = 0
            gpu_read_sum = 0
            gpu_write_sum = 0
            dice_total_sum = 0
            dice_read_sum = 0
            dice_write_sum = 0

            for run_index, launch_uid in enumerate(common_uids, start=1):
                gpu_row = gpu_by_uid[launch_uid]
                dice_row = dice_by_uid[launch_uid]

                gpu_total = int(gpu_row["reg_total_delta"])
                gpu_read = int(gpu_row["reg_read_delta"])
                gpu_write = int(gpu_row["reg_write_delta"])
                dice_total = int(dice_row["reg_total_delta"])
                dice_read = int(dice_row["reg_read_delta"])
                dice_write = int(dice_row["reg_write_delta"])

                gpu_total_sum += gpu_total
                gpu_read_sum += gpu_read
                gpu_write_sum += gpu_write
                dice_total_sum += dice_total
                dice_read_sum += dice_read
                dice_write_sum += dice_write

                gpu_read_pct = safe_pct(gpu_read, gpu_total)
                gpu_write_pct = safe_pct(gpu_write, gpu_total)
                dice_read_pct = safe_pct(dice_read, gpu_total)
                dice_write_pct = safe_pct(dice_write, gpu_total)
                dice_total_pct = safe_pct(dice_total, gpu_total)

                launch_rows.append(
                    {
                        "app": app,
                        "kernel_label": kernel_label,
                        "kernel_name": kernel_name,
                        "run_index": run_index,
                        "matched_kernel_launch_uid": launch_uid,
                        "gpu_rtx3070_date_time": gpu_date,
                        "dice_rtx3070_date_time": dice_date,
                        "gpu_reg_total_delta": gpu_total,
                        "gpu_reg_read_delta": gpu_read,
                        "gpu_reg_write_delta": gpu_write,
                        "dice_reg_total_delta": dice_total,
                        "dice_reg_read_delta": dice_read,
                        "dice_reg_write_delta": dice_write,
                        "gpu_read_pct": gpu_read_pct,
                        "gpu_write_pct": gpu_write_pct,
                        "dice_read_pct": dice_read_pct,
                        "dice_write_pct": dice_write_pct,
                        "dice_total_pct": dice_total_pct,
                        "dice_reduced_pct": 100.0 - dice_total_pct,
                    }
                )

            gpu_read_pct_kernel = safe_pct(gpu_read_sum, gpu_total_sum)
            gpu_write_pct_kernel = safe_pct(gpu_write_sum, gpu_total_sum)
            dice_read_pct_kernel = safe_pct(dice_read_sum, gpu_total_sum)
            dice_write_pct_kernel = safe_pct(dice_write_sum, gpu_total_sum)
            dice_total_pct_kernel = safe_pct(dice_total_sum, gpu_total_sum)

            summary_rows.append(
                {
                    "app": app,
                    "kernel_label": kernel_label,
                    "kernel_name": kernel_name,
                    "run_count": len(common_uids),
                    "RTX3070_read_pct": gpu_read_pct_kernel,
                    "RTX3070_write_pct": gpu_write_pct_kernel,
                    "DICE3070_read_pct": dice_read_pct_kernel,
                    "DICE3070_write_pct": dice_write_pct_kernel,
                    "DICE3070_total_pct": dice_total_pct_kernel,
                    "DICE3070_reduced_pct": 100.0 - dice_total_pct_kernel,
                }
            )

    ordered_labels = [label for _, _, label in ordered_kernels]
    rows_by_label = {str(row["kernel_label"]): row for row in summary_rows}
    ordered_summary_rows = [rows_by_label[label] for label in ordered_labels]

    gpu_read_geomean, gpu_write_geomean = scale_pair_to_total(
        geomean([row["RTX3070_read_pct"] for row in ordered_summary_rows]),
        geomean([row["RTX3070_write_pct"] for row in ordered_summary_rows]),
        100.0,
    )
    dice_total_geomean = geomean([row["DICE3070_total_pct"] for row in ordered_summary_rows])
    dice_read_geomean, dice_write_geomean = scale_pair_to_total(
        geomean([row["DICE3070_read_pct"] for row in ordered_summary_rows]),
        geomean([row["DICE3070_write_pct"] for row in ordered_summary_rows]),
        dice_total_geomean,
    )

    ordered_summary_rows.append(
        {
            "app": "OVERALL",
            "kernel_label": "GEOMEAN",
            "kernel_name": "GEOMEAN",
            "run_count": len(ordered_summary_rows),
            "RTX3070_read_pct": gpu_read_geomean,
            "RTX3070_write_pct": gpu_write_geomean,
            "DICE3070_read_pct": dice_read_geomean,
            "DICE3070_write_pct": dice_write_geomean,
            "DICE3070_total_pct": dice_total_geomean,
            "DICE3070_reduced_pct": 100.0 - dice_total_geomean,
        }
    )

    return launch_rows, ordered_summary_rows, ordered_labels + ["GEOMEAN"]


def make_plot(summary_rows: list[dict[str, object]], labels: list[str], png_path: Path, pdf_path: Path) -> None:
    plot_lookup = {str(row["kernel_label"]): row for row in summary_rows}

    x_base = np.arange(len(labels) - 1, dtype=float)
    gap_size = 0.5
    x = np.append(x_base, x_base[-1] + 1 + gap_size)
    width = 0.35

    gpu_read = np.array([plot_lookup[label]["RTX3070_read_pct"] for label in labels], dtype=float)
    gpu_write = np.array([plot_lookup[label]["RTX3070_write_pct"] for label in labels], dtype=float)
    dice_read = np.array([plot_lookup[label]["DICE3070_read_pct"] for label in labels], dtype=float)
    dice_write = np.array([plot_lookup[label]["DICE3070_write_pct"] for label in labels], dtype=float)
    dice_reduced = np.array([plot_lookup[label]["DICE3070_reduced_pct"] for label in labels], dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 2.6))

    read_color = "#7095c1"
    write_color = "#f5a657"
    read_color_light = "#b3c8e0"
    write_color_light = "#fac894"

    bars1 = ax.bar(x - width / 2, gpu_read, width, color=read_color, edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(
        x - width / 2,
        gpu_write,
        width,
        bottom=gpu_read,
        color=write_color,
        edgecolor="black",
        linewidth=0.8,
    )
    bars3 = ax.bar(x + width / 2, dice_read, width, color=read_color_light, edgecolor="black", linewidth=0.8)
    bars4 = ax.bar(
        x + width / 2,
        dice_reduced,
        width,
        bottom=dice_read,
        color="white",
        edgecolor="black",
        linewidth=0.8,
    )
    bars5 = ax.bar(
        x + width / 2,
        dice_write,
        width,
        bottom=dice_read + dice_reduced,
        color=write_color_light,
        edgecolor="black",
        linewidth=0.8,
    )

    for bar in bars1:
        bar.set_hatch("///")
    for bar in bars2:
        bar.set_hatch("---")
    for bar in bars3:
        bar.set_hatch("///")
    for bar in bars4:
        bar.set_hatch("...")
    for bar in bars5:
        bar.set_hatch("---")

    for idx, label in enumerate(labels):
        reduced_height = dice_reduced[idx]
        if reduced_height <= 5:
            continue
        ax.text(
            x[idx] + width / 2,
            dice_read[idx] + reduced_height / 2,
            f"-{reduced_height:.0f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black",
            rotation=90,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )

    separator_x = (x[-2] + x[-1]) / 2 - gap_size / 2
    ax.axvline(x=separator_x, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(y=100, color="black", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_ylabel("Normalized RF Access (%)\n(vs. RTX3070)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.set_ylim(0, 102)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_xlim(-0.5, x[-1] + 0.5)

    legend_elements = [
        Patch(facecolor=read_color, edgecolor="black", hatch="///", label="RTX3070 Read"),
        Patch(facecolor=write_color, edgecolor="black", hatch="---", label="RTX3070 Write"),
        Patch(facecolor=read_color_light, edgecolor="black", hatch="///", label="DICE-3070 Read"),
        Patch(facecolor=write_color_light, edgecolor="black", hatch="---", label="DICE-3070 Write"),
        Patch(facecolor="white", edgecolor="black", hatch="...", label="Reduced by DICE-3070"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        fontsize=11,
        ncol=3,
        bbox_to_anchor=(0.5, 1.42),
        frameon=True,
    )

    ax.text(
        0.02,
        1.01,
        "*:geometric mean of multiple kernel runs",
        transform=ax.transAxes,
        fontsize=12,
        style="italic",
        ha="left",
        va="bottom",
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    launch_rows, summary_rows, labels = build_outputs(args.dice_root, args.gpu_root)

    launch_fieldnames = [
        "app",
        "kernel_label",
        "kernel_name",
        "run_index",
        "matched_kernel_launch_uid",
        "gpu_rtx3070_date_time",
        "dice_rtx3070_date_time",
        "gpu_reg_total_delta",
        "gpu_reg_read_delta",
        "gpu_reg_write_delta",
        "dice_reg_total_delta",
        "dice_reg_read_delta",
        "dice_reg_write_delta",
        "gpu_read_pct",
        "gpu_write_pct",
        "dice_read_pct",
        "dice_write_pct",
        "dice_total_pct",
        "dice_reduced_pct",
    ]
    summary_fieldnames = [
        "app",
        "kernel_label",
        "kernel_name",
        "run_count",
        "RTX3070_read_pct",
        "RTX3070_write_pct",
        "DICE3070_read_pct",
        "DICE3070_write_pct",
        "DICE3070_total_pct",
        "DICE3070_reduced_pct",
    ]

    args.launch_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.launch_csv, launch_rows, launch_fieldnames)
    write_csv(args.summary_csv, summary_rows, summary_fieldnames)
    make_plot(summary_rows, labels, args.png, args.pdf)

    print(f"Wrote {len(launch_rows)} launch rows to {args.launch_csv}")
    print(f"Wrote {len(summary_rows)} summary rows to {args.summary_csv}")
    print(f"Wrote plot to {args.png}")
    print(f"Wrote plot to {args.pdf}")


if __name__ == "__main__":
    main()
