#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


VARIANT_ORDER = [
    "DICE-naive",
    "DICE-naive+unroll",
    "DICE-naive+tmcu",
    "DICE-full",
]

BENCHMARK_ORDER = [
    "nn_cuda",
    "bfs",
    "backprop",
    "streamcluster",
    "gaussian",
    "hotspot",
    "pathfinder",
]

VARIANT_MAP = {
    (0, 0): "DICE-naive",
    (1, 0): "DICE-naive+unroll",
    (0, 1): "DICE-naive+tmcu",
    (1, 1): "DICE-full",
}

PLOT_SERIES = [
    ("RTX2060S", None),
    ("DICE-naive", "DICE-naive"),
    ("DICE-naive+unroll", "DICE-naive+unroll"),
    ("DICE-naive+TMCU", "DICE-naive+tmcu"),
    ("DICE", "DICE-full"),
]

PLOT_COLORS = ["#91b2d4", "#f7bc81", "#ee9a9c", "#b4d9d8", "#9dc994"]
PLOT_HATCHES = ["", "///", "---", "\\\\\\", "xxxx"]

UNROLL_RE = re.compile(r"-dice_enable_unrolling\s+([01])")
TMCU_RE = re.compile(r"-dice_ldst_unit_enable_temporal_coalescing\s+([01])")
N_CLUSTERS_RE = re.compile(r"^-gpgpu_n_clusters\s+(\d+)", re.MULTILINE)
PIPELINE_RE = re.compile(r"^-gpgpu_shader_core_pipeline\s+(\S+)", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Collect DICE and GPU summary data, compute per-kernel speedups, "
            "and generate CSV/plot outputs."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated CSV/plot outputs",
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
        "--launch-csv",
        type=Path,
        default=None,
        help="Output CSV with per-launch metrics",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output CSV with per-kernel and overall geomean speedups",
    )
    parser.add_argument(
        "--plot",
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

    output_dir = args.output_dir if args.output_dir is not None else script_dir
    if args.launch_csv is None:
        args.launch_csv = output_dir / "dice_gpu_launch_metrics.csv"
    if args.summary_csv is None:
        args.summary_csv = output_dir / "dice_gpu_speedup_summary.csv"
    if args.plot is None:
        args.plot = output_dir / "dice_gpu_speedup.png"
    if args.pdf is None:
        args.pdf = output_dir / "dice_gpu_speedup.pdf"

    return args


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            # Some summary CSVs contain a partially written row with the
            # kernel name but no launch UID/counters. Skip those here so all
            # downstream grouping logic sees only complete kernel records.
            if not (row.get("date_time") or "").strip():
                continue
            if not (row.get("kernel_name") or "").strip():
                continue
            if "kernel_launch_uid" in row and not (row.get("kernel_launch_uid") or "").strip():
                continue
            rows.append(row)
    return rows


def as_int(value: str) -> int | None:
    value = value.strip()
    return int(value) if value else None


def safe_ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def geomean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None and value > 0]
    if not filtered:
        return None
    return math.exp(sum(math.log(value) for value in filtered) / len(filtered))


def detect_variant(log_path: Path) -> str:
    unroll = None
    tmcu = None

    with log_path.open(errors="replace") as handle:
        for line in handle:
            if unroll is None:
                match = UNROLL_RE.search(line)
                if match:
                    unroll = int(match.group(1))
            if tmcu is None:
                match = TMCU_RE.search(line)
                if match:
                    tmcu = int(match.group(1))
            if unroll is not None and tmcu is not None:
                break

    if unroll is None or tmcu is None:
        raise ValueError(f"Could not determine DICE variant from {log_path}")

    return VARIANT_MAP[(unroll, tmcu)]


def parse_log_profile(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(errors="replace")

    clusters_match = N_CLUSTERS_RE.search(text)
    pipeline_match = PIPELINE_RE.search(text)

    if clusters_match is None:
        raise ValueError(f"Could not determine gpgpu_n_clusters from {log_path}")
    if pipeline_match is None:
        raise ValueError(f"Could not determine gpgpu_shader_core_pipeline from {log_path}")

    return {
        "n_clusters": int(clusters_match.group(1)),
        "shader_core_pipeline": pipeline_match.group(1),
    }


def is_base_gpu_log(log_path: Path) -> bool:
    profile = parse_log_profile(log_path)
    return profile["n_clusters"] == 34


def is_base_dice_log(log_path: Path) -> bool:
    profile = parse_log_profile(log_path)
    return profile["n_clusters"] == 34 and profile["shader_core_pipeline"] == "512:32"


def date_log_pairs(rows: list[dict[str, str]], app_dir: Path, prefix: str) -> list[tuple[str, Path]]:
    result_dates = sorted({row["date_time"] for row in rows})
    log_files = sorted(
        path for path in app_dir.glob(f"{prefix}_*.log") if not path.name.endswith("_asan.log")
    )

    if len(result_dates) != len(log_files):
        raise ValueError(
            f"{app_dir.name}: found {len(result_dates)} date groups in result.csv but "
            f"{len(log_files)} raw {prefix} logs"
        )

    return list(zip(result_dates, log_files))


def build_dice_date_variant_map(app_dir: Path, dice_rows: list[dict[str, str]]) -> dict[str, str]:
    return {date_time: detect_variant(log_path) for date_time, log_path in date_log_pairs(dice_rows, app_dir, "test_dice")}


def latest_base_gpu_rows(app: str, gpu_root: Path) -> tuple[str, list[dict[str, str]]]:
    rows = read_csv_rows(gpu_root / app / f"{app}.result.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["date_time"]].append(row)

    matches = [
        (date_time, grouped[date_time])
        for date_time, log_path in date_log_pairs(rows, gpu_root / app, "test_gpu")
        if is_base_gpu_log(log_path)
    ]
    if not matches:
        raise ValueError(f"{app}: no base RTX2060S GPU run found")
    return matches[-1]


def latest_base_dice_dates_by_variant(app: str, dice_root: Path) -> dict[str, str]:
    rows = read_csv_rows(dice_root / app / f"{app}.result.csv")
    selected: dict[str, str] = {}

    for date_time, log_path in date_log_pairs(rows, dice_root / app, "test_dice"):
        if not is_base_dice_log(log_path):
            continue
        selected[detect_variant(log_path)] = date_time

    missing = [variant for variant in VARIANT_ORDER if variant not in selected]
    if missing:
        raise ValueError(f"{app}: missing base DICE runs for variants {missing}")

    return selected


def latest_rows(rows: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["date_time"]].append(row)
    latest = max(grouped)
    return latest, grouped[latest]


def grouped_kernel_runs(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: int(item["kernel_launch_uid"])):
        grouped[row["kernel_name"]].append(row)
    return grouped


def kernel_label_map(gpu_root: Path) -> tuple[dict[str, dict[str, str]], list[tuple[str, str, str]]]:
    per_app: dict[str, dict[str, str]] = {}
    ordered: list[tuple[str, str, str]] = []

    for app in BENCHMARK_ORDER:
        _, latest = latest_base_gpu_rows(app, gpu_root)
        kernels_in_order: list[str] = []
        for row in sorted(latest, key=lambda item: int(item["kernel_launch_uid"])):
            kernel = row["kernel_name"]
            if kernel not in kernels_in_order:
                kernels_in_order.append(kernel)

        per_app[app] = {}
        multi_kernel = len(kernels_in_order) > 1
        for idx, kernel in enumerate(kernels_in_order, start=1):
            label = f"{app}-{idx}" if multi_kernel else app
            per_app[app][kernel] = label
            ordered.append((app, kernel, label))

    return per_app, ordered


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_outputs(
    dice_root: Path,
    gpu_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[tuple[str, str, str]]]:
    label_map, ordered_kernels = kernel_label_map(gpu_root)

    launch_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    variant_speedups: dict[str, list[float]] = defaultdict(list)

    for app in BENCHMARK_ORDER:
        gpu_date, gpu_latest_rows = latest_base_gpu_rows(app, gpu_root)
        gpu_by_kernel = grouped_kernel_runs(gpu_latest_rows)

        dice_csv_path = dice_root / app / f"{app}.result.csv"
        dice_rows = read_csv_rows(dice_csv_path)
        dice_date_to_variant = build_dice_date_variant_map(dice_root / app, dice_rows)
        selected_dice_dates = latest_base_dice_dates_by_variant(app, dice_root)

        dice_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in dice_rows:
            dice_grouped[row["date_time"]].append(row)

        for variant in VARIANT_ORDER:
            date_time = selected_dice_dates[variant]
            dice_by_kernel = grouped_kernel_runs(dice_grouped[date_time])

            for kernel_name, gpu_kernel_rows in gpu_by_kernel.items():
                dice_kernel_rows = dice_by_kernel.get(kernel_name, [])
                if len(dice_kernel_rows) != len(gpu_kernel_rows):
                    raise ValueError(
                        f"{app} {variant} {kernel_name}: DICE has {len(dice_kernel_rows)} runs, "
                        f"GPU has {len(gpu_kernel_rows)} runs"
                    )

                kernel_label = label_map[app][kernel_name]
                kernel_speedups: list[float | None] = []
                kernel_read_ratios: list[float | None] = []
                kernel_write_ratios: list[float | None] = []

                for run_index, (dice_row, gpu_row) in enumerate(
                    zip(dice_kernel_rows, gpu_kernel_rows), start=1
                ):
                    dice_cycle = as_int(dice_row["gpu_sim_cycle"])
                    gpu_cycle = as_int(gpu_row["gpu_sim_cycle"])
                    dice_read = as_int(dice_row["gpgpu_n_tot_regfile_read_acesses"])
                    gpu_read = as_int(gpu_row["gpgpu_n_tot_regfile_read_acesses"])
                    dice_write = as_int(dice_row["gpgpu_n_tot_regfile_write_acesses"])
                    gpu_write = as_int(gpu_row["gpgpu_n_tot_regfile_write_acesses"])
                    dice_total = as_int(dice_row["gpgpu_n_tot_regfile_acesses"])
                    gpu_total = as_int(gpu_row["gpgpu_n_tot_regfile_acesses"])

                    cycle_speedup = safe_ratio(gpu_cycle, dice_cycle)
                    read_ratio = safe_ratio(dice_read, gpu_read)
                    write_ratio = safe_ratio(dice_write, gpu_write)

                    kernel_speedups.append(cycle_speedup)
                    kernel_read_ratios.append(read_ratio)
                    kernel_write_ratios.append(write_ratio)

                    launch_rows.append(
                        {
                            "app": app,
                            "kernel_label": kernel_label,
                            "kernel_name": kernel_name,
                            "variant": variant,
                            "dice_date_time": date_time,
                            "gpu_date_time": gpu_date,
                            "run_index": run_index,
                            "dice_kernel_launch_uid": dice_row["kernel_launch_uid"],
                            "gpu_kernel_launch_uid": gpu_row["kernel_launch_uid"],
                            "dice_gpu_sim_cycle": dice_cycle,
                            "gpu_gpu_sim_cycle": gpu_cycle,
                            "cycle_speedup_vs_gpu": cycle_speedup,
                            "dice_regfile_read_accesses": dice_read,
                            "gpu_regfile_read_accesses": gpu_read,
                            "regfile_read_ratio_vs_gpu": read_ratio,
                            "dice_regfile_write_accesses": dice_write,
                            "gpu_regfile_write_accesses": gpu_write,
                            "regfile_write_ratio_vs_gpu": write_ratio,
                            "dice_regfile_total_accesses": dice_total,
                            "gpu_regfile_total_accesses": gpu_total,
                        }
                    )

                kernel_geomean_speedup = geomean(kernel_speedups)
                kernel_geomean_read_ratio = geomean(kernel_read_ratios)
                kernel_geomean_write_ratio = geomean(kernel_write_ratios)

                if kernel_geomean_speedup is not None:
                    variant_speedups[variant].append(kernel_geomean_speedup)

                summary_rows.append(
                    {
                        "app": app,
                        "kernel_label": kernel_label,
                        "kernel_name": kernel_name,
                        "variant": variant,
                        "run_count": len(kernel_speedups),
                        "geomean_speedup_vs_gpu": kernel_geomean_speedup,
                        "geomean_regfile_read_ratio_vs_gpu": kernel_geomean_read_ratio,
                        "geomean_regfile_write_ratio_vs_gpu": kernel_geomean_write_ratio,
                    }
                )

    for variant in VARIANT_ORDER:
        summary_rows.append(
            {
                "app": "OVERALL",
                "kernel_label": "GeoMean",
                "kernel_name": "GeoMean",
                "variant": variant,
                "run_count": len(variant_speedups[variant]),
                "geomean_speedup_vs_gpu": geomean(variant_speedups[variant]),
                "geomean_regfile_read_ratio_vs_gpu": "",
                "geomean_regfile_write_ratio_vs_gpu": "",
            }
        )

    return launch_rows, summary_rows, ordered_kernels


def make_plot(
    summary_rows: list[dict[str, object]],
    ordered_kernels: list[tuple[str, str, str]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    by_key = {
        (row["kernel_label"], row["variant"]): row["geomean_speedup_vs_gpu"]
        for row in summary_rows
    }
    overall = {
        row["variant"]: row["geomean_speedup_vs_gpu"]
        for row in summary_rows
        if row["app"] == "OVERALL"
    }

    labels = [label for _, _, label in ordered_kernels] + ["GeoMean"]
    gap_size = 0.5
    positions = list(np.arange(len(labels) - 1, dtype=float)) + [len(labels) - 1 + gap_size]

    group_width = 0.75
    bar_width = group_width / len(PLOT_SERIES)

    fig, ax = plt.subplots(figsize=(10.5, 3.4))

    containers = []
    for series_idx, ((series_label, variant), color, hatch) in enumerate(
        zip(PLOT_SERIES, PLOT_COLORS, PLOT_HATCHES)
    ):
        if variant is None:
            values = [1.0] * len(labels)
        else:
            values = [
                by_key.get((label, variant), np.nan)
                for _, _, label in ordered_kernels
            ] + [overall.get(variant, np.nan)]

        x_positions = [
            pos - group_width / 2 + series_idx * bar_width
            for pos in positions
        ]
        bars = ax.bar(
            x_positions,
            values,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
            label=series_label,
        )
        containers.append((series_label, bars))

    separator_x = (positions[-2] + positions[-1]) / 2 - gap_size / 2
    ax.axvline(x=separator_x, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5)

    ax.set_ylabel("Normalized Speedup (x)\n(vs. RTX2060S)", fontsize=15)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_xlim(-0.5, positions[-1] + 0.5)

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="black",
            hatch=hatch,
            linewidth=0.8,
            label=series_label,
        )
        for (series_label, _), color, hatch in zip(PLOT_SERIES, PLOT_COLORS, PLOT_HATCHES)
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.26),
        ncol=5,
        frameon=True,
        fancybox=False,
        columnspacing=0.5,
        handlelength=1.5,
        handletextpad=0.3,
    )

    ax.text(
        0.01,
        0.92,
        "*: geometric mean of multiple kernel runs in a program",
        transform=ax.transAxes,
        fontsize=12,
        style="italic",
        ha="left",
        va="bottom",
    )

    for series_label, bars in containers:
        if series_label != "DICE":
            continue
        geomean_bar = bars[-1]
        height = geomean_bar.get_height()
        if height is None or np.isnan(height):
            break
        ax.annotate(
            f"{height:.2f}",
            xy=(geomean_bar.get_x() + geomean_bar.get_width() / 2 + 0.2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )
        break

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    launch_rows, summary_rows, ordered_kernels = build_outputs(args.dice_root, args.gpu_root)
    missing_regfile_rows = sum(
        1
        for row in launch_rows
        if row["dice_regfile_read_accesses"] is None or row["dice_regfile_write_accesses"] is None
    )

    launch_fieldnames = [
        "app",
        "kernel_label",
        "kernel_name",
        "variant",
        "dice_date_time",
        "gpu_date_time",
        "run_index",
        "dice_kernel_launch_uid",
        "gpu_kernel_launch_uid",
        "dice_gpu_sim_cycle",
        "gpu_gpu_sim_cycle",
        "cycle_speedup_vs_gpu",
        "dice_regfile_read_accesses",
        "gpu_regfile_read_accesses",
        "regfile_read_ratio_vs_gpu",
        "dice_regfile_write_accesses",
        "gpu_regfile_write_accesses",
        "regfile_write_ratio_vs_gpu",
        "dice_regfile_total_accesses",
        "gpu_regfile_total_accesses",
    ]

    summary_fieldnames = [
        "app",
        "kernel_label",
        "kernel_name",
        "variant",
        "run_count",
        "geomean_speedup_vs_gpu",
        "geomean_regfile_read_ratio_vs_gpu",
        "geomean_regfile_write_ratio_vs_gpu",
    ]

    write_csv(args.launch_csv, launch_rows, launch_fieldnames)
    write_csv(args.summary_csv, summary_rows, summary_fieldnames)
    make_plot(summary_rows, ordered_kernels, args.plot, args.pdf)

    print(f"Wrote {len(launch_rows)} launch rows to {args.launch_csv}")
    print(f"Wrote {len(summary_rows)} summary rows to {args.summary_csv}")
    print(f"Wrote plot to {args.plot}")
    print(f"Wrote plot to {args.pdf}")
    if missing_regfile_rows:
        print(
            "Warning: "
            f"{missing_regfile_rows} launch row(s) had missing DICE regfile read/write counters; "
            "the detailed CSV keeps those fields blank and the geomean ratio skips them."
        )


if __name__ == "__main__":
    main()
