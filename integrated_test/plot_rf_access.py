#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_speedup import (
    BENCHMARK_ORDER,
    detect_variant,
    geomean,
    is_base_dice_log,
    is_base_gpu_log,
    kernel_label_map,
)


KERNEL_NAME_RE = re.compile(r"kernel_name = (?P<kernel_name>\S+)")
LAUNCH_UID_RE = re.compile(r"kernel_launch_uid = (?P<uid>\d+)")
REG_TOTAL_RE = re.compile(r"gpgpu_n_tot_regfile_acesses = (?P<value>\d+)")
REG_READ_RE = re.compile(r"gpgpu_n_tot_regfile_read_acesses = (?P<value>\d+)")
REG_WRITE_RE = re.compile(r"gpgpu_n_tot_regfile_write_acesses = (?P<value>\d+)")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Rebuild per-launch RF access from raw DICE/GPU logs, compare "
            "DICE-full against RTX2060S, and generate a plot plus CSVs."
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
        help="Output CSV for per-launch regfile deltas",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output CSV for per-kernel normalized RF access",
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

    output_dir = args.output_dir if args.output_dir is not None else script_dir
    if args.launch_csv is None:
        args.launch_csv = output_dir / "rf_access_launch_deltas.csv"
    if args.summary_csv is None:
        args.summary_csv = output_dir / "rf_access_summary.csv"
    if args.png is None:
        args.png = output_dir / "rf_access_reduction.png"
    if args.pdf is None:
        args.pdf = output_dir / "rf_access_reduction.pdf"

    return args


def parse_kernel_entries(log_path: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    with log_path.open(errors="replace") as handle:
        for line in handle:
            match = KERNEL_NAME_RE.search(line)
            if match:
                if current is not None:
                    entries.append(current)
                current = {
                    "kernel_name": match.group("kernel_name"),
                }
                continue

            if current is None:
                continue

            if (match := LAUNCH_UID_RE.search(line)):
                current["kernel_launch_uid"] = int(match.group("uid"))
                continue
            if (match := REG_TOTAL_RE.search(line)):
                current["reg_total_cumulative"] = int(match.group("value"))
                continue
            if (match := REG_READ_RE.search(line)):
                current["reg_read_cumulative"] = int(match.group("value"))
                continue
            if (match := REG_WRITE_RE.search(line)):
                current["reg_write_cumulative"] = int(match.group("value"))
                continue

    if current is not None:
        entries.append(current)

    parsed: list[dict[str, object]] = []
    for entry in entries:
        required = (
            "kernel_name",
            "kernel_launch_uid",
            "reg_total_cumulative",
            "reg_read_cumulative",
            "reg_write_cumulative",
        )
        if not all(key in entry for key in required):
            missing = [key for key in required if key not in entry]
            raise ValueError(f"{log_path}: missing fields {missing} in one kernel entry")
        parsed.append(entry)

    parsed.sort(key=lambda item: int(item["kernel_launch_uid"]))
    return parsed


def compute_launch_deltas(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    prev_total = 0
    prev_read = 0
    prev_write = 0
    per_kernel_counts: dict[str, int] = defaultdict(int)
    launch_rows: list[dict[str, object]] = []

    for entry in entries:
        total = int(entry["reg_total_cumulative"])
        read = int(entry["reg_read_cumulative"])
        write = int(entry["reg_write_cumulative"])

        delta_total = total - prev_total
        delta_read = read - prev_read
        delta_write = write - prev_write

        if delta_total < 0 or delta_read < 0 or delta_write < 0:
            raise ValueError("Encountered negative regfile delta while parsing raw log")

        kernel_name = str(entry["kernel_name"])
        per_kernel_counts[kernel_name] += 1

        launch_rows.append(
            {
                "kernel_name": kernel_name,
                "kernel_launch_uid": int(entry["kernel_launch_uid"]),
                "gridDim": entry.get("gridDim", ""),
                "blockDim": entry.get("blockDim", ""),
                "run_index": per_kernel_counts[kernel_name],
                "reg_total_delta": delta_total,
                "reg_read_delta": delta_read,
                "reg_write_delta": delta_write,
            }
        )

        prev_total = total
        prev_read = read
        prev_write = write

    return launch_rows


def group_by_kernel(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["kernel_name"])].append(row)
    return grouped


def latest_gpu_log(app_dir: Path) -> Path:
    matches = [
        path
        for path in sorted(
            path for path in app_dir.glob("test_gpu_*.log") if not path.name.endswith("_asan.log")
        )
        if is_base_gpu_log(path)
    ]
    if not matches:
        raise FileNotFoundError(f"No base RTX2060S GPU log found in {app_dir}")
    return matches[-1]


def latest_dice_full_log(app_dir: Path) -> Path:
    matches = []
    for log_path in sorted(path for path in app_dir.glob("test_dice_*.log") if not path.name.endswith("_asan.log")):
        if is_base_dice_log(log_path) and detect_variant(log_path) == "DICE-full":
            matches.append(log_path)
    if not matches:
        raise FileNotFoundError(f"No base DICE-full log found in {app_dir}")
    return matches[-1]


def safe_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        raise ValueError("Encountered zero GPU regfile total while normalizing RF access")
    return numerator / denominator * 100.0


def scale_pair_to_total(
    first: float | None,
    second: float | None,
    total: float | None,
) -> tuple[float | None, float | None]:
    if first is None or second is None or total is None:
        return first, second
    pair_sum = first + second
    if pair_sum <= 0:
        return first, second
    scale = total / pair_sum
    return first * scale, second * scale


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
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    label_map, ordered_kernels = kernel_label_map(gpu_root)

    launch_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for app in BENCHMARK_ORDER:
        gpu_log = latest_gpu_log(gpu_root / app)
        dice_log = latest_dice_full_log(dice_root / app)

        gpu_launches = compute_launch_deltas(parse_kernel_entries(gpu_log))
        dice_launches = compute_launch_deltas(parse_kernel_entries(dice_log))

        gpu_by_kernel = group_by_kernel(gpu_launches)
        dice_by_kernel = group_by_kernel(dice_launches)

        for kernel_name, kernel_label in label_map[app].items():
            gpu_runs = gpu_by_kernel[kernel_name]
            dice_runs = dice_by_kernel[kernel_name]

            if len(gpu_runs) != len(dice_runs):
                raise ValueError(
                    f"{app} {kernel_name}: GPU has {len(gpu_runs)} runs but DICE-full has {len(dice_runs)}"
                )

            gpu_total_sum = 0
            gpu_read_sum = 0
            gpu_write_sum = 0
            dice_total_sum = 0
            dice_read_sum = 0
            dice_write_sum = 0

            for gpu_row, dice_row in zip(gpu_runs, dice_runs):
                gpu_total = int(gpu_row["reg_total_delta"])
                gpu_read = int(gpu_row["reg_read_delta"])
                gpu_write = int(gpu_row["reg_write_delta"])
                dice_total = int(dice_row["reg_total_delta"])
                dice_read = int(dice_row["reg_read_delta"])
                dice_write = int(dice_row["reg_write_delta"])

                rtx_read_pct = safe_pct(gpu_read, gpu_total)
                rtx_write_pct = safe_pct(gpu_write, gpu_total)
                dice_read_pct = safe_pct(dice_read, gpu_total)
                dice_write_pct = safe_pct(dice_write, gpu_total)
                dice_total_pct = safe_pct(dice_total, gpu_total)

                gpu_total_sum += gpu_total
                gpu_read_sum += gpu_read
                gpu_write_sum += gpu_write
                dice_total_sum += dice_total
                dice_read_sum += dice_read
                dice_write_sum += dice_write

                launch_rows.append(
                    {
                        "app": app,
                        "kernel_label": kernel_label,
                        "kernel_name": kernel_name,
                        "run_index": gpu_row["run_index"],
                        "gpu_kernel_launch_uid": gpu_row["kernel_launch_uid"],
                        "dice_kernel_launch_uid": dice_row["kernel_launch_uid"],
                        "gpu_reg_total_delta": gpu_total,
                        "gpu_reg_read_delta": gpu_read,
                        "gpu_reg_write_delta": gpu_write,
                        "dice_reg_total_delta": dice_total,
                        "dice_reg_read_delta": dice_read,
                        "dice_reg_write_delta": dice_write,
                        "rtx_read_pct": rtx_read_pct,
                        "rtx_write_pct": rtx_write_pct,
                        "dice_read_pct": dice_read_pct,
                        "dice_write_pct": dice_write_pct,
                        "dice_total_pct": dice_total_pct,
                        "dice_reduced_pct": 100.0 - dice_total_pct,
                    }
                )

            rtx_read_pct_kernel = safe_pct(gpu_read_sum, gpu_total_sum)
            rtx_write_pct_kernel = safe_pct(gpu_write_sum, gpu_total_sum)
            dice_read_pct_kernel = safe_pct(dice_read_sum, gpu_total_sum)
            dice_write_pct_kernel = safe_pct(dice_write_sum, gpu_total_sum)
            dice_total_pct_kernel = safe_pct(dice_total_sum, gpu_total_sum)
            dice_reduced_pct_kernel = 100.0 - dice_total_pct_kernel

            summary_rows.append(
                {
                    "app": app,
                    "kernel_label": kernel_label,
                    "kernel_name": kernel_name,
                    "run_count": len(gpu_runs),
                    "RTX2060S_read_pct": rtx_read_pct_kernel,
                    "RTX2060S_write_pct": rtx_write_pct_kernel,
                    "DICE_read_pct": dice_read_pct_kernel,
                    "DICE_write_pct": dice_write_pct_kernel,
                    "DICE_total_pct": dice_total_pct_kernel,
                    "DICE_reduced_pct": dice_reduced_pct_kernel,
                }
            )

    ordered_labels = [label for _, _, label in ordered_kernels]
    rows_by_label = {row["kernel_label"]: row for row in summary_rows}
    ordered_summary_rows = [rows_by_label[label] for label in ordered_labels]

    rtx_read_pct_geomean, rtx_write_pct_geomean = scale_pair_to_total(
        geomean([row["RTX2060S_read_pct"] for row in ordered_summary_rows]),
        geomean([row["RTX2060S_write_pct"] for row in ordered_summary_rows]),
        100.0,
    )
    dice_total_pct_geomean = geomean([row["DICE_total_pct"] for row in ordered_summary_rows])
    dice_read_pct_geomean, dice_write_pct_geomean = scale_pair_to_total(
        geomean([row["DICE_read_pct"] for row in ordered_summary_rows]),
        geomean([row["DICE_write_pct"] for row in ordered_summary_rows]),
        dice_total_pct_geomean,
    )

    geomean_row = {
        "app": "OVERALL",
        "kernel_label": "GEOMEAN",
        "kernel_name": "GEOMEAN",
        "run_count": len(ordered_summary_rows),
        "RTX2060S_read_pct": rtx_read_pct_geomean,
        "RTX2060S_write_pct": rtx_write_pct_geomean,
        "DICE_read_pct": dice_read_pct_geomean,
        "DICE_write_pct": dice_write_pct_geomean,
        "DICE_total_pct": dice_total_pct_geomean,
    }
    geomean_row["DICE_reduced_pct"] = 100.0 - geomean_row["DICE_total_pct"]
    ordered_summary_rows.append(geomean_row)

    return launch_rows, ordered_summary_rows, ordered_labels + ["GEOMEAN"]


def make_plot(summary_rows: list[dict[str, object]], labels: list[str], png_path: Path, pdf_path: Path) -> None:
    plot_lookup = {row["kernel_label"]: row for row in summary_rows}

    x_base = np.arange(len(labels) - 1, dtype=float)
    gap_size = 0.5
    x = np.append(x_base, x_base[-1] + 1 + gap_size)
    width = 0.35

    rtx_read = np.array([plot_lookup[label]["RTX2060S_read_pct"] for label in labels], dtype=float)
    rtx_write = np.array([plot_lookup[label]["RTX2060S_write_pct"] for label in labels], dtype=float)
    dice_read = np.array([plot_lookup[label]["DICE_read_pct"] for label in labels], dtype=float)
    dice_write = np.array([plot_lookup[label]["DICE_write_pct"] for label in labels], dtype=float)
    dice_reduced = np.array([plot_lookup[label]["DICE_reduced_pct"] for label in labels], dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 2.6))

    read_color = "#7095c1"
    write_color = "#f5a657"
    read_color_light = "#b3c8e0"
    write_color_light = "#fac894"

    bars1 = ax.bar(
        x - width / 2,
        rtx_read,
        width,
        color=read_color,
        edgecolor="black",
        linewidth=0.8,
    )
    bars2 = ax.bar(
        x - width / 2,
        rtx_write,
        width,
        bottom=rtx_read,
        color=write_color,
        edgecolor="black",
        linewidth=0.8,
    )
    bars3 = ax.bar(
        x + width / 2,
        dice_read,
        width,
        color=read_color_light,
        edgecolor="black",
        linewidth=0.8,
    )
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

    ax.set_ylabel("Normalized RF Access (%)\n(vs. RTX2060S)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.set_ylim(0, 102)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_xlim(-0.5, x[-1] + 0.5)

    legend_elements = [
        Patch(facecolor=read_color, edgecolor="black", hatch="///", label="RTX2060S Read"),
        Patch(facecolor=write_color, edgecolor="black", hatch="---", label="RTX2060S Write"),
        Patch(facecolor=read_color_light, edgecolor="black", hatch="///", label="DICE Read"),
        Patch(facecolor=write_color_light, edgecolor="black", hatch="---", label="DICE Write"),
        Patch(facecolor="white", edgecolor="black", hatch="...", label="Reduced by DICE"),
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
        "gpu_kernel_launch_uid",
        "dice_kernel_launch_uid",
        "gpu_reg_total_delta",
        "gpu_reg_read_delta",
        "gpu_reg_write_delta",
        "dice_reg_total_delta",
        "dice_reg_read_delta",
        "dice_reg_write_delta",
        "rtx_read_pct",
        "rtx_write_pct",
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
        "RTX2060S_read_pct",
        "RTX2060S_write_pct",
        "DICE_read_pct",
        "DICE_write_pct",
        "DICE_total_pct",
        "DICE_reduced_pct",
    ]

    write_csv(args.launch_csv, launch_rows, launch_fieldnames)
    write_csv(args.summary_csv, summary_rows, summary_fieldnames)
    make_plot(summary_rows, labels, args.png, args.pdf)

    print(f"Wrote {len(launch_rows)} launch rows to {args.launch_csv}")
    print(f"Wrote {len(summary_rows)} summary rows to {args.summary_csv}")
    print(f"Wrote plot to {args.png}")
    print(f"Wrote plot to {args.pdf}")
    print("\nRegister Access Reduction by DICE:")
    for row in summary_rows:
        print(
            f"{row['kernel_label']}: {row['DICE_reduced_pct']:.1f}% reduction "
            f"(DICE uses {row['DICE_total_pct']:.1f}% of RTX2060S)"
        )


if __name__ == "__main__":
    main()
