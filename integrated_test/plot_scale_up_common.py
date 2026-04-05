#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from plot_speedup import BENCHMARK_ORDER, detect_variant, geomean, grouped_kernel_runs, read_csv_rows


DICE_BASE_MODE = "DICE"
DICE_U_MODE = "DICE-U"

BASE_CORE_RE = re.compile(r"-gpgpu_n_cores_per_cluster\s+4\b")
BASE_PIPELINE_RE = re.compile(r"-gpgpu_shader_core_pipeline\s+512:32\b")
U_CORE_RE = re.compile(r"-gpgpu_n_cores_per_cluster\s+2\b")
U_PIPELINE_RE = re.compile(r"-gpgpu_shader_core_pipeline\s+1024:32\b")


def detect_scale_mode(log_path: Path) -> str | None:
    if detect_variant(log_path) != "DICE-full":
        return None

    saw_base = False
    saw_u = False

    with log_path.open(errors="replace") as handle:
        for line in handle:
            if not saw_base and (BASE_CORE_RE.search(line) or BASE_PIPELINE_RE.search(line)):
                saw_base = True
            if not saw_u and (U_CORE_RE.search(line) or U_PIPELINE_RE.search(line)):
                saw_u = True

            if saw_base and saw_u:
                break

    if saw_u and not saw_base:
        return DICE_U_MODE
    if saw_base and not saw_u:
        return DICE_BASE_MODE

    raise ValueError(f"Could not determine whether {log_path} is DICE or DICE-U")


def build_date_mode_map(app_dir: Path, dice_rows: list[dict[str, str]]) -> dict[str, str | None]:
    result_dates = sorted({row["date_time"] for row in dice_rows})
    log_files = sorted(path for path in app_dir.glob("test_dice_*.log") if not path.name.endswith(("_asan.log", "_stderr.log")))

    if len(log_files) < len(result_dates):
        raise ValueError(
            f"{app_dir.name}: found {len(result_dates)} DICE date groups in result.csv but "
            f"only {len(log_files)} primary raw DICE logs"
        )
    if len(log_files) > len(result_dates):
        log_files = log_files[-len(result_dates):]

    return {
        date_time: detect_scale_mode(log_path)
        for date_time, log_path in zip(result_dates, log_files)
    }


def latest_rows_for_mode(dice_root: Path, app: str, mode: str) -> tuple[str, list[dict[str, str]]]:
    csv_path = dice_root / app / f"{app}.result.csv"
    rows = read_csv_rows(csv_path)
    date_to_mode = build_date_mode_map(dice_root / app, rows)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["date_time"]].append(row)

    matches = [date_time for date_time, detected_mode in date_to_mode.items() if detected_mode == mode]
    if not matches:
        raise FileNotFoundError(f"No {mode} rows found in {csv_path}")

    latest = max(matches)
    return latest, grouped[latest]


def latest_log_for_mode(app_dir: Path, mode: str) -> Path:
    matches = []
    for log_path in sorted(path for path in app_dir.glob("test_dice_*.log") if not path.name.endswith(("_asan.log", "_stderr.log"))):
        try:
            if detect_scale_mode(log_path) == mode:
                matches.append(log_path)
        except ValueError:
            continue

    if not matches:
        raise FileNotFoundError(f"No {mode} log found in {app_dir}")

    return matches[-1]


def kernel_label_map_from_mode(
    dice_root: Path,
    mode: str = DICE_BASE_MODE,
) -> tuple[dict[str, dict[str, str]], list[tuple[str, str, str]]]:
    per_app: dict[str, dict[str, str]] = {}
    ordered: list[tuple[str, str, str]] = []

    for app in BENCHMARK_ORDER:
        _, rows = latest_rows_for_mode(dice_root, app, mode)
        kernels_in_order: list[str] = []
        for row in sorted(rows, key=lambda item: int(item["kernel_launch_uid"])):
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


__all__ = [
    "BENCHMARK_ORDER",
    "DICE_BASE_MODE",
    "DICE_U_MODE",
    "build_date_mode_map",
    "detect_scale_mode",
    "geomean",
    "grouped_kernel_runs",
    "kernel_label_map_from_mode",
    "latest_log_for_mode",
    "latest_rows_for_mode",
    "write_csv",
]
