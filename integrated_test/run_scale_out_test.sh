#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Run this script with 'bash integrated_test/run_scale_out_test.sh' or './integrated_test/run_scale_out_test.sh'; do not source it." >&2
  return 0
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"
RUN_DICE="${RUN_DICE:-0}"
RUN_GPU="${RUN_GPU:-1}"
RUN_SCALE_OUT_SPEEDUP="${RUN_SCALE_OUT_SPEEDUP:-1}"
RUN_3070_SPEEDUP="${RUN_3070_SPEEDUP:-1}"
RUN_3070_RF="${RUN_3070_RF:-1}"

DICE_SIM_DIR="${REPO_ROOT}/dice_gpgpu-sim"
DICE_TEST_DIR="${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test"
GPU_SIM_DIR="${REPO_ROOT}/gpgpu-sim_distribution"
GPU_TEST_DIR="${REPO_ROOT}/gpu-rodinia/cuda/gpu_test"
SCALE_OUT_SPEEDUP_SCRIPT="${SCRIPT_DIR}/plot_scale_out_speedup.py"
SCALE_OUT_3070_SPEEDUP_SCRIPT="${SCRIPT_DIR}/plot_scale_out_3070_speedup.py"
SCALE_OUT_3070_RF_SCRIPT="${SCRIPT_DIR}/plot_scale_out_3070_rf.py"
OUTPUT_DIR="${SCRIPT_DIR}/generated_scale_out"

DICE_SCALE_OUT_RUNS=(
  "${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test/cfg/gpgpusim_dice_rtx5000.config:sw_20pe"
  "${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test/cfg/gpgpusim_dice_rtx6000.config:sw_20pe"
  "${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test/cfg/gpgpusim_dice_rtx3070.config:sw_rtx3070"
)

GPU_SCALE_OUT_CFGS=(
  "${REPO_ROOT}/gpu-rodinia/cuda/gpu_test/cfg/gpgpusim_gpu_rtx5000.config"
  "${REPO_ROOT}/gpu-rodinia/cuda/gpu_test/cfg/gpgpusim_gpu_rtx6000.config"
  "${REPO_ROOT}/gpu-rodinia/cuda/gpu_test/cfg/gpgpusim_gpu_rtx3070.config"
)

usage() {
  cat <<EOF
Usage: bash integrated_test/run_scale_out_test.sh [options]

Options:
  --run-dice {0|1}      Build and run the DICE scale-out sweeps. Default: ${RUN_DICE}
  --run-gpu {0|1}       Build and run the GPU scale-out sweeps. Default: ${RUN_GPU}
  --run-scale-out-speedup {0|1}
                        Run plot_scale_out_speedup.py. Default: ${RUN_SCALE_OUT_SPEEDUP}
  --run-3070-speedup {0|1}
                        Run plot_scale_out_3070_speedup.py. Default: ${RUN_3070_SPEEDUP}
  --run-3070-rf {0|1}   Run plot_scale_out_3070_rf.py. Default: ${RUN_3070_RF}
  -h, --help            Show this help message.

Environment variables:
  RUN_DICE, RUN_GPU, RUN_SCALE_OUT_SPEEDUP, RUN_3070_SPEEDUP, RUN_3070_RF, JOBS

Examples:
  bash integrated_test/run_scale_out_test.sh
  bash integrated_test/run_scale_out_test.sh --run-dice 1 --run-gpu 0
  RUN_DICE=0 RUN_GPU=1 RUN_SCALE_OUT_SPEEDUP=1 RUN_3070_SPEEDUP=1 RUN_3070_RF=1 bash integrated_test/run_scale_out_test.sh
EOF
}

parse_bool() {
  case "${1,,}" in
    1|true|yes|on) echo 1 ;;
    0|false|no|off) echo 0 ;;
    *)
      echo "Expected boolean value 0/1/true/false/yes/no/on/off, got: ${1}" >&2
      exit 1
      ;;
  esac
}

while (($# > 0)); do
  case "$1" in
    --run-dice)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      RUN_DICE="$(parse_bool "$2")"
      shift 2
      ;;
    --run-gpu)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      RUN_GPU="$(parse_bool "$2")"
      shift 2
      ;;
    --run-scale-out-speedup)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      RUN_SCALE_OUT_SPEEDUP="$(parse_bool "$2")"
      shift 2
      ;;
    --run-3070-speedup)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      RUN_3070_SPEEDUP="$(parse_bool "$2")"
      shift 2
      ;;
    --run-3070-rf)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      RUN_3070_RF="$(parse_bool "$2")"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_dir() {
  local dir="$1"
  [[ -d "${dir}" ]] || {
    echo "Missing directory: ${dir}" >&2
    exit 1
  }
}

require_file() {
  local file="$1"
  [[ -f "${file}" ]] || {
    echo "Missing file: ${file}" >&2
    exit 1
  }
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || {
    echo "Missing environment variable: ${name}" >&2
    echo "Export ${name} in the shell before running this script." >&2
    exit 1
  }
}

source_setup_environment() {
  local had_errexit=0
  local had_nounset=0
  local status=0

  [[ $- == *e* ]] && had_errexit=1
  [[ $- == *u* ]] && had_nounset=1

  set +e
  set +u
  source ./setup_environment "$@"
  status=$?

  (( had_nounset )) && set -u
  (( had_errexit )) && set -e

  return "${status}"
}

echo "==> Validating repository layout"
printf '   -> RUN_DICE=%s RUN_GPU=%s RUN_SCALE_OUT_SPEEDUP=%s RUN_3070_SPEEDUP=%s RUN_3070_RF=%s JOBS=%s\n' \
  "${RUN_DICE}" "${RUN_GPU}" "${RUN_SCALE_OUT_SPEEDUP}" "${RUN_3070_SPEEDUP}" "${RUN_3070_RF}" "${JOBS}"

if [[ "${RUN_DICE}" == "1" ]]; then
  require_dir "${DICE_SIM_DIR}"
  require_dir "${DICE_TEST_DIR}"
  require_env CUDA_INSTALL_PATH

  echo "==> Building DICE GPGPU-Sim in ${DICE_SIM_DIR}"
  pushd "${DICE_SIM_DIR}" >/dev/null
  source_setup_environment debug
  make -j"${JOBS}"
  popd >/dev/null

  echo "==> Running DICE scale-out benchmark sweeps in ${DICE_TEST_DIR}"
  pushd "${DICE_TEST_DIR}" >/dev/null
  for run_spec in "${DICE_SCALE_OUT_RUNS[@]}"; do
    cfg="${run_spec%%:*}"
    sw_dir="${run_spec##*:}"
    require_file "${cfg}"
    require_dir "${DICE_TEST_DIR}/${sw_dir}"
    echo "   -> CFG_dice=${cfg} SW_DIR=${sw_dir} RUN_SCRIPT=run_scale_out"
    make -j"${JOBS}" test_dice_all CFG_dice="${cfg}" SW_DIR="${sw_dir}" RUN_SCRIPT=run_scale_out
  done
  popd >/dev/null
else
  echo "==> Skipping DICE scale-out sweeps"
fi

if [[ "${RUN_GPU}" == "1" ]]; then
  require_dir "${GPU_SIM_DIR}"
  require_dir "${GPU_TEST_DIR}"
  require_env CUDA_INSTALL_PATH

  echo "==> Building baseline GPGPU-Sim in ${GPU_SIM_DIR}"
  pushd "${GPU_SIM_DIR}" >/dev/null
  source_setup_environment
  make -j"${JOBS}"
  popd >/dev/null

  echo "==> Running GPU scale-out benchmark sweeps in ${GPU_TEST_DIR}"
  pushd "${GPU_TEST_DIR}" >/dev/null
  for cfg in "${GPU_SCALE_OUT_CFGS[@]}"; do
    require_file "${cfg}"
    echo "   -> CFG_gpu=${cfg} RUN_SCRIPT=run_scale_out"
    make -j"${JOBS}" test_all CFG_gpu="${cfg}" RUN_SCRIPT=run_scale_out
  done
  popd >/dev/null
else
  echo "==> Skipping GPU scale-out sweeps"
fi

if [[ "${RUN_SCALE_OUT_SPEEDUP}" == "1" ]]; then
  require_file "${SCALE_OUT_SPEEDUP_SCRIPT}"
  echo "==> Generating scale-out RTX5000/RTX6000 speedup CSV/plots"
  pushd "${SCRIPT_DIR}" >/dev/null
  python3 "${SCALE_OUT_SPEEDUP_SCRIPT}" --output-dir "${OUTPUT_DIR}"
  popd >/dev/null
else
  echo "==> Skipping scale-out RTX5000/RTX6000 speedup post-processing"
fi

if [[ "${RUN_3070_SPEEDUP}" == "1" ]]; then
  require_file "${SCALE_OUT_3070_SPEEDUP_SCRIPT}"
  echo "==> Generating DICE-3070 vs RTX3070 speedup CSV/plots"
  pushd "${SCRIPT_DIR}" >/dev/null
  python3 "${SCALE_OUT_3070_SPEEDUP_SCRIPT}" --output-dir "${OUTPUT_DIR}"
  popd >/dev/null
else
  echo "==> Skipping DICE-3070 speedup post-processing"
fi

if [[ "${RUN_3070_RF}" == "1" ]]; then
  require_file "${SCALE_OUT_3070_RF_SCRIPT}"
  echo "==> Generating DICE-3070 vs RTX3070 RF-access CSV/plots"
  pushd "${SCRIPT_DIR}" >/dev/null
  python3 "${SCALE_OUT_3070_RF_SCRIPT}" --output-dir "${OUTPUT_DIR}"
  popd >/dev/null
else
  echo "==> Skipping DICE-3070 RF-access post-processing"
fi
