#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Run this script with 'bash integrated_test/run_base_test.sh' or './integrated_test/run_base_test.sh'; do not source it." >&2
  return 0
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"
RUN_DICE="${RUN_DICE:-1}"
RUN_DICE_U="${RUN_DICE_U:-1}"
RUN_GPU="${RUN_GPU:-1}"
RUN_SPEEDUP="${RUN_SPEEDUP:-1}"
RUN_RF="${RUN_RF:-1}"
RUN_SCALE_UP_PERF="${RUN_SCALE_UP_PERF:-1}"
RUN_SCALE_UP_RF="${RUN_SCALE_UP_RF:-1}"

DICE_SIM_DIR="${REPO_ROOT}/dice_gpgpu-sim"
DICE_TEST_DIR="${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test"
GPU_SIM_DIR="${REPO_ROOT}/gpgpu-sim_distribution"
GPU_TEST_DIR="${REPO_ROOT}/gpu-rodinia/cuda/gpu_test"
SPEEDUP_SCRIPT="${SCRIPT_DIR}/plot_speedup.py"
RF_SCRIPT="${SCRIPT_DIR}/plot_rf_access.py"
SCALE_UP_PERF_SCRIPT="${SCRIPT_DIR}/plot_scale_up_perf.py"
SCALE_UP_RF_SCRIPT="${SCRIPT_DIR}/plot_scale_up_rf.py"
OUTPUT_DIR="${SCRIPT_DIR}/generated_base"
SCALE_UP_OUTPUT_DIR="${SCRIPT_DIR}/generated_scale_up"

DICE_CFGS=(
  "${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test/cfg/gpgpusim_dice_rtx2060s_naive.config"
  "${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test/cfg/gpgpusim_dice_rtx2060s_unroll.config"
  "${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test/cfg/gpgpusim_dice_rtx2060s_tmcu.config"
  "${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test/cfg/gpgpusim_dice_rtx2060s.config"
)
DICE_U_CFG="${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test/cfg/gpgpusim_dice_u.config"
GPU_CFG="${REPO_ROOT}/gpu-rodinia/cuda/gpu_test/cfg/gpgpusim_gpu_rtx2060s.config"

usage() {
  cat <<EOF
Usage: bash integrated_test/run_base_test.sh [options]

Options:
  --run-dice {0|1}      Build and run the DICE simulator/tests. Default: ${RUN_DICE}
  --run-dice-u {0|1}    Run the DICE-U scale-up sweep after normal DICE. Default: ${RUN_DICE_U}
  --run-gpu {0|1}       Build and run the baseline GPU simulator/tests. Default: ${RUN_GPU}
  --run-speedup {0|1}   Run plot_speedup.py after the tests. Default: ${RUN_SPEEDUP}
  --run-rf {0|1}        Run plot_rf_access.py after the tests. Default: ${RUN_RF}
  --run-scale-up-perf {0|1}
                        Run plot_scale_up_perf.py after the tests. Default: ${RUN_SCALE_UP_PERF}
  --run-scale-up-rf {0|1}
                        Run plot_scale_up_rf.py after the tests. Default: ${RUN_SCALE_UP_RF}
  -h, --help            Show this help message.

Environment variables:
  RUN_DICE, RUN_DICE_U, RUN_GPU, RUN_SPEEDUP, RUN_RF, RUN_SCALE_UP_PERF,
  RUN_SCALE_UP_RF, JOBS

Examples:
  bash integrated_test/run_base_test.sh
  bash integrated_test/run_base_test.sh --run-dice 1 --run-dice-u 1 --run-gpu 0 --run-speedup 0 --run-rf 0 --run-scale-up-perf 1 --run-scale-up-rf 1
  RUN_DICE=0 RUN_DICE_U=0 RUN_GPU=0 RUN_SPEEDUP=1 RUN_RF=1 RUN_SCALE_UP_PERF=1 RUN_SCALE_UP_RF=1 bash integrated_test/run_base_test.sh
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
    --run-dice-u)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      RUN_DICE_U="$(parse_bool "$2")"
      shift 2
      ;;
    --run-speedup)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      RUN_SPEEDUP="$(parse_bool "$2")"
      shift 2
      ;;
    --run-rf)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      RUN_RF="$(parse_bool "$2")"
      shift 2
      ;;
    --run-scale-up-perf)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      RUN_SCALE_UP_PERF="$(parse_bool "$2")"
      shift 2
      ;;
    --run-scale-up-rf)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      RUN_SCALE_UP_RF="$(parse_bool "$2")"
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
printf '   -> RUN_DICE=%s RUN_DICE_U=%s RUN_GPU=%s RUN_SPEEDUP=%s RUN_RF=%s RUN_SCALE_UP_PERF=%s RUN_SCALE_UP_RF=%s JOBS=%s\n' \
  "${RUN_DICE}" "${RUN_DICE_U}" "${RUN_GPU}" "${RUN_SPEEDUP}" "${RUN_RF}" "${RUN_SCALE_UP_PERF}" "${RUN_SCALE_UP_RF}" "${JOBS}"

if [[ "${RUN_DICE}" == "1" ]]; then
  require_dir "${DICE_SIM_DIR}"
  require_dir "${DICE_TEST_DIR}"
  require_env CUDA_INSTALL_PATH

  echo "==> Building DICE GPGPU-Sim in ${DICE_SIM_DIR}"
  pushd "${DICE_SIM_DIR}" >/dev/null
  source_setup_environment debug
  make -j"${JOBS}"
  popd >/dev/null

  echo "==> Running DICE benchmark sweep in ${DICE_TEST_DIR}"
  pushd "${DICE_TEST_DIR}" >/dev/null
  for cfg in "${DICE_CFGS[@]}"; do
    require_file "${cfg}"
    echo "   -> CFG_dice=${cfg}"
    make -j"${JOBS}" test_dice_all CFG_dice="${cfg}"
  done
  popd >/dev/null
else
  echo "==> Skipping DICE simulator/test sweep"
fi

if [[ "${RUN_DICE_U}" == "1" ]]; then
  require_dir "${DICE_TEST_DIR}"
  require_file "${DICE_U_CFG}"
  require_env CUDA_INSTALL_PATH

  echo "==> Running DICE-U scale-up sweep in ${DICE_TEST_DIR}"
  pushd "${DICE_TEST_DIR}" >/dev/null
  echo "   -> CFG_dice=${DICE_U_CFG} SW_DIR=sw_40pe"
  make -j"${JOBS}" test_dice_all CFG_dice="${DICE_U_CFG}" SW_DIR=sw_40pe
  popd >/dev/null
else
  echo "==> Skipping DICE-U scale-up sweep"
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

  echo "==> Running GPU benchmark sweep in ${GPU_TEST_DIR}"
  pushd "${GPU_TEST_DIR}" >/dev/null
  require_file "${GPU_CFG}"
  echo "   -> CFG_gpu=${GPU_CFG}"
  make -j"${JOBS}" test_all CFG_gpu="${GPU_CFG}"
  popd >/dev/null
else
  echo "==> Skipping baseline GPU simulator/test sweep"
fi

if [[ "${RUN_SPEEDUP}" == "1" ]]; then
  require_file "${SPEEDUP_SCRIPT}"
  echo "==> Generating speedup CSV/plots"
  pushd "${SCRIPT_DIR}" >/dev/null
  python3 "${SPEEDUP_SCRIPT}" --output-dir "${OUTPUT_DIR}"
  popd >/dev/null
else
  echo "==> Skipping speedup post-processing"
fi

if [[ "${RUN_RF}" == "1" ]]; then
  require_file "${RF_SCRIPT}"
  echo "==> Generating RF-access CSV/plots"
  pushd "${SCRIPT_DIR}" >/dev/null
  python3 "${RF_SCRIPT}" --output-dir "${OUTPUT_DIR}"
  popd >/dev/null
else
  echo "==> Skipping RF-access post-processing"
fi

if [[ "${RUN_SCALE_UP_PERF}" == "1" ]]; then
  require_file "${SCALE_UP_PERF_SCRIPT}"
  echo "==> Generating scale-up performance CSV/plots"
  pushd "${SCRIPT_DIR}" >/dev/null
  python3 "${SCALE_UP_PERF_SCRIPT}" --output-dir "${SCALE_UP_OUTPUT_DIR}"
  popd >/dev/null
else
  echo "==> Skipping scale-up performance post-processing"
fi

if [[ "${RUN_SCALE_UP_RF}" == "1" ]]; then
  require_file "${SCALE_UP_RF_SCRIPT}"
  echo "==> Generating scale-up RF-access CSV/plots"
  pushd "${SCRIPT_DIR}" >/dev/null
  python3 "${SCALE_UP_RF_SCRIPT}" --output-dir "${SCALE_UP_OUTPUT_DIR}"
  popd >/dev/null
else
  echo "==> Skipping scale-up RF-access post-processing"
fi
