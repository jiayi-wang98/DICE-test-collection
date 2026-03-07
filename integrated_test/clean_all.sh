#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Run this script with 'bash integrated_test/clean_all.sh' or './integrated_test/clean_all.sh'; do not source it." >&2
  return 0
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CLEAN_SIMULATORS="${CLEAN_SIMULATORS:-0}"

DICE_SIM_DIR="${REPO_ROOT}/dice_gpgpu-sim"
DICE_TEST_DIR="${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test"
GPU_SIM_DIR="${REPO_ROOT}/gpgpu-sim_distribution"
GPU_TEST_DIR="${REPO_ROOT}/gpu-rodinia/cuda/gpu_test"

usage() {
  cat <<EOF
Usage: bash integrated_test/clean_all.sh [options]

Options:
  --clean-simulators {0|1}
                        Also run 'make clean' in both simulator trees.
                        Default: ${CLEAN_SIMULATORS}
  -h, --help            Show this help message.

Environment variables:
  CLEAN_SIMULATORS

Examples:
  bash integrated_test/clean_all.sh
  bash integrated_test/clean_all.sh --clean-simulators 1
  CLEAN_SIMULATORS=1 bash integrated_test/clean_all.sh
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
    --clean-simulators)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      CLEAN_SIMULATORS="$(parse_bool "$2")"
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

echo "==> Cleaning benchmark outputs and generated plots"
printf '   -> CLEAN_SIMULATORS=%s\n' "${CLEAN_SIMULATORS}"

require_dir "${DICE_TEST_DIR}"
require_dir "${GPU_TEST_DIR}"

make -C "${DICE_TEST_DIR}" clean_all
make -C "${GPU_TEST_DIR}" clean_all

rm -rf "${SCRIPT_DIR}/generated_base"
rm -rf "${SCRIPT_DIR}/generated_scale_up"
rm -rf "${SCRIPT_DIR}/generated_scale_out"
rm -rf "${SCRIPT_DIR}/__pycache__"

if [[ "${CLEAN_SIMULATORS}" == "1" ]]; then
  require_dir "${DICE_SIM_DIR}"
  require_dir "${GPU_SIM_DIR}"

  echo "==> Cleaning simulator build trees"
  make -C "${DICE_SIM_DIR}" clean
  make -C "${GPU_SIM_DIR}" clean
else
  echo "==> Keeping simulator build trees"
fi

echo "==> Cleanup complete."
