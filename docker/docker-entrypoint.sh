#!/usr/bin/env bash
set -euo pipefail

export CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH:-/usr/local/cuda-11.7}"
export PTXAS_CUDA_INSTALL_PATH="${PTXAS_CUDA_INSTALL_PATH:-${CUDA_INSTALL_PATH}}"
export JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"

cd /opt/DICE-test-collection

case "${1:-shell}" in
  base)
    shift
    exec bash integrated_test/run_base_test.sh "$@"
    ;;
  scale_out)
    shift
    exec bash integrated_test/run_scale_out_test.sh "$@"
    ;;
  all)
    shift
    bash integrated_test/run_base_test.sh "$@"
    exec bash integrated_test/run_scale_out_test.sh
    ;;
  shell)
    shift || true
    exec /bin/bash "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
