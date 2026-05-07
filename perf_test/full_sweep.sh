#!/usr/bin/env bash
# Run all 7 DICE benchmarks (default RTX2060S config, full original inputs)
# via the standard `make test_dice` flow, capture wall time + fingerprint.
#
# Usage:
#   bash perf_test/full_sweep.sh <variant_label>
#
# Output:
#   perf_test/full_results.csv  one row per (variant,app)
#   perf_test/logs/full_<variant>_<app>.log  full simulator output
#   perf_test/logs/full_<variant>_<app>.fp   normalized fingerprint

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DICE_TEST_DIR="${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test"
PERF_DIR="${REPO_ROOT}/perf_test"
RESULTS_CSV="${PERF_DIR}/full_results.csv"
LOG_DIR="${PERF_DIR}/logs"
mkdir -p "${LOG_DIR}"

variant="${1:-}"
[[ -n "${variant}" ]] || { echo "Usage: $0 <variant_label>" >&2; exit 1; }

# Ensure simulator env active
pushd "${REPO_ROOT}/dice_gpgpu-sim" >/dev/null
set +eu
# shellcheck disable=SC1091
source ./setup_environment debug >/dev/null
set -eu
popd >/dev/null
[[ -n "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]] || { echo "[ERROR] setup_environment failed" >&2; exit 1; }
echo "GPGPUSIM_CONFIG=${GPGPUSIM_CONFIG}"

ALL_APPS=(nn_cuda bfs backprop streamcluster gaussian hotspot pathfinder)

ensure_header() {
  if [[ ! -s "${RESULTS_CSV}" ]]; then
    echo "variant,app,sim_time_s,inst_per_s,cycle_per_s,wall_s,gpu_tot_sim_cycle,gpu_tot_sim_insn,L1I_acc,L1I_miss,L1D_acc,L1D_miss,L1B_acc,L1C_acc,L1T_acc,exit_code" > "${RESULTS_CSV}"
  fi
}

extract_field() {
  grep -E "^[[:space:]]*$2[[:space:]]*=" "$1" | tail -1 | awk -F= '{gsub(/[ \t]+/,""); print $2}'
}

run_one() {
  local app="$1"
  local stem="full_${variant}_${app}"
  local log_target="${LOG_DIR}/${stem}.log"
  local timefile="${LOG_DIR}/${stem}.time"

  echo "==> ${variant} / ${app}: starting at $(date '+%H:%M:%S')"
  local rc=0
  ( cd "${DICE_TEST_DIR}" \
      && /usr/bin/time -f '%e %U %S' -o "${timefile}" \
         make test_dice app="${app}" >/dev/null 2>&1 ) || rc=$?

  local summary_dir="${DICE_TEST_DIR}/result_summary/${app}"
  local latest_log
  latest_log=$(ls -t "${summary_dir}"/test_dice_*.log 2>/dev/null | grep -v _stderr | head -1)
  if [[ -n "${latest_log}" ]]; then
    cp "${latest_log}" "${log_target}"
  else
    echo "[WARN] no summary log for ${app} (rc=${rc})" >&2
    return 1
  fi

  local wall=NA
  [[ -f "${timefile}" ]] && wall=$(awk '{print $1}' "${timefile}")

  local tot_cycle tot_insn l1i l1i_m l1d l1d_m l1b l1c l1t
  tot_cycle=$(extract_field "${log_target}" "gpu_tot_sim_cycle")
  tot_insn=$(extract_field "${log_target}"  "gpu_tot_sim_insn")
  l1i=$(extract_field   "${log_target}" "L1I_total_cache_accesses")
  l1i_m=$(extract_field "${log_target}" "L1I_total_cache_misses")
  l1d=$(extract_field   "${log_target}" "L1D_total_cache_accesses")
  l1d_m=$(extract_field "${log_target}" "L1D_total_cache_misses")
  l1b=$(extract_field   "${log_target}" "L1B_total_cache_accesses")
  l1c=$(extract_field   "${log_target}" "L1C_total_cache_accesses")
  l1t=$(extract_field   "${log_target}" "L1T_total_cache_accesses")

  local sim_time inst_rate cycle_rate
  sim_time=$(grep -E '^gpgpu_simulation_time' "${log_target}" 2>/dev/null | tail -1 | sed -E 's/.*\(([0-9]+) sec\).*/\1/')
  inst_rate=$(grep -E '^gpgpu_simulation_rate.*inst/sec'  "${log_target}" 2>/dev/null | tail -1 | sed -E 's/.*= ([0-9]+).*/\1/')
  cycle_rate=$(grep -E '^gpgpu_simulation_rate.*cycle/sec' "${log_target}" 2>/dev/null | tail -1 | sed -E 's/.*= ([0-9]+).*/\1/')

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d\n' \
    "${variant}" "${app}" \
    "${sim_time:-NA}" "${inst_rate:-NA}" "${cycle_rate:-NA}" \
    "${wall:-NA}" \
    "${tot_cycle:-NA}" "${tot_insn:-NA}" \
    "${l1i:-NA}" "${l1i_m:-NA}" "${l1d:-NA}" "${l1d_m:-NA}" \
    "${l1b:-NA}" "${l1c:-NA}" "${l1t:-NA}" \
    "${rc}" \
    >> "${RESULTS_CSV}"

  # Normalized fingerprint for diff
  local fp="${LOG_DIR}/${stem}.fp"
  {
    grep -E "^[[:space:]]*(gpu_sim_cycle|gpu_tot_sim_cycle|gpu_sim_insn|gpu_tot_sim_insn)[[:space:]]*=" "${log_target}"
    grep -E "_total_cache_(accesses|misses|miss_rate|pending_hits|reservation_fails)[[:space:]]*=" "${log_target}"
  } > "${fp}"

  echo "    rc=${rc} sim_time=${sim_time:-NA}s wall=${wall}s tot_cycle=${tot_cycle:-NA} L1D_acc=${l1d:-NA}"
  return ${rc}
}

ensure_header
overall_rc=0
for app in "${ALL_APPS[@]}"; do
  if ! run_one "${app}"; then
    overall_rc=1
    echo "[WARN] ${app} non-zero exit; continuing" >&2
  fi
done
echo "==> done. results in ${RESULTS_CSV} (overall exit=${overall_rc})"
exit ${overall_rc}
