#!/usr/bin/env bash
# Thin wrapper around `make test_dice app=<app>` that records wall time and the
# fingerprint we use to verify behavior is unchanged across refactors.
#
# Workloads are configured via the bench dirs themselves:
#   nn_cuda/run      -> filelist_4
#   bfs/run          -> graph4096.txt
#   backprop/run     -> 4096 (already)
#
# Usage:
#   bash perf_test/bench.sh <variant>      # e.g. baseline / refactor1 / refactor2
#
# Output:
#   perf_test/results.csv  appended row per (variant, app)
#   perf_test/logs/<variant>_<app>.log  copy of the simulator log
#   perf_test/logs/<variant>_<app>.fp   fingerprint of cycles + cache stats

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DICE_TEST_DIR="${REPO_ROOT}/dice-test-gpu-rodinia/cuda/dice_test"
PERF_DIR="${REPO_ROOT}/perf_test"
RESULTS_CSV="${PERF_DIR}/results.csv"
LOG_DIR="${PERF_DIR}/logs"
mkdir -p "${LOG_DIR}"

APPS=(nn_cuda bfs backprop)

variant="${1:-}"
if [[ -z "${variant}" ]]; then
  echo "Usage: $0 <variant>" >&2
  exit 1
fi

# Ensure simulator env is active (LD_LIBRARY_PATH -> dice debug build)
pushd "${REPO_ROOT}/dice_gpgpu-sim" >/dev/null
set +eu
# shellcheck disable=SC1091
source ./setup_environment debug >/dev/null
set -eu
popd >/dev/null
[[ -n "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]] || { echo "[ERROR] setup_environment failed" >&2; exit 1; }
echo "GPGPUSIM_CONFIG=${GPGPUSIM_CONFIG}"

ensure_header() {
  if [[ ! -s "${RESULTS_CSV}" ]]; then
    echo "variant,app,sim_time_s,inst_per_s,cycle_per_s,wall_s,user_s,sys_s,gpu_tot_sim_cycle,gpu_tot_sim_insn,L1I_acc,L1I_miss,L1D_acc,L1D_miss,L1B_acc,L1C_acc,L1T_acc" > "${RESULTS_CSV}"
  fi
}

extract_field() {
  # last occurrence of `<key> = <value>` in the log
  local log="$1" key="$2"
  grep -E "^[[:space:]]*${key}[[:space:]]*=" "${log}" | tail -1 | awk -F= '{gsub(/[ \t]+/,""); print $2}'
}

run_one() {
  local app="$1"
  local stem="${variant}_${app}"
  local log_target="${LOG_DIR}/${stem}.log"
  local timefile="${LOG_DIR}/${stem}.time"

  echo "==> ${variant} / ${app}"
  ( cd "${DICE_TEST_DIR}" \
      && /usr/bin/time -f '%e %U %S' -o "${timefile}" \
         make test_dice app="${app}" >/dev/null 2>&1 ) \
    || { echo "[ERROR] make test_dice app=${app} failed"; return 1; }

  # Find the latest summary log produced by the test harness
  local summary_dir="${DICE_TEST_DIR}/result_summary/${app}"
  local latest_log
  latest_log=$(ls -t "${summary_dir}"/test_dice_*.log 2>/dev/null | grep -v _stderr | head -1)
  [[ -n "${latest_log}" ]] || { echo "[ERROR] no summary log under ${summary_dir}"; return 1; }
  cp "${latest_log}" "${log_target}"

  local wall user sys
  read -r wall user sys < "${timefile}"

  local tot_cycle tot_insn l1i l1i_m l1d l1d_m l1b l1c l1t
  tot_cycle=$(extract_field "${log_target}" "gpu_tot_sim_cycle")
  tot_insn=$(extract_field "${log_target}" "gpu_tot_sim_insn")
  l1i=$(extract_field "${log_target}" "L1I_total_cache_accesses")
  l1i_m=$(extract_field "${log_target}" "L1I_total_cache_misses")
  l1d=$(extract_field "${log_target}" "L1D_total_cache_accesses")
  l1d_m=$(extract_field "${log_target}" "L1D_total_cache_misses")
  l1b=$(extract_field "${log_target}" "L1B_total_cache_accesses")
  l1c=$(extract_field "${log_target}" "L1C_total_cache_accesses")
  l1t=$(extract_field "${log_target}" "L1T_total_cache_accesses")

  # Simulator-reported cumulative timing (last entry = total)
  local sim_time inst_rate cycle_rate
  sim_time=$(grep -E '^gpgpu_simulation_time' "${log_target}" | tail -1 | sed -E 's/.*\(([0-9]+) sec\).*/\1/')
  inst_rate=$(grep -E '^gpgpu_simulation_rate.*inst/sec' "${log_target}" | tail -1 | sed -E 's/.*= ([0-9]+).*/\1/')
  cycle_rate=$(grep -E '^gpgpu_simulation_rate.*cycle/sec' "${log_target}" | tail -1 | sed -E 's/.*= ([0-9]+).*/\1/')

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${variant}" "${app}" \
    "${sim_time:-NA}" "${inst_rate:-NA}" "${cycle_rate:-NA}" \
    "${wall}" "${user}" "${sys}" \
    "${tot_cycle:-NA}" "${tot_insn:-NA}" \
    "${l1i:-NA}" "${l1i_m:-NA}" "${l1d:-NA}" "${l1d_m:-NA}" \
    "${l1b:-NA}" "${l1c:-NA}" "${l1t:-NA}" \
    >> "${RESULTS_CSV}"

  # Save a normalized fingerprint (no times, no run-specific info) for diffing
  local fp="${LOG_DIR}/${stem}.fp"
  {
    grep -E "^[[:space:]]*(gpu_sim_cycle|gpu_tot_sim_cycle|gpu_sim_insn|gpu_tot_sim_insn)[[:space:]]*=" "${log_target}"
    grep -E "_total_cache_(accesses|misses|miss_rate|pending_hits|reservation_fails)[[:space:]]*=" "${log_target}"
  } > "${fp}"

  echo "    sim_time=${sim_time:-NA}s (cycle/sec=${cycle_rate:-NA}, inst/sec=${inst_rate:-NA}) wall=${wall}s tot_cycle=${tot_cycle:-NA}"
}

ensure_header
for app in "${APPS[@]}"; do
  run_one "${app}"
done
echo "==> done. results in ${RESULTS_CSV}"
