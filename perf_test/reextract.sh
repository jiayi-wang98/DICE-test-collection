#!/usr/bin/env bash
# Re-extract a results.csv row from an existing log file.
# Usage: bash perf_test/reextract.sh <variant> <app> [logfile]
set -euo pipefail

PERF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_CSV="${PERF_DIR}/results.csv"
LOG_DIR="${PERF_DIR}/logs"

variant="$1"
app="$2"
log="${3:-${LOG_DIR}/${variant}_${app}.log}"
timefile="${LOG_DIR}/${variant}_${app}.time"

extract_field() {
  grep -E "^[[:space:]]*$2[[:space:]]*=" "$1" | tail -1 | awk -F= '{gsub(/[ \t]+/,""); print $2}'
}

if [[ ! -s "${RESULTS_CSV}" ]]; then
  echo "variant,app,sim_time_s,inst_per_s,cycle_per_s,wall_s,user_s,sys_s,gpu_tot_sim_cycle,gpu_tot_sim_insn,L1I_acc,L1I_miss,L1D_acc,L1D_miss,L1B_acc,L1C_acc,L1T_acc" > "${RESULTS_CSV}"
fi

wall=NA; user=NA; sys=NA
[[ -f "${timefile}" ]] && read -r wall user sys < "${timefile}"

tot_cycle=$(extract_field "${log}" "gpu_tot_sim_cycle")
tot_insn=$(extract_field "${log}"  "gpu_tot_sim_insn")
l1i=$(extract_field   "${log}" "L1I_total_cache_accesses")
l1i_m=$(extract_field "${log}" "L1I_total_cache_misses")
l1d=$(extract_field   "${log}" "L1D_total_cache_accesses")
l1d_m=$(extract_field "${log}" "L1D_total_cache_misses")
l1b=$(extract_field   "${log}" "L1B_total_cache_accesses")
l1c=$(extract_field   "${log}" "L1C_total_cache_accesses")
l1t=$(extract_field   "${log}" "L1T_total_cache_accesses")

sim_time=$(grep   -E '^gpgpu_simulation_time' "${log}" | tail -1 | sed -E 's/.*\(([0-9]+) sec\).*/\1/')
inst_rate=$(grep  -E '^gpgpu_simulation_rate.*inst/sec'  "${log}" | tail -1 | sed -E 's/.*= ([0-9]+).*/\1/')
cycle_rate=$(grep -E '^gpgpu_simulation_rate.*cycle/sec' "${log}" | tail -1 | sed -E 's/.*= ([0-9]+).*/\1/')

printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
  "${variant}" "${app}" \
  "${sim_time:-NA}" "${inst_rate:-NA}" "${cycle_rate:-NA}" \
  "${wall:-NA}" "${user:-NA}" "${sys:-NA}" \
  "${tot_cycle:-NA}" "${tot_insn:-NA}" \
  "${l1i:-NA}" "${l1i_m:-NA}" "${l1d:-NA}" "${l1d_m:-NA}" \
  "${l1b:-NA}" "${l1c:-NA}" "${l1t:-NA}" \
  >> "${RESULTS_CSV}"

echo "${variant} ${app}: sim_time=${sim_time:-NA}s cycle/sec=${cycle_rate:-NA} inst/sec=${inst_rate:-NA} cycle=${tot_cycle:-NA}"
