# Workflow: adding a new state-PE / CTA-atomic benchmark

This is the runbook for setting up a new benchmark that compares **GPU
SIMT** vs **DICE** vs **DICE+state-PE**. Follow it top-to-bottom for a
new benchmark; the gotchas section at the bottom captures bugs hit
during prefix_sum / reduce_large development that are worth checking
explicitly when something misbehaves.

---

## 0. Prerequisites (one-time)

| Path | What it is |
|---|---|
| `gpgpu-sim_distribution/` | Unmodified gpgpu-sim 4.0 + AccelWattch. Used for the GPU SIMT baseline. |
| `dice_gpgpu-sim/` | DICE-fork gpgpu-sim. Used for DICE and DICE+state-PE. |
| `dice-ilp/` | DICE compiler (DICETool.py). Generates `.pptx` / `.meta` from `.ptx`. |
| `gpu-rodinia/cuda/` | Per-benchmark source dirs for GPU SIMT baselines. |
| `dice-test-gpu-rodinia/cuda/` | Per-benchmark source dirs for DICE backends. |
| `dice-test-gpu-rodinia/cuda/dice_test/dice_atomics.h` | `dice_cta_acc_add` intrinsic header. |
| `dice-test-gpu-rodinia/cuda/dice_test/cfg/accelwattch_dice_sim.xml` | DICEwattch energy XML. |
| `gpu-rodinia/cuda/gpu_test/cfg/gpgpusim_gpu_rtx2060s.config` | GPU baseline gpgpu-sim config (1.47 GHz, 34 SMs, AccelWattch PTX-sim). |
| `dice-test-gpu-rodinia/cuda/dice_test/cfg/gpgpusim_dice_rtx2060s.config` | DICE backend config (1.47 GHz, 34 clusters, DICEwattch). |

The GPU and DICE configs are matched: same clock (1.47 GHz), same SM
count (34), same DRAM model. Don't drift these or comparisons stop
being apples-to-apples.

---

## 1. Pick the benchmark archetype

State-PE is a **CTA-scope many-to-one reduction or inclusive scan**
against a compile-time-known SMEM slot. It applies when:

- All threads of a CTA contribute to a single (or small number of)
  accumulator(s).
- The accumulator slot index is a compile-time constant (or
  `__COUNTER__`-derived; see `dice_atomics.h`).
- The compiler can recognise the slot as belonging to
  `__dice_acc_slots[]`.

Good fits: prefix scan, sum / max / min / xor reductions, vector dot,
norms, k-means cluster updates (per-cluster slots), histogram if the
bin set is compile-time-bounded.

Bad fits (state-PE is **not** the right primitive):
- Data-dependent SMEM atomic addresses (histograms with runtime bin
  indices — these are RF/SMEM "Pattern A", not state-PE).
- Cross-CTA atomics (those go through L2 atomic unit; state-PE is
  CTA-scope only).

---

## 2. Source-code template

Write three `.cu` files for the new benchmark `bench`:

| File | Role |
|---|---|
| `gpu-rodinia/cuda/bench/bench.cu` | GPU SIMT baseline. SMEM-tree or HS for scan; SMEM-tree for reduce. |
| `dice-test-gpu-rodinia/cuda/bench/bench.cu` | DICE backend baseline. *Same source as GPU baseline* (just live in a different dir so the DICE-specific PPTX/meta get generated next to it). |
| `dice-test-gpu-rodinia/cuda/bench_acc/bench_acc.cu` | DICE+state-PE variant. Uses `dice_cta_acc_add` from `../dice_test/dice_atomics.h`. |

**Conventions (preserves comparability):**
- Use the same input init (`srand(9); h_in[i] = (rand() & 0xFF) - 128`)
  so all three backends compute the same expected output and the
  correctness check just looks for `last=<expected>`.
- `BLOCK_SIZE` is a `#define` (overridable from the makefile). Default
  to 256 unless you need K2 single-CTA to handle more block_sums.
- **DICE caps CTA size at 512 threads** — never use `BLOCK_SIZE` >
  512 or the simulator will assert.
- Take `N` from `argv[1]` so the sweep harness can drive multiple
  problem sizes from one binary.
- Print `"CPU and GPU results match (N=<n>, sum=<v>)"` on success and
  `"MISMATCH ..."` on failure. The harness pattern-matches on this.

**state-PE variant skeleton:**

```c
#include "../dice_test/dice_atomics.h"

__global__ void bench_acc_kernel(const int *in, int *out, int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    // __shared__ is uninitialized per CUDA spec; thread 0 resets the slot
    // BEFORE the first dice_cta_acc_add. Skipping this leaks state
    // between CTAs because multiple CTAs share the same SM and slot 0.
    if (tid == 0) __dice_acc_slots[0] = 0;
    __syncthreads();

    int v = (gid < N) ? in[gid] : 0;
    int prefix = 0;
    dice_cta_acc_add(prefix, v);   // OLD value -> prefix (inclusive)

    // ... use `prefix` immediately in the same DBB if possible ...
    // ... if the OLD value is dead, that's fine; the compiler will
    //     route the atom dst to a %w wire (no scoreboard).

    // For a reduction (publish total to global counter), do it OUTSIDE
    // the same kernel if possible (see section 6: known issue with
    // atom.shared + atom.global in the same kernel — fixed but worth
    // a regression test).
    __syncthreads();
    if (tid == 0)
        atomicAdd(out, (int) __dice_acc_slots[0]);
}
```

---

## 3. Set up directories

Copy Makefile + run script from a known-working benchmark (e.g.
`reduce_large` / `reduce_large_acc`):

```bash
mkdir -p gpu-rodinia/cuda/bench && cd gpu-rodinia/cuda/bench
cp ../reduce_large/Makefile .
cp ../reduce_large/run .
sed -i 's/reduce_large/bench/g' Makefile run
chmod +x run
# (write bench.cu)
```

Same for `dice-test-gpu-rodinia/cuda/bench/` and `bench_acc/`. The
DICE `Makefile` template handles building + generating the binary;
DICETool is invoked separately (see §4).

---

## 4. Build and DICETool-compile

### 4a. GPU baseline

```bash
cd gpu-rodinia/cuda/gpu_test
source /…/gpgpu-sim_distribution/setup_environment > /dev/null 2>&1
make test_gpu app=bench
# checks: built binary, copied config, ran with default N, collected CSV
```

### 4b. DICE backends (baseline + acc)

For each DICE benchmark dir, the build is two-phase:

**Phase 1 — compile .cu to .ptx:**
```bash
cd dice-test-gpu-rodinia/cuda/bench
make                                                    # nvcc → binary
./bench 1024 > /dev/null 2>&1 || true                    # first run aborts (no .pptx yet) but emits .ptx
ls bench.1.sm_52.ptx                                    # confirm .ptx exists
```

**Phase 2 — DICETool .ptx → .pptx + .meta:**
```bash
cd dice-ilp
python3 DICETool.py -opt \
    -i /…/dice-test-gpu-rodinia/cuda/bench/bench.1.sm_52.ptx \
    --cfg_file config/non_ilp_config.json
# expect "[SUCCESS] Completed DICE Compilation!" at the end
ls /…/dice-test-gpu-rodinia/cuda/bench/bench.1.sm_52.{pptx,meta}
```

Repeat for the `_acc` dir.

**Verify the atom landed correctly in the state-PE variant:**

```bash
grep "atom" /…/bench_acc/bench_acc.1.sm_52.pptx
# expect (at minimum) one line like:
#   atom.shared.add.u32 %w1, [%w0], %r3;
# atom.shared dst should be %w if OLD value is dead; %r if used cross-DBB.
# atom.global dst MUST be %r (writeback needs a real RF slot).

grep -B1 -A5 "atom" /…/bench_acc/bench_acc.1.sm_52.meta
# expect the atom's DBB to show UNROLLING_FACTOR = 1
# (state-PE serializes through the slot; >1 unrolling deadlocks).
```

If something looks wrong here, **stop and check section 6** before
running.

### 4c. Convenience: combined invocation

The `dice_test/Makefile` does this end-to-end with `DICE_REGEN=1`:

```bash
cd dice-test-gpu-rodinia/cuda/dice_test
source /…/dice_gpgpu-sim/setup_environment > /dev/null 2>&1
make test_dice app=bench     DICE_REGEN=1
make test_dice app=bench_acc DICE_REGEN=1
```

---

## 5. Run and verify

### 5a. Single run for sanity

```bash
cd dice-test-gpu-rodinia/cuda/bench_acc
rm -f gpgpusim_dice_power_report.log
source /…/dice_gpgpu-sim/setup_environment > /dev/null 2>&1
./bench_acc 65536 > sim.out 2>&1
grep -E "results match|MISMATCH" sim.out
grep "gpu_sim_cycle =" sim.out | head
grep -E "kernel_avg_power|dice_acc_ops" gpgpusim_dice_power_report.log | head
```

**Expected, working output:**
```
bench_acc: CPU and GPU results match (N=65536, …).
gpu_sim_cycle = <N>
kernel_avg_power = <W>
dice_acc_ops = 65536, e_acc_ops_nJ = …
```

If `dice_acc_ops = 0` — the state-PE intrinsic was not recognised
(slot address didn't resolve to `__dice_acc_slots[]`). Check that
the `_acc` source `#include "../dice_test/dice_atomics.h"` and uses
`dice_cta_acc_add(prefix, v)`.

### 5b. Repeat for the other two backends to lock in cycle/energy:

```bash
# GPU baseline
cd gpu-rodinia/cuda/bench && source GPU env && ./run

# DICE baseline (SMEM tree, no acc-PE)
cd dice-test-gpu-rodinia/cuda/bench && source DICE env && ./run
```

---

## 6. Known failure modes and triage (in priority order)

| Symptom (what you see) | Root cause | Fix |
|---|---|---|
| `Assertion 's != NULL' failed` at sim start | Missing `.pptx`/`.meta` next to the binary | Re-run DICETool (§4b phase 2) |
| `Shader kernel CTA size is too large (max 512)` | `BLOCK_SIZE` > 512 in source | Reduce to ≤ 512; if you need larger N, use recursive K2 or a smaller block |
| `assertion 'm_reg_num_valid' failed` (ptx_ir.h:295) | Predicated cvta or store with non-register dst hitting the LDST queue | Already fixed in cuda-sim.cc:2461 (`pI->dst().reg_num_valid()` gate) — if it reappears, the gate regressed |
| `Deadlock detected: last writeback core ...` after K1-ish cycles | acc-PE's atom.shared dst was promoted to `%r` (phantom scoreboard slot, never released) | Already fixed in `DICENonILP.py` — check that atom.shared dst is `%w` in PPTX (`grep atom *.pptx`). Atom.global dst MUST be `%r`. |
| `MISMATCH` starting at i=0 with `dice_acc_ops = 0` | atom got DCE'd by DICETool (the original IR-type bug) | Should be fixed via `IRCfg.atom` + side-effect classification — if it reappears, check `_is_side_effect_instruction()` in DICENonILP.py still includes IRCfg.atom |
| `MISMATCH` only on Blelloch at `BLOCK_THREADS ≥ 256` | DICETool reg renaming for Blelloch's shrinking-active-thread predicate at larger blocks | Open issue. Workaround: use `BLOCK_THREADS=128` for Blelloch, or just use HS for the baseline (Blelloch loses to HS anyway). |
| Cycles look 4× lower than expected | UNROLLING_FACTOR=4 on the atom DBB (not 1) | DICETool should auto-set `UF=1` when `has_atom` is true on the block. Check `Printer.py`'s DICEMetaBlock construction — `has_atom=True` must be passed. |
| `kernel_avg_power = 0` or completely missing power report | DICE backend not configured with the power model | Check `gpgpusim.config` has `-gpgpu_dice_power_model 1` and `-gpgpu_dice_power_xml /.../accelwattch_dice_sim.xml`. The `dice_test/Makefile`'s `copy_config` step handles this automatically. |

---

## 7. Extract energy numbers

Use the harness in `sweep_n.py`. The exclusion list for SM-only
dynamic energy is hard-coded there:

```
EXCLUDE = {"L2CP", "MCP", "NOCP", "DRAMP", "IDLE_COREP",
           "CONST_DYNAMICP", "CONSTP", "STATICP"}
```

Power components ending in `P` not in this set are summed per
kernel; integrated over `cycles / freq_GHz` to get μJ. For a single
benchmark / single N, the inline parsing block in `sweep_n.py` is
~30 lines — easier to copy than reimplement.

**Chip-level energy** (whole-package): use the `kernel_avg_power`
field directly × cycles / freq. Always include both in the .md
results doc; the paper uses SM-only.

---

## 8. Sweep across N

If you want scaling data:

1. Add a new entry to the `BENCHES` dict in `sweep_n.py` with the
   four dirs (`gpu_dir`, `dice_dir`, `acc_dir`) and a list of `Ns`.
2. Mind the K2 cap: for any `prefix_sum`-style 3-kernel pipeline,
   `N ≤ BLOCK_SIZE × BLOCK_SIZE` (because K2 single-CTA-scans
   `nblocks = N/BLOCK_SIZE` and `nblocks ≤ BLOCK_SIZE`). For
   reduction-style benchmarks no such cap.
3. `nohup python3 -u sweep_n.py > sweep_n.log 2>&1 &` — runs are
   sequential; each cleans its own power-log before launching so
   per-kernel breakdowns are clean.
4. After it finishes, run `sweep_n_plot.py` for a log-log
   cycles+SM-energy plot.

---

## 9. Update the paper

Two files to edit:

| Where | What |
|---|---|
| `DICE-RFSMEM-HPCA-2027/main.tex` §VII.B (Stateful PE eval) | Add the new benchmark as a row group in Table V. Keep one headline ratio sentence per benchmark in the §Headline paragraph. |
| `DICE_ISCA_Eval/sweep_results.md` | Append the new benchmark's per-N tables. This is the supplementary doc the paper references for the scaling-invariance argument. |

The paper uses the largest measured N per benchmark for the headline
number, not the geomean across N. The sweep CSV is for showing
scaling invariance (a "we tried 11 N points, ratio is flat" sentence
in §Scaling).

---

## 10. Final regression checklist before declaring done

Run all of these from a clean state and confirm they pass:

```bash
# Existing benchmarks — make sure your changes haven't broken anything
for d in prefix_sum_large prefix_sum_large_acc prefix_sum_large_blelloch \
         reduce_large reduce_large_acc; do
  cd /…/dice-test-gpu-rodinia/cuda/$d
  rm -f gpgpusim_dice_power_report.log
  ./run > sim.out 2>&1
  echo "[$d] $(grep -E 'results match|MISMATCH' sim.out | head -1)"
done

# New benchmark itself
cd /…/dice-test-gpu-rodinia/cuda/<new_bench>_acc
./run > sim.out 2>&1
grep -E 'results match|MISMATCH|dice_acc_ops' sim.out gpgpusim_dice_power_report.log | head
```

All should report `CPU and GPU results match`, and the new
`_acc` benchmark should show `dice_acc_ops > 0`.

---

## Appendix: Where each compiler/sim fix landed

For future debugging — these are the touchpoints that make state-PE
work end-to-end. If anything regresses, check here first.

| Component | File | What |
|---|---|---|
| Atom IR type | `dice-ilp/src/include/tools/IRCfg.py` | `IRCfg.atom = "atom"`; `_getIRType_PTX` checks `atom` before `op` (since "add" is a substring of "atom.global.add") |
| Atom PTX classifier | `dice-ilp/src/lib/frontend/PTX/PTXDesc.json` | `"atom": ["atom"]` |
| Atom dataflow parsing | `dice-ilp/src/lib/opt/pre_compiler/passes/DataflowPass.py` | `elif dice_type == "atom"`: extract dst, [addr], src regs |
| Atom is side-effect (no DCE) | `dice-ilp/src/lib/opt/ptx_non_ilp/DICENonILP.py` `_is_side_effect_instruction` | Includes `IRCfg.atom` |
| Atom LDST resource cost | `DICENonILP.py` `_get_inst_resource_delta` | Atom = 1 LDST port |
| Atom code-motion anchor | `DICENonILP.py` `anchor_ops` (3 sites) | Atom is a memory-order anchor |
| Atom dst register pool | `DICENonILP.py` (around line 2876) | atom.shared dst → `%w` if dead; atom.global dst → `%r` always (LDST writeback needs RF slot) |
| Atom UF=1 constraint | `dice-ilp/src/include/primitives/meta/DICEMetaBlock.py` + Printer.py | `has_atom` flag forces UNROLLING_FACTOR=1 |
| Atom emission syntax | `dice-ilp/src/include/primitives/opt/DICEInst.py` `toString` | `atom <dst>, [<addr>], <src>` (bracket the address) |
| Atom meta as load | `dice-ilp/src/lib/backend/printer/DICE/Printer.py` `match` block | `case IRCfg.atom`: behaves like load for `LD_DEST_REGS` |
| Atom address-operand index | `DICEInst.calcAddressOffset` | reg_index=2 for atom |
| Sim: atom.shared synchronous fire | `dice_gpgpu-sim/src/cuda-sim/cuda-sim.cc:~2332` | inline RMW for shared-space atom; no LDST queue |
| Sim: atom.global → LDST queue | `cuda-sim.cc:~2350` | `add_callback` + `add_mem_op` (memory_load, is_atomic=true) |
| Sim: dst reg gating for stores | `cuda-sim.cc:2461` | `dst_reg_num = 0` if `!reg_num_valid()` (avoids assertion on pure-store dst operands) |
| Sim: DICEwattch acc-op counter | `dice_gpgpu-sim/src/gpuwattch/gpgpu_sim_wrapper.{h,cc}` | `dice_acc_op_e` (default 0.0257 nJ = PIPE\_A); folds into PIPEP. Counter bumped at `cuda-sim.cc:2344` (`g_dice_acc_op_count++`). |
| DICEwattch XML | `dice-test-gpu-rodinia/cuda/dice_test/cfg/accelwattch_dice_sim.xml` | `DICE_ACC_OP_E` param to tune per-op energy |
