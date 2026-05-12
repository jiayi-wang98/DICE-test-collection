# RFU v7 Hand-Optimization Log

Baseline = current RFU v7 (post-migration, pre-optimization).
SMEM baseline = `cuda/dice_test/sw_20pe/` (pathfinder/hotspot/backprop) or
`*.bak_lds` (stencil2d/3d) or `sw_20pe/stencil1d.1.sm_52.{pptx,meta}.lds_baseline`.

| kernel | SMEM baseline | RFU v7 (pre-opt) | speedup vs SMEM |
|---|---:|---:|---:|
| stencil1d | 3447 | 3345 | 1.030 |
| stencil2d | 4376 | 4630 | 0.945 |
| stencil3d | 5144 | 6546 | 0.786 |
| hotspot | 89139 | 84410 | 1.056 |
| pathfinder | 75578 | 95891 | 0.788 |
| backprop layerforward | 7577 | 10524 | 0.720 |
| backprop adjust_weights | 4434 | 4570 | 0.969 |

Allowed optimizations (programming-model rule: write-`%rN` and SRF-read-`%rN`
must be in different DBBs):
- **#2** UF bank-disjoint register renaming
- **#3** DBB fusion (no SRF/race violation)
- **#4** Instruction-level scheduling inside a DBB

Each step: backup as `*.optstep<N>.bak`, apply, run `./run`, record cycles and
correctness. If correctness fails, revert and document.

## Steps

### Step 1 — pathfinder DBB 6 UF=1 → UF=2 (technique #2)

- **Change:** Loop-body compute DBB (BARRIER + 2× ld.srf + min/min/add) bumped
  from UF=1 to UF=2.
- **Rationale:** PE=3 (min/min/add), fits 6 at UF=2 within 16 budget.  Operand
  fetch banks {T+7,T+8,T+9,T+20}∪{T+15,T+16,T+17,T+28} = 8 distinct.  SRF banks
  at cycle 2 distribute (W/E clamped to distinct tids → distinct banks).
- **Backup:** `pathfinder.1.sm_52.{pptx,meta}.optstep0_v7`
- **Correctness:** `Iteration 1: CPU and GPU results match`
- **Cycles:** 95891 → **81567** (14.9% speedup vs v7-pre, now 8% slower than SMEM 75578)

### Step 2 — stencil3d %r16→%r22 rename + DBB 6 UF=1→2 (technique #2)

- **Change:** `.reg %r<22>` → `%r<23>`; DBB 5 LD_DEST `%r16`→`%r22`, DBB 6
  selp source and IN_REGS updated.  DBB 6 UF=1 → UF=2.
- **Rationale:** Original DBB 6 IN_REGS (%r8,%r15,%r16) had %r8–%r16 distance 8
  → forced UF=1.  %r22 is coset 6 alternative not aliasing any DBB-5/6 reg.
- **Backup:** `stencil3d.1.sm_52.{pptx,meta}.optstep1_v7`
- **Correctness:** stencil3d match 64×64×16, 0 mismatches.
- **Cycles:** 6546 → **6243** (4.6% speedup)

### Step 3 — stencil3d {%r14,%r17,%r18,%r21} renamed for DBB 9 UF=2 (technique #2)

- **Change:** `.reg %r<23>` → `%r<32>`.  Renames: `%r14→%r24`, `%r17→%r27`,
  `%r18→%r28`, `%r21→%r23`.  DBB 9 UF=1 → UF=2.
- **Rationale:** DBB 9 store had 8 %r IN_REGS spread across only 4 cosets mod 8
  (cosets {1,2,5,6} with 2 regs each → distance-8 collisions).  Rename targets
  picked one per missing coset {0,3,4,7} via free regs {%r23,%r24,%r27,%r28}.
- **Backup:** `stencil3d.1.sm_52.{pptx,meta}.optstep2_v7`
- **Correctness:** stencil3d match 64×64×16, 0 mismatches.
- **Cycles:** 6243 → **5988** (4.1% speedup, total 8.5% from v7-pre)

### Step 4 — backprop layerforward %r12→%r13 + compute DBBs UF=1→2 (technique #2)

- **Change:** Kernel 1 only.  Publication reg `%r12` → `%r13` everywhere in
  kernel 1; SRF addr offset `+12` → `+13` in DBB 1 (`my_base_addr`) and DBB 12
  (partial-sum tx==0 ld.srf).  DBBs 3/5/7/9 UF=1 → UF=2.
- **Rationale:** Compute DBB IN_REGS were (%r1,%r4,%r13_old=%r12).  %r4 and
  %r12 both coset 4 → distance 8 → forced UF=1.  Moving pub reg to %r13
  (coset 5) leaves cosets {1,4,5} = all distinct.
- **Backup:** `backprop.1.sm_52.{pptx,meta}.optstep2_v7`
- **Correctness:** layerforward `[**SUCCESS]`, adjust_weights `CPU and GPU match`.
- **Cycles:** layerforward 10524 → **9255** (12.1%), total 15094 → **13777** (8.7%).

## Current state (after steps 1-4)

| kernel | SMEM baseline | RFU v7-pre | RFU post-opt | Δ vs v7-pre | speedup vs SMEM |
|---|---:|---:|---:|---:|---:|
| stencil1d | 3447 | 3345 | 3345 | – | **1.030** ✓ |
| stencil2d | 4376 | 4630 | 4630 | – | 0.945 |
| stencil3d | 5144 | 6546 | 5988 | -8.5% | 0.859 |
| hotspot | 89139 | 84410 | 84410 | – | **1.056** ✓ |
| pathfinder | 75578 | 95891 | 81567 | -14.9% | 0.927 |
| backprop layerforward | 7577 | 10524 | 9255 | -12.1% | 0.819 |
| backprop total | 12011 | 15094 | 13777 | -8.7% | 0.872 |

Outstanding bottlenecks (technique #2/#3 won't fix without simulator-side or
programming-model change):
- `stencil2d` DBB 4: N/S `ld.srf` lands on same bank (stride ±16 mod 32 = 0)
  → `DISPATCH_II = 2`.  Same root cause as `stencil3d` DBB 8 (T/B stride ±64
  mod 32 = 0) and `hotspot` DBB 7 (N/S stride ±16).  Fix requires asymmetric
  publication-reg-per-row-parity which is structurally complex.
- Setup DBBs (e.g. `backprop` DBB 1 at 18 PE, `pathfinder` DBB 1/2/3) forced to
  UF=1 by PE budget.  Cost is amortized over the loop body, not per iter.
- Split-publish DBBs in iterative kernels (pathfinder/backprop) still spend
  half their loop-body cycles on the dedicated publish DBB.  Required by the
  programming-model rule.

## Pathfinder deep-dive (steps 5–12)

After the user pointed out the load-to-use rule violation, reverted DBB 5/6
fusion attempts (optsteps 5 and 6) and instead applied loop-invariant hoisting.

### Step 5 (REVERTED) — DBB 5+6 fusion
Putting `ld.global %r20` and its consumer `add.s32 %w6, %w5, %r20` in the same
DBB violates the load-to-use rule.  Backup `optstep4_v7`.

### Step 6 (REVERTED) — DBB 4+6 fusion
Reverted along with step 5.  Backup `optstep5_v7`.

### Step 7 (technique #4) — DBB 8 not-last-iter gate dropped
- **Change:** `@%p13 mov %r7, %r15` (gated on `i+1<iter`) → `@%p10 mov %r7, %r15`
  (gated only on computed-this-iter).  Last-iter publish to `%r7` is harmless
  because the post-loop store reads `%r15`, not `%r7`.
- **Saves:** 1 add + 1 setp + 1 and = 3 ops per iter.

### Step 8 (technique #4) — hoist gpuWall pointer + stride into DBB 3
- **Change:** Precompute `%r10 = base = &gpuWall[startStep*cols + xidx]` and
  `%r11 = 4*cols` in DBB 3.  DBB 5 becomes a single `ld.global` from `[%r10]`.
  DBB 8 does `add.s64 %r10, %r10, %r11` to advance.
- **Rationale:** Load-to-use rule kept (load in DBB 5, use in DBB 6).
- **Effect alone:** 81567 → 81382 cycles (the dispatch was tid-count bound, not
  PE-bound, so trimming DBB 5 ops alone doesn't speed it up).  But it unlocks
  the next two steps.

### Step 9 (technique #2) — DBB 5 UF=2 → UF=4
- **Now possible:** DBB 5 is 1 op, 0 PE, 1 LD.  At UF=4: 0 PE, 4 LD.  Reg
  cosets {%r10=2} — operand-fetch banks for T,T+8,T+16,T+24 = {T+10, T+18,
  T+26, T+2} mod 32, all distinct.
- **Cycles:** 81382 → **74670** (8.2% speedup).

### Step 10 (technique #2) — DBB 8 UF=2 → UF=4
- **PE:** 3 (add.s64, add.s32, setp) × 4 = 12 within 16 budget.  Reg cosets
  {r4=4, r10=2, r11=3, r15=7} all distinct; pairwise distances {6,7,11,1,5,4}
  none in {8,16,24}.
- **Cycles:** 74670 → **68340** (8.5% speedup).

### Step 11 (technique #2) — DBB 6 UF=2 → UF=4
- **PE:** 3 (min/min/add) × 4 = 12.  Operand-fetch banks at UF=4 for regs
  {r7,r8,r9,r20} × {T,T+8,T+16,T+24} = 16 distinct banks.  SRF reads at cycle
  2: 8 banks at distinct positions mod 32, fits the 8 LD ports the simulator
  is configured with (`-dice_cgra_core_num_ld_ports 8` in gpgpusim.config).
- **Cycles:** 68340 → **63795** (6.7% speedup).

### Step 12 (technique #4) — DBB 4 UF=2 → UF=4 via predicate refactor
- **Change:** Hoist `%r13 = tx-1` and `%r14 = 255-tx` into DBB 3.  Rewrite
  DBB 4 predicate as `(%r14 > i) AND (%r13 ≥ i)` instead of `(255-i) ≥ tx
  AND tx > i`.  Drops `mov 254` + `sub 254-i`, leaving 4 PE ops (2 setp +
  2 and).
- **PE:** 4 × 4 = 16 = at budget.  Reg cosets {r4=4, r13=5, r14=6} distinct.
- **Cycles:** 63795 → **57477** (9.9% speedup).

## Pathfinder cumulative summary

| stage | cycles | speedup vs prev | speedup vs v7-pre | speedup vs SMEM (75578) |
|---|---:|---:|---:|---:|
| v7 (pre-opt) | 95891 | — | — | 0.79× |
| step 1 (DBB 6 UF=2) | 81567 | 1.18× | 1.18× | 0.93× |
| step 8 (precompute) | 81382 | 1.00× | 1.18× | 0.93× |
| step 9 (DBB 5 UF=4) | 74670 | 1.09× | 1.28× | 1.01× |
| step 10 (DBB 8 UF=4) | 68340 | 1.09× | 1.40× | 1.11× |
| step 11 (DBB 6 UF=4) | 63795 | 1.07× | 1.50× | 1.18× |
| **step 12 (DBB 4 UF=4)** | **57477** | **1.11×** | **1.67×** | **1.32×** |

## Step 13 — sanity-rule corrections

Two violations caught in the rule audit:
1. **DBB 6 UF=4 issued 8 ld.srf at cycle 2 — exceeds the 4 SRF-port budget.**
   The `-dice_cgra_core_num_ld_ports 8` flag in `gpgpusim.config` refers to LD
   ports broadly, not SRF specifically.  SRF stays at 4 per
   `non_ilp_config.json`.
2. **DBB 3 had 17 PE ops at UF=1 — over the 16 PE budget.**

Fixes:
- DBB 6: UF=4 → UF=2 (4 SRF reads at cycle 2 → fits 4 ports).  Reg cosets
  and operand-fetch banks unchanged.
- DBB 3 split into DBB 3 (11 PE, UF=1: W/E SRF addrs + isValid + counter init)
  and new **DBB 15** (6 PE, UF=2: gpuWall base/stride + `tx-1`/`255-tx`
  precompute).  Flow `2 → 3 → 15 → 4 → 5 → 6 → 8 → 4 → ...`.

Backup: `pathfinder.1.sm_52.{pptx,meta}.optstep12_v7`.

| stage | cycles | speedup vs prev | speedup vs v7-pre | speedup vs SMEM |
|---|---:|---:|---:|---:|
| step 12 (DBB 4 UF=4) | 57477 | — | 1.67× | 1.32× |
| **step 13 (rule-compliant)** | **63228** | 0.91× | **1.52×** | **1.20×** |

### Sanity check (post step-13)

| check | result |
|---|---|
| PE ≤ 16 per DBB | ✓ all DBBs ≤ 16 (DBB 3 at 11, DBB 15 at 12 with UF=2) |
| SRF ≤ 4 per DBB | ✓ DBB 6 has SRF=2 at UF=2 → 4 ports, fits |
| LD ≤ 4 per DBB | ✓ DBB 5 has LD=1 at UF=4 → 4 ports, fits |
| Bank conflicts (cycle-0 + cycle-2) | ✓ all `max_bank_pressure = 1` |
| ld.global dest is %r | ✓ |
| ld.srf dest is %w | ✓ |
| Load-to-use | ✓ DBB 1 → DBB 2/3 (%r7); DBB 5 → DBB 6 (%r20) |
| Publication race (%r7 SRF-read + write same DBB) | ✓ DBB 6 reads, DBB 1+8 write — disjoint |
| %w wires intra-DBB | ✓ |
| Branch targets exist | ✓ DBB 2→7, DBB 8→4 |
| Correctness | ✓ `Iteration 1: CPU and GPU results match`, 63228 cycles |

## Hotspot rule-compliant optimization

Original v7-pre had multiple violations: DBB 1 at 19 PE, DBB 3 at 17 PE, DBB 5
at 23 PE, DBB 6 at 9 PE × UF=2 = 18, DBB 16 at 12 PE × UF=2 = 24.  The
simulator doesn't enforce PE budgets in cycle counts, so the violating version
appeared "fast" at 84410 cycles.  Rule-compliant rewrite must respect 16 PE
per DBB and 4 SRF/4 LD/4 ST ports per cycle.

### Steps applied (all to hotspot only):

1. **#1 quick fix** — Dropped UF on DBB 6 (was 18 PE) and DBB 16 (was 24 PE)
   to UF=1.  Backup `optstep0_v7`.

2. **#3 split DBB 1** (19 PE → 15 PE) — moved index calc + ld.global into
   new DBB 11.

3. **#3 split DBB 3** (17 PE → 9 PE) — moved isValid combine + early-exit
   branch into new DBB 12.

4. **#3 split DBB 5** (23 PE → 12 PE) — moved W/E clamps + W/E SRF addrs
   into new DBB 13.  Also **hoisted** `tx-1`, `15-tx`, `ty-1`, `15-ty` into
   DBB 13 so DBB 6 can drop its `255-i` calc.

5. **#4 trim DBB 6** — use precomputed bounds, predicate goes from 9 PE to
   8 PE, putting it back at UF=2 within budget.

6. **#3 fuse DBB 11 → DBB 2** — collapsed back to fewer pre-loop DBBs (13
   PE + 2 LD at UF=1).

7. **#3 fuse DBB 4 → DBB 5** — combined rcp/cvt with N/S addr (16 PE at UF=1).

8. **#4 reuse %p20 in DBB 16** — last-iter narrowing predicate already gives
   the right gate for the store; eliminate the redundant predicate compute.
   DBB 16 drops from 12 PE to 2 PE, now UF=4.

9. **#4 simplify validY/validX** — `max(-blkY,0)` via `max.s32 %r, %w, 0`
   instead of `neg + shr + and`; `min(15, cols-blkY-1)` instead of `setp +
   selp`.  Saves 4 PE per side.  Now DBB 2+3 fuse to 14 PE at UF=1.

10. **#4 trim DBB 8** — drop not-last-iter predicate (publish on last iter is
    harmless since %r10 is unused after the loop).  DBB 8 drops from 5 PE
    to 2 PE; now UF=4.

### Cycle progression

| stage | cycles | speedup vs prev | vs v7-pre (84410) | vs SMEM (89139) |
|---|---:|---:|---:|---:|
| v7-pre (with violations) | 84410 | — | 1.000 | 1.056 |
| after splits (rules-compliant, no fusion) | 107239 | 0.79× | 0.787 | 0.831 |
| + fuse 11→2, 4→5 (#3) | 98908 | 1.08× | 0.853 | 0.901 |
| + reuse %p20 in DBB 16 (#4) | 93684 | 1.06× | 0.901 | 0.951 |
| + validY/X simplify + fuse 3→2 (#4) | 86040 | 1.09× | 0.981 | 1.036 |
| + DBB 8 trim → UF=4 (#4) | **82478** | **1.04×** | **1.023** | **1.081** |

Final: **82478 cycles, 1.08× faster than SMEM**, fully rule-compliant.

### Sanity (post-optstep10)

| check | result |
|---|---|
| PE ≤ 16 per DBB | ✓ all (DBB 5 at 16 = at budget; rest ≤ 15) |
| SRF ≤ 4 per DBB | ✓ DBB 7 has 4 SRF at UF=1 (max 4) |
| LD ≤ 4 per DBB | ✓ DBB 2 has 2 LD at UF=1 |
| ST ≤ 4 per DBB | ✓ DBB 16 has 1 ST × UF=4 = 4 ports |
| Bank pressure ≤ 1 | ✓ |
| ld.global dest = %r | ✓ |
| ld.srf dest = %w | ✓ |
| Load-to-use (different DBB) | ✓ DBB 2's `ld.global %r10/%r11` → DBB 7 (much later) |
| Publication race | ✓ DBB 7 reads neighbors' %r10 via SRF; DBB 8 writes %r10 — disjoint DBBs |
| Correctness | ✓ `Maximum difference: 0.000000` |

## Stencil2d optimization

| step | technique | change | cycles |
|---|---|---|---:|
| – | v7-post-migration | (baseline) | 4630 |
| 1 | #3+#4 | mad fuses shl+add for gx/gy/gz, SRF base fuse (`mad %r8, %r8, 32, 7`); DBB 2 fused into DBB 1.  14 PE at UF=1.  Saves one whole DBB-traversal per CTA. | **4288** |

**Speedup vs SMEM (4376): 1.02× faster.** All sanity checks pass.

## Stencil3d optimization

| step | technique | change | cycles |
|---|---|---|---:|
| – | v7-post-migration | (baseline, post-step-3 from earlier session) | 5988 |
| 4 | #3+#4 | mad fuses shl+add for gx/gy/gz; edge predicates moved from DBB 2 into DBB 1; byte-addr + ld.global + SRF-base moved into DBB 2 (now UF=2). | 5829 |
| 5 | #4 | hoist `byte_step_z = cols_x*cols_y*4` into DBB 2 as `%r25`; DBB 7 drops from 9 PE to 7 PE → UF=2 | **5528** |

**Speedup vs SMEM (5144): 0.93× (7% slower).**  Still slower because DBB 8
(T/B SRF) has structural `DISPATCH_II=2` (stride ±64 mod 32 = 0).

### Sanity (both stencils, post-opt)

| check | stencil2d | stencil3d |
|---|---|---|
| PE ≤ 16 per DBB | ✓ (DBB 1 at 14) | ✓ (DBB 1 at 16 at budget) |
| SRF ≤ 4 per DBB | ✓ | ✓ |
| LD ≤ 4 per DBB | ✓ | ✓ |
| Bank pressure | ✓ all 1 | ✓ all 1 |
| ld.global/ld.srf dest types | ✓ | ✓ |
| Load-to-use | ✓ DBB 1→3 (%r7), DBB 3→4 (%r10/%r11) | ✓ DBB 2→9 (%r10), DBB 3→4 (%r11/%r12), DBB 5→6 (...), DBB 7→8 |
| Publication race | n/a (no iterative loop) | n/a (no iterative loop) |
| Correctness | ✓ 0 mismatches | ✓ 0 mismatches |
