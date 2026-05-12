# DICE Hand-Optimization Rules

Sanity-check rules for hand-written `.pptx` and `.meta` files of DICE
RF-unified kernels. These supplement `dice_compiler_design.md` (the
compiler-flow contract) and codify the lessons learned while migrating
stencil1d / stencil2d / stencil3d / hotspot / pathfinder / backprop.

Two pieces of tooling enforce most of this:

| script | what it checks |
|---|---|
| `/tmp/check_rfu.py` | per-DBB resource budgets, register declarations, BSL vs pptx op count |
| `/tmp/check_banks_v2.py` | pipeline-aware bank conflicts (operand fetch + SRF reads) |

Both must report `OK` on every active DBB before declaring a kernel done.

---

## 1. Per-DBB hard constraints

Inherited from `dice_compiler_design.md` (and still all in force):

1. **No internal branch.** At most one branch instruction per DBB, and it
   must be the terminator.
2. **No load-to-use inside the same p-graph.** If `ld.global` writes `%rX`,
   any subsequent read of `%rX` must live in a downstream DBB.
3. **No internal barrier.** `BARRIER` is a metadata flag on the DBB that
   logically "starts" with the barrier (the SRF-issuing DBB in our
   kernels); no `bar.sync` in pptx.
4. **Resource budget** (see §3 below).
5. **Register index limits.** `%r`, `%c`, `%p` indices must be `<
   reg_count` (32 in our config).
6. **Load destination class.**
   - `ld.global` / `ld.shared` / `ld.local` **must** target `%r` (or `%c`
     for `ld.param`). Never `%w`.
   - `ld.srf` is the exception: its destination **must** be `%w` (an
     intra-DBB wire — fabric forwarding only, no RF write).

---

## 2. SRF (cross-thread RF) rules

### 2.1 Address encoding (CGRA-visible)

The SRF address is in **register-word units** (low 2 byte-bits dropped):

```
bits [4:0]  = real register index   (%r0 .. %r31)
bits [14:5] = thread index in CTA
```

So `addr = (target_tid << 5) | reg_idx`.

The simulator decodes:

```
reg_idx     = addr & 0x1F
tid_in_cta  = addr >> 5
```

and looks up the producer thread's `%r{reg_idx}` symbol directly via the
function's symbol table. **No `SRF_PUBLICATION_REGS` and no
`LD_SRF_REGS`** are needed in the metadata anymore — the address itself
names the register.

### 2.2 Physical bank function

```
bank = (tid + reg_idx) mod 32
```

This swizzle lives in an address converter inside the SRF crossbar.
Local-RF reads and SRF reads share the same physical bank array — they
are not parallel structures.

### 2.3 `ld.srf` placement rule

Because `%w` is a wire, the **consumer of an `ld.srf` result must live
in the same DBB**. A `mov.b32 %r, %w` capture is **NOT** a valid
workaround — it defeats the RF-unification motivation by forcing the
value through a register write. Move the consumer into the SRF DBB
instead.

### 2.4 Producer/consumer ordering rule (the iterative-loop trap)

A DBB may **not** both `ld.srf` neighbor `%rN` and write to local `%rN`.
The simulator runs `dice_exec_block` (full functional update) per
dispatched thread, so a later-dispatched thread's `ld.srf` in the same
DBB sees an earlier-dispatched thread's updated `%rN`. Inside a DBB
there is no ordering between threads — only between DBBs.

Correct pattern for loop bodies that re-publish a register every iter:

| DBB | Role |
|-----|------|
| compute DBB | `BARRIER`, `ld.srf` old neighbor `%rN`, compute, capture result into a *different* `%r` (e.g. `%r15`). **No write to `%rN`.** |
| publish DBB | `mov %rN, %r15` (predicated), counter++, branch back |

`cgra_fabric_done` enforces "all threads finish compute DBB before any
thread enters publish DBB," so the publish-DBB writes can't be seen by
any compute-DBB `ld.srf` in the current iter. The publish DBB does not
need its own BARRIER since it has no `ld.srf`.

Symptom of getting this wrong: results close to correct but
systematically off by small amounts (e.g. GPU=24 vs CPU=22 in a min/add
chain — one neighbor read picks up the new value, so the result inherits
one extra `+wall[i]`).

---

## 3. Resource budgets (per p-graph, per cycle)

Match `dice-ilp/config/non_ilp_config.json`:

| resource | budget | notes |
|---|---|---|
| `pe_count`        | **16** | every ALU/compare/select op |
| `sfu_count`       | 4      | `div`, `rcp`, `sin`, `cos`, `sqrt`, `exp`, ... |
| `ldst_port_count` | 4      | `ld.global` + `ld.shared` + `st.global` + ... |
| SRF read ports    | 4      | per-cycle ceiling on `ld.srf` issue |
| `reg_count`       | 32     | `%r`, `%c`, `%p` are each ≤ 32 entries |

**Op categorization for PE counting:**

- **PE**: `add`, `sub`, `mul`, `mad`, `shl`, `shr`, `and`, `or`, `xor`,
  `min`, `max`, `setp`, `selp`, `and.pred`, `not.pred`, `cvt.f*`, ...
- **FREE** (no PE slot): `mov.*`, `cvta.*`, and `cvt.*` between integer
  32-bit/64-bit conversions (i.e. `cvt.s32.s64`, `cvt.u64.u32`, etc.).
- **LDST**: `ld.global`, `st.global`, `ld.shared`, `st.shared`,
  `ld.local`, `st.local`.
- **SRF**: `ld.srf` (no `st.srf` — never write SRF).
- **PARAM_LD**: `ld.param` — *excluded* from LDST budget.
- **BRANCH / RET**: do not contribute to PE.

**Per-DBB rule:**

```
PE_ops      × UF  <=  pe_count       ( = 16 )
ld + st ops × UF  <=  ldst_port_count ( = 4  )
SRF reads   × UF  <=  4
```

### 3.1 UF dispatch pattern (matters for bank checking)

From `cgra_core.cc::dispatcher_rfu_t::next_active_thread`:

```
UF = 1 : { T }
UF = 2 : { T, T+8 }
UF = 4 : { T, T+8, T+16, T+24 }
```

All lanes are spaced **8 threads apart** and step by 1 within an 8-thread
window before jumping to the next window. This pattern, combined with the
`(tid + reg_idx) mod 32` swizzle, dictates the UF register-naming rules.

### 3.2 UF register-naming rules (operand-fetch bank-disjointness)

For a DBB with `%r` set `R = IN_REGS.%r`, the UF lanes
{T+0, T+8, T+16, T+24} simultaneously read banks `(T + off + r) mod 32`
for every `(off, r) ∈ {lane offsets} × R`. To keep one bank
serving one read per cycle:

- **UF = 2**: no two indices `a, b ∈ R` may satisfy `a - b ≡ ±8 (mod 32)`.
- **UF = 4**: no two indices `a, b ∈ R` may satisfy
  `a - b ≡ ±8, ±16, ±24 (mod 32)` — equivalently `a - b ≡ 0 (mod 8)` and
  `a ≠ b`.

If this rule is violated, either **rename one of the colliding
registers** (preferred) or accept `DISPATCH_II = 2`/`4` for that DBB.

---

## 4. Pipeline model and II derivation

Per-thread reads happen at exactly two stages inside a DBB:

| stage | what is read | cost |
|:---:|---|---|
| 0   | every `%r` in `IN_REGS` (operand fetch)           | local RF, all in one cycle |
| k=2 | every `ld.srf` (cross-thread read)                | SRF crossbar, after the 1-cycle wire compute of the address |
| 1, 3, 4, ... | wires only                              | no RF, no SRF |

Intermediate `%w` values are wires; they never touch RF.

At steady-state pipelined dispatch (II=1, with LAT-deep pipeline), per
cycle two threads issue RF traffic: the just-dispatched thread T does
operand fetch, and thread T-2 does its SRF reads. The bank multiset is:

```
B_fetch(T)    = { (T + r) mod 32        | r ∈ IN_REGS.%r }
B_srf(T - 2)  = { (T - 2 + δ + R) mod 32 | each ld.srf (δ, R) }
            = { (T + δ + R - 2) mod 32 }
```

`DISPATCH_II = ceil(max_per_bank(B_fetch ∪ B_srf) / read_ports_per_bank)`,
with `read_ports_per_bank = 1` in our conservative model.

The checker at `/tmp/check_banks_v2.py` enumerates `T` across one CTA,
extends `B_fetch` and `B_srf` across all UF lanes, and reports the
worst-case bank pressure plus required II.

### Worked example — stencil1d DBB 3 (UF=1)

```
IN_REGS.%r  = {0, 2, 4, 5, 9, 10}
SRF reads   = (δ=-1, R=4), (δ=+1, R=4)
B_fetch(T)  = T + {0, 2, 4, 5, 9, 10}
B_srf(T-2)  = T + {-3+4, -1+4}  = T + {1, 3}
union       = T + {0, 1, 2, 3, 4, 5, 9, 10}   ← 8 distinct banks
max_per_bank = 1   → II = 1
```

---

## 5. Metadata field checklist (per DBB)

Required:

| field | when |
|---|---|
| `DBB_ID = N`               | always |
| `BITSTREAM_ADDR = $DICE_BBf_N` | always |
| `BITSTREAM_LENGTH = M`     | except for RET-only block |
| `UNROLLING_FACTOR = u`     | except for RET-only block |
| `UNROLLING_STRATEGY = 0`   | always when UNROLLING_FACTOR present |
| `LAT = lat`                | except for RET-only block |
| `IN_REGS = (...)`          | non-empty input set |
| `OUT_REGS = (...)`         | when DBB produces values consumed downstream |
| `LD_DEST_REGS = (...)`     | one entry per outstanding load destination |
| `IS_PARAMETER_LOAD`        | only on the param-load entry block |
| `STORE = n`                | number of stores |
| `BARRIER`                  | metadata flag on the DBB that issues `ld.srf` |
| `DISPATCH_II = n`          | only if II>1 (default 1) |
| `BRANCH = 1` + `BRANCH_PRED`/`BRANCH_TARGET`/`BRANCH_RECVPC` | branching DBB |
| `RET`                      | terminator block |

Removed / deprecated:

| field | reason |
|---|---|
| `SRF_PUBLICATION_REGS = (...)` | replaced by direct symbol lookup via the SRF address `reg_idx` |
| `LD_SRF_REGS = (...)`          | ld.srf destinations are intra-DBB wires; no metadata reflection needed |

---

## 6. PPTX style guidelines (RF-unified kernels)

1. **No `bar.sync` in pptx.** Use the `BARRIER` metadata flag on the
   SRF-issuing DBB.
2. **No `st.shared`.** RF-unification removes SMEM round-trips entirely
   (with one explicit exception: backprop keeps `input_node` SMEM as a
   broadcast-shaped table; document such exceptions in the pptx header).
3. **`ld.shared` only if intentionally retained** (same exception).
4. **`ld.srf` destination is `%w`.** Always. Never `%r`.
5. **`ld.global` destination is `%r`.** Never `%w`.
6. **Address arithmetic for `ld.srf`** must produce
   `addr = (target_tid << 5) | reg_idx`. Pre-compute the base
   `tid << 5 | reg_idx` in an earlier DBB if it saves PE budget in the
   SRF DBB; add the neighbor offset in the SRF DBB itself
   (`±32` for 1-stride neighbors, `±(BX*32)` for row neighbors, etc.).
7. **Predicate-name discipline.** Predicates produced in DBB X and read
   in DBB Y > X must keep their names across the boundary — do not
   reuse `%pN` for a different value inside a downstream DBB.
8. **Boundary fallback for SRF on CTA edges.** When the SRF address
   would point off-CTA (e.g., `tx-1` for `tx=0`), let it fall back to
   self via the `selp` offset gating, then override the result with a
   predicated `ld.global` of the cross-block value inside the SRF DBB.
   Document this in a comment line above the relevant ld.srf.
9. **Every SRF DBB should carry an inline cycle-by-cycle bank comment**
   listing operand-fetch banks (offsets from T), the SRF banks (also
   offsets from T), and the resulting II.

Template comment block to copy onto each SRF DBB:

```
// Cycle 0: operand fetch %r{...} -> banks T+{...}
// Cycle 1: wire-only address compute below
// Cycle 2: ld.srf -> bank ((tid + δ + reg_idx) mod 32) = T+{...}
// Combined banks T+{...} are <N> distinct -> DISPATCH_II = <ii>.
```

---

## 7. Sanity-check workflow

Before declaring a kernel done:

1. **Functional**: `bash run` under `setup_environment` from
   `dice_gpgpu-sim/`. Must print `CPU and GPU results match`.
2. **Resource check**: `python3 /tmp/check_rfu.py` — every active DBB
   must show `OK` (no PE / LD / ST / SRF / register-OOR / BSL-mismatch
   complaints). The RET-block `BSL-MISMATCH: meta=0 pptx=1` is benign
   noise.
3. **Bank-conflict check**: `python3 /tmp/check_banks_v2.py` — every
   active DBB must show `max_bank_pressure = 1` (or the meta must
   declare `DISPATCH_II` matching the reported pressure).
4. **Cycle-count regression**: compare against the previous
   hand-optimized baseline and confirm the change is in the expected
   direction.

A failing sanity check that you intentionally accept (e.g., taking a
known II=2 because renaming would cost more) must be **noted in the
pptx header comment** along with the reason.

---

## 8. Known pitfalls (learned the hard way)

- **`setup_environment` not sourced** → the test binary loads real CUDA,
  the simulator never runs, every result is garbage zeros with
  `CPU≠GPU`. Always source first; ldd should show
  `libcudart.so => /data2/.../dice_gpgpu-sim/lib/...`.
- **Quoting `SMEM baseline`**: the "raw compiler-emitted SMEM" cycle
  count is far worse than the "hand-optimized SMEM" cycle count for the
  same kernel. Apples-to-apples comparison vs RFU uses the
  hand-optimized baseline only.
- **PE budget is 16, not 20.** Easy mistake when scanning the simulator
  config (which has 20 lanes for unrelated reasons).
- **Register-bank UF rule** (no `a ≡ b mod (32/UF)` aside from `a = b`):
  applies to the *operand-fetch* `%r` set, not just to the SRF reads.
- **`cvta` and `mov` are FREE.** Don't count them as PE.
- **`mov %r, %w` is not a valid cross-DBB transfer for `ld.srf`
  results** — it defeats the unification motivation. Move the
  consumer instead.
- **Pipeline gap `k` defaults to 2** in our DBBs because the SRF
  address takes one cycle of wire arithmetic to form. If you split the
  SRF DBB further (e.g., move the address compute into an earlier DBB
  and start the SRF DBB at the actual `ld.srf`), `k` would drop to 0
  and the bank analysis would shift — re-run the checker.

---

## Change log

- 2026-05-11 — initial version covering RFU-v7 stencil1d migration and
  the new word-unit SRF address encoding.
- 2026-05-11 — added §2.4 producer/consumer ordering rule (the
  iterative-loop trap discovered while migrating pathfinder/hotspot/
  backprop): publication and `ld.srf` of the same `%rN` cannot live in
  the same DBB.  Use a compute DBB + publish DBB pair separated by
  `cgra_fabric_done`.  Pathfinder, hotspot, backprop layerforward all
  use this split.  Migration complete for all 6 RFU kernels.
