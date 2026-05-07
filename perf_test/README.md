# perf_test

Lightweight harness for measuring DICE simulator wall-clock and verifying
correctness across the cleanup-speedup work. Two harnesses, one per scope:

## Small-input single-config harness

`bench.sh` runs **NN(filelist_4) / BFS(graph4096) / backprop(4096)** through the
standard `make test_dice` flow on the default RTX2060S DICE config. Used for
fast (~30s end-to-end) iteration during refactors.

```bash
bash perf_test/bench.sh <variant_label>
```

Output:
- `perf_test/results.csv` — appends one row per (variant, app)
- `perf_test/logs/<variant>_<app>.log` — copy of the simulator's per-run log
- `perf_test/logs/<variant>_<app>.fp` — normalized fingerprint of cycle counts
  + L1I/D/B/C/T accesses+misses for diff-against-baseline correctness checks

## Full-suite harness

`full_sweep.sh` runs **all 7 Rodinia DICE benchmarks at full original input
size** (`filelist_512k`, `graph65536`, etc.) on the default RTX2060S DICE
config. Takes ~20 min on the optimized branch, ~60 min on baseline.

```bash
bash perf_test/full_sweep.sh <variant_label>
```

Output: `perf_test/full_results.csv` and `perf_test/logs/full_<variant>_<app>.{log,fp}`.

## Compare two runs

```bash
# correctness: cycle counts + cache stats must match
diff perf_test/logs/full_baseline_bfs.fp perf_test/logs/full_optimized_bfs.fp
```

## reextract

`reextract.sh` re-parses an existing log file and appends a row to
`results.csv` without re-running the simulator. Useful when extending the
extracted column set.
