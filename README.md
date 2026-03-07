# DICE_ISCA_Eval

This repository contains the simulator trees, benchmark suites, and analysis scripts used to evaluate DICE against baseline GPGPU-Sim across three experiment families:

- base comparison on RTX2060S
- scale-up comparison between baseline DICE and DICE-U
- scale-out comparison across RTX5000, RTX6000, and RTX3070-style configurations

The main entry points live in [`integrated_test`](./integrated_test).

## Repository Layout

- `dice_gpgpu-sim`
  - DICE simulator tree
- `gpgpu-sim_distribution`
  - baseline GPGPU-Sim tree
- `dice-test-gpu-rodinia`
  - Rodinia benchmarks and DICE-side test harness
- `gpu-rodinia`
  - Rodinia benchmarks and GPU-side test harness
- `integrated_test`
  - top-level automation, plotting, and cleanup scripts

## Benchmarks

Both test harnesses use the same benchmark list:

- `nn_cuda`
- `bfs`
- `backprop`
- `streamcluster`
- `gaussian`
- `hotspot`
- `pathfinder`

## Prerequisites

Before running the wrappers, export the CUDA path in your shell:

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
export PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda
```

The wrapper scripts handle `setup_environment` internally. Do not `source` the wrapper scripts themselves.

Run wrappers with:

```bash
bash integrated_test/run_base_test.sh
bash integrated_test/run_scale_out_test.sh
```

## Base Flow

[`integrated_test/run_base_test.sh`](./integrated_test/run_base_test.sh) drives the base RTX2060S experiments.

It can:

- build and run the DICE simulator
- run four DICE variants:
  - `gpgpusim_dice_rtx2060s_naive.config`
  - `gpgpusim_dice_rtx2060s_unroll.config`
  - `gpgpusim_dice_rtx2060s_tmcu.config`
  - `gpgpusim_dice_rtx2060s.config`
- run the DICE-U scale-up sweep:
  - `gpgpusim_dice_u.config`
  - `SW_DIR=sw_40pe`
- build and run the baseline GPU simulator with:
  - `gpgpusim_gpu_rtx2060s.config`
- generate base plots and scale-up plots

### Common usage

Run everything:

```bash
bash integrated_test/run_base_test.sh
```

Only regenerate plots from existing data:

```bash
RUN_DICE=0 RUN_DICE_U=0 RUN_GPU=0 \
RUN_SPEEDUP=1 RUN_RF=1 \
RUN_SCALE_UP_PERF=1 RUN_SCALE_UP_RF=1 \
bash integrated_test/run_base_test.sh
```

### Base wrapper switches

- `--run-dice {0|1}`
- `--run-dice-u {0|1}`
- `--run-gpu {0|1}`
- `--run-speedup {0|1}`
- `--run-rf {0|1}`
- `--run-scale-up-perf {0|1}`
- `--run-scale-up-rf {0|1}`

### Base outputs

- DICE raw summaries:
  - `dice-test-gpu-rodinia/cuda/dice_test/result_summary`
- GPU raw summaries:
  - `gpu-rodinia/cuda/gpu_test/result_summary`
- generated base analysis:
  - `integrated_test/generated_base`
- generated scale-up analysis:
  - `integrated_test/generated_scale_up`

## Scale-Out Flow

[`integrated_test/run_scale_out_test.sh`](./integrated_test/run_scale_out_test.sh) drives the scale-out experiments.

It runs DICE with:

- `gpgpusim_dice_rtx5000.config` and `SW_DIR=sw_20pe`
- `gpgpusim_dice_rtx6000.config` and `SW_DIR=sw_20pe`
- `gpgpusim_dice_rtx3070.config` and `SW_DIR=sw_rtx3070`

It runs baseline GPU with:

- `gpgpusim_gpu_rtx5000.config`
- `gpgpusim_gpu_rtx6000.config`
- `gpgpusim_gpu_rtx3070.config`

Both harnesses use `RUN_SCRIPT=run_scale_out`.

### Common usage

Run the full scale-out flow:

```bash
bash integrated_test/run_scale_out_test.sh
```

Only regenerate plots from existing scale-out data:

```bash
RUN_DICE=0 RUN_GPU=0 \
RUN_SCALE_OUT_SPEEDUP=1 \
RUN_3070_SPEEDUP=1 \
RUN_3070_RF=1 \
bash integrated_test/run_scale_out_test.sh
```

### Scale-out wrapper switches

- `--run-dice {0|1}`
- `--run-gpu {0|1}`
- `--run-scale-out-speedup {0|1}`
- `--run-3070-speedup {0|1}`
- `--run-3070-rf {0|1}`

### Scale-out outputs

- generated scale-out analysis:
  - `integrated_test/generated_scale_out`

## Analysis Scripts

All plotting scripts can also be run directly from `integrated_test`.

### Base comparison

- `plot_speedup.py`
  - DICE variants vs baseline GPU
- `plot_rf_access.py`
  - DICE-full RF access vs baseline GPU

### Scale-up

- `plot_scale_up_perf.py`
  - DICE-U performance vs baseline DICE
- `plot_scale_up_rf.py`
  - DICE-U RF access vs baseline DICE

### Scale-out

- `plot_scale_out_speedup.py`
  - `DICE-RTX5000`, `DICE-RTX6000`, and `RTX6000` speedup vs `RTX5000`
- `plot_scale_out_3070_speedup.py`
  - `DICE-3070` speedup vs `RTX3070`
- `plot_scale_out_3070_rf.py`
  - `DICE-3070` RF access reduction vs `RTX3070`

Each script writes per-launch CSVs, per-kernel summary CSVs, and PNG/PDF plots to its default output directory.

## Cleanup

Use the single cleanup wrapper:

```bash
bash integrated_test/clean_all.sh
```

This removes:

- benchmark logs
- benchmark build directories
- benchmark `result_summary` directories
- generated plot/output directories under `integrated_test`

To also clean both simulator build trees:

```bash
bash integrated_test/clean_all.sh --clean-simulators 1
```

## Notes and Assumptions

- The scale-out plotting scripts assume the latest three collected result groups correspond to:
  - `rtx5000`
  - `rtx6000`
  - `rtx3070`
- If a bad run is appended to `result_summary`, delete or clean it before regenerating plots.
- BFS can occasionally produce a merged kernel record in DICE output. The analysis scripts include a UID-gap safeguard so the broken launch is excluded instead of corrupting the geomean.
- `streamcluster` may need manual reruns if a bad scale-out run was appended. The plotting scripts operate on the current contents of `result_summary`.
