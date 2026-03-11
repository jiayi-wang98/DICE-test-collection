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

## Pre-Setup

### 1. Clone the repository with its submodules

```bash
git clone git@github.com:jiayi-wang98/DICE-test-collection.git
cd DICE-test-collection
git submodule update --init --recursive
```

### 2. Install host-side build dependencies

The two simulator trees inherit the usual GPGPU-Sim build requirements. The local simulator READMEs list:

- `build-essential`
- `xutils-dev`
- `bison`
- `flex`
- `zlib1g-dev`
- `libglu1-mesa-dev`
- `libxi-dev`
- `libxmu-dev`
- `libglut3-dev`

On Ubuntu, the typical install command is:

```bash
sudo apt-get install build-essential xutils-dev bison flex zlib1g-dev \
  libglu1-mesa-dev libxi-dev libxmu-dev libglut3-dev
```

Optional extras:

- `python3`, `python3-numpy`, `python3-matplotlib`
  - needed for the local analysis/plotting scripts in `integrated_test`
- `doxygen`, `graphviz`
  - only needed if you want to rebuild simulator docs

### 3. Install CUDA and point the repo at it

The benchmark makefiles are currently configured for CUDA 11.7 in:

- [`dice-test-gpu-rodinia/common/make.config`](/data2/jwang710/DICE_ISCA_Eval/dice-test-gpu-rodinia/common/make.config)
- [`gpu-rodinia/common/make.config`](/data2/jwang710/DICE_ISCA_Eval/gpu-rodinia/common/make.config)

Current expected path:

```text
/usr/local/cuda-11.7
```

Before running any wrapper, export:

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda-11.7
export PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda-11.7
```

If your CUDA installation lives somewhere else, update both `common/make.config` files and export the matching environment variables above.

### 4. Rodinia benchmark data

Both benchmark trees expect their own local `data/` directory:

- `dice-test-gpu-rodinia/data`
- `gpu-rodinia/data`

For the current automated benchmark set, the required datasets are already kept in git in both trees. A normal clone of this repository plus submodules should already contain the data needed by:

- `bfs`
- `gaussian`
- `hotspot`
- `nn_cuda`
- `backprop`
- `streamcluster`
- `pathfinder`

So for the current flows, you normally do not need to download Rodinia data separately.

The currently tracked benchmark input files are:

- `data/bfs/graph65536.txt`
- `data/bfs/graph128k.txt`
- `data/gaussian/matrix208.txt`
- `data/hotspot/temp_512`
- `data/hotspot/power_512`
- `data/nn/inputGen/cane512k.db`

Notes:

- `backprop` and `streamcluster` do not require external files in the current `run` / `run_scale_out` scripts.
- `pathfinder` also does not use an external dataset; `data/pathfinder/.gitkeep` is present only to keep the directory in the repo.

If your local checkout is missing the data directories, or if you want to repopulate them from the original Rodinia package, the repository layout expects the Rodinia 3.1 data package contents to be extracted there.

If you already have `rodinia-3.1-data.tar.gz`, extract it into both trees:

```bash
tar -xf rodinia-3.1-data.tar.gz -C /data2/jwang710/DICE_ISCA_Eval/dice-test-gpu-rodinia/data
tar -xf rodinia-3.1-data.tar.gz -C /data2/jwang710/DICE_ISCA_Eval/gpu-rodinia/data
```

If one tree is already populated, the simplest way to mirror it into the other is:

```bash
rsync -a /data2/jwang710/DICE_ISCA_Eval/gpu-rodinia/data/ \
  /data2/jwang710/DICE_ISCA_Eval/dice-test-gpu-rodinia/data/
```

or the reverse direction if the DICE tree already has the data.

For the currently automated benchmarks, the important subdirectories are:

- `data/bfs`
- `data/gaussian`
- `data/hotspot`
- `data/nn`
- `data/pathfinder`

### 5. DICE metadata bundles

The DICE test harness depends on the pre-generated metadata/PPTX bundles stored in:

- `dice-test-gpu-rodinia/cuda/dice_test/sw_20pe`
- `dice-test-gpu-rodinia/cuda/dice_test/sw_40pe`
- `dice-test-gpu-rodinia/cuda/dice_test/sw_rtx3070`

The wrappers use them as follows:

- base DICE: `sw_20pe`
- DICE-U: `sw_40pe`
- DICE-3070: `sw_rtx3070`
- DICE-RTX5000 / DICE-RTX6000: `sw_20pe`

### 6. Simulator environment

For manual simulator builds, use the simulator `setup_environment` scripts:

```bash
cd dice_gpgpu-sim
source setup_environment debug
make -j
```

```bash
cd gpgpu-sim_distribution
source setup_environment
make -j
```

The wrapper scripts handle `setup_environment` internally. Do not `source` the wrapper scripts themselves.

### 7. Quick start

Run the two top-level experiment flows with:

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
