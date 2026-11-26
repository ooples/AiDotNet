# AiDotNet ⇄ PyTorch parity benchmark

## Run all 11 model families

This command builds this checkout in Release and runs its models against Python PyTorch: MLP, CNN, LSTM, Transformer, GRU, ResNet-18, ResNet-50, VGG-11, MobileNetV2, DenseNet-121 and ViT, on CPU and on an NVIDIA GPU.

Install the .NET 10 SDK and create a Python 3.13 virtual environment. On Windows, open the x64 Developer PowerShell for Visual Studio with the C++ build tools, because `torch.compile` needs `cl.exe`. On Linux, install a C++ compiler such as gcc. For CUDA, install an NVIDIA driver that supports your PyTorch wheel. The reference run used these wheels:

```shell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install psutil==7.2.2 numpy==2.5.4
```

PyTorch publishes Triton, which compiles its GPU kernels, only for Linux. On Windows with CUDA, also install the community build: `python -m pip install triton-windows==3.9.0.post30`. For a machine without an NVIDIA GPU, install from `https://download.pytorch.org/whl/cpu` and pass `--devices cpu`. From the repository root, run:

```shell
python benchmarks/AiDotNet.PyTorchParity/run_matrix.py
```

The full run takes several hours on a laptop. For a quick execution check, run `python benchmarks/AiDotNet.PyTorchParity/run_matrix.py --models cnn --devices cpu --smoke`. Smoke timings are not measurements.

### What it measures

- **Separate processes.** Each model, device and launch runs AiDotNet and every PyTorch configuration in a new process. The default is five launches. The order rotates each launch, so no framework always runs first or last.
- **The fastest PyTorch configuration.** PyTorch runs in every configuration below that applies to the device and model. Every configuration uses `torch.compile` with static shapes, so each inference batch size gets its own specialized graph. Each training row and each inference batch size is compared with the PyTorch configuration that has the lowest median time for that row.

  | Configuration | Devices | Setup |
  | --- | --- | --- |
  | `torch-compile` | CPU, CUDA | `torch.compile` default mode, PyTorch's default AdamW (foreach on CUDA) |
  | `torch-compile-fused` | CPU, CUDA | `torch.compile` default mode, fused AdamW |
  | `torch-cudagraphs-fused` | CUDA | `torch.compile(mode="reduce-overhead")` with CUDA graphs, fused AdamW |
  | `torch-compile-fused-rnn` | CPU, GRU only | As `torch-compile-fused`, with `torch._dynamo.config.allow_rnn` so the GRU is compiled |
  | `torch-cudagraphs-fused-rnn` | CUDA, LSTM and GRU | As `torch-cudagraphs-fused`, with `allow_rnn` so the LSTM or GRU is compiled and captured in CUDA graphs |

  By default `torch.compile` does not compile `nn.LSTM` or `nn.GRU`. Dynamo skips any model that contains one, so the first three configurations run the LSTM and GRU rows eagerly, on cuDNN or oneDNN. Eager is often PyTorch's fastest choice for these models, so those configurations stay in the comparison. The two `-rnn` configurations compile the recurrent models as well. On CPU, PyTorch 2.14 cannot compile the LSTM (`expected Tensor() for op: torch.ops.aten.mkldnn_rnn_layer.default`), so the CPU LSTM row uses eager oneDNN. Each PyTorch report records how many graphs Dynamo compiled for the model. The runner rejects a compiled configuration that compiled nothing, unless the model is an LSTM or GRU that the configuration was not set to compile.

- **Same workload.** Both frameworks use eight CPU threads, seed 1234, AdamW with learning rate 1e-3, global gradient clipping at 1.0, and three epochs of 20 updates at batch size 64. The first epoch includes compilation and is excluded. The training figure is the median steady-state epoch time divided by 20. Inference runs at batch sizes 1, 8 and 16. After at least 10 calls and 1 second of warm-up, the runner records the median of 100 timed calls. GPU work is synchronized before each clock reading.
- **Same function where it can be checked.** Before any timing, MLP, CNN, LSTM and Transformer must reproduce PyTorch's first update from the same weights on each device. The check covers the forward pass, loss, gradients and parameter change. The other seven rows are timing comparisons of the same architecture. Three have slightly different parameter counts: GRU (AiDotNet 19,274, PyTorch 19,466), ViT (2,690,122 and 2,693,578: AiDotNet's attention has no query, key and value biases) and MobileNetV2 (2,253,738 and 2,236,682). Those rows are flagged `parameters-differ`. [benchmark.py](pytorch/benchmark.py) gives the PyTorch definitions and model notes.
- **The right engine and binary.** CUDA rows must report AiDotNet's CUDA backend, never OpenCL or HIP, and CPU rows must report the CPU engine. The benchmark must load the `AiDotNet.dll` built in this run. Every PyTorch report records the compile mode, optimizer and CUDA-graph use, and the runner checks them against the configuration.
- **Machine conditions.** The runner refuses to start on battery power. Each process receipt records the power source, system CPU load, GPU clocks, temperature and utilization, and any GitHub Actions job running on the machine. The summary flags affected rows:

  | Flag | Meaning |
  | --- | --- |
  | `battery` | A launch ran on battery power (only with `--allow-battery`) |
  | `busy-cpu` | System CPU use exceeded 20% just before a launch |
  | `busy-gpu` | GPU utilization exceeded 10% just before a CUDA launch |
  | `ci-job` | A GitHub Actions job was running on the machine |
  | `unstable` | One side's slowest launch exceeded its fastest by more than 1.5× |
  | `parameters-differ` | The two models have different trainable parameter counts |
  | `missing-launches` | A launch failed, so the row has fewer than the requested launches |
  | `gpu-clock-low` | A CUDA launch ended with the GPU busy below 60% of its maximum SM clock, or below 90% of the clock declared with `--gpu-clock-lock` |

The printed run directory contains the following:

- `protocol.json`, with the hardware, driver, runtime and package versions and the git revision.
- `assemblies.json`, with the assembly hashes and whether Tensors came from NuGet or source.
- The build and first-update check logs.
- Each process's command, log, receipt and raw JSON report.
- `measurements.csv`, with every launch.
- `summary.csv`, with each row's median and range per framework, the PyTorch configuration it was compared with, every configuration's median and the flags.
- `summary.md`, with the table below.

In both CSVs, `torch_over_ai` is PyTorch's time divided by AiDotNet's, so a value above 1 means AiDotNet took less time. Failed processes are listed in `completion.json`, and the runner exits nonzero. To rebuild the CSVs and table from the raw reports of any run directory or extracted archive, run `python benchmarks/AiDotNet.PyTorchParity/run_matrix.py --summarize RUN_DIR`. The output goes to `RUN_DIR/regenerated`.

By default the build uses the AiDotNet.Tensors NuGet package pinned in `Directory.Packages.props`. To test a source build, pass `--tensors /path/to/AiDotNet.Tensors`. The runner then checks that the loaded assembly is that build's.

### Results from November 21–26, 2026

The machine was a Ryzen 7 4800H laptop on AC power, with a 6 GB GTX 1660 Ti, NVIDIA driver 617.42, Windows 11, .NET 10 and PyTorch 2.14.1+cu126. Every run used AiDotNet.Tensors 0.151.0 from NuGet and five launches. Each row compares AiDotNet with PyTorch's fastest configuration for that row, including the configurations that compile the LSTM and GRU. GRU, ViT and MobileNetV2 carry ⚑ on every row because their parameter counts differ slightly (`parameters-differ`).

The tables combine four runs:

| Rows | Commit | Finished (UTC) | Why it is separate |
| --- | --- | --- | --- |
| CPU, all models except ViT | `b23d88ba` | 03:58, November 21 | |
| CUDA, all models except ViT | `11142fbe` | 17:40, November 21 | The CUDA rows of the run above were measured while the platform held the GPU at 855 of 2,100 MHz, so they were rerun. |
| CPU ViT | `709c001a` | 21:41, November 24 | ViT was fixed after these runs; see [the ViT fix](#the-vit-fix). |
| CUDA ViT | `9dc1c828` | 12:15, November 26 | As above. |

Between `b23d88ba` and `9dc1c828` the only changes to the measured code are the ViT fix and the benchmark runner (large-data mode, GPU clock checks and flags). The runner's main-comparison protocol did not change. Later commits on this branch merge AiDotNet master, which moved to AiDotNet.Tensors 0.152.1 and changed other library code; none of these runs measured that. To reproduce them, check out `9dc1c828`, as AiDotNet.Tensors' [model-family launcher](https://github.com/ooples/AiDotNet.Tensors/tree/main/parity/model-families) does with `--snapshot 2026-11-26`.

**CUDA**: PyTorch time ÷ AiDotNet time, median of launches (range of launches). Above 1 means AiDotNet took less time. ⚑ marks a row with a flag in summary.csv.

| Model | Training, batch 64 | Inference, batch 1 | Inference, batch 8 | Inference, batch 16 |
| --- | ---: | ---: | ---: | ---: |
| mlp | 7.05 (6.68–8.98) | 1.35 (1.27–2.01) | 1.62 (1.00–2.09) ⚑ | 1.55 (1.47–2.63) ⚑ |
| lstm | 5.77 (5.64–6.51) | 1.24 (1.22–1.42) | 1.37 (1.25–1.49) | 1.53 (1.50–2.04) |
| transformer | 2.12 (1.98–2.33) ⚑ | 0.42 (0.36–0.70) ⚑ | 0.56 (0.52–1.08) ⚑ | 0.72 (0.68–0.80) ⚑ |
| cnn | 2.29 (2.18–2.48) | 1.09 (1.00–1.11) | 1.35 (1.33–1.55) | 1.50 (1.39–1.86) |
| resnet18 | 1.36 (1.30–1.50) | 1.95 (1.89–2.48) | 1.88 (1.12–2.01) ⚑ | 1.99 (1.79–2.06) |
| resnet50 | 1.59 (1.51–1.82) | 1.41 (1.25–1.51) | 1.46 (1.21–1.54) | 1.33 (1.25–1.34) |
| vgg11 | 1.19 (1.18–1.19) | 2.24 (1.62–2.40) | 1.72 (1.68–1.75) | 1.48 (1.47–1.52) |
| vit | 1.02 (1.02–1.02) ⚑ | 0.58 (0.57–0.71) ⚑ | 0.98 (0.74–1.08) ⚑ | 0.91 (0.89–0.95) ⚑ |
| gru | 7.60 (7.33–9.31) ⚑ | 1.76 (1.23–2.25) ⚑ | 1.82 (1.67–2.05) ⚑ | 2.09 (1.76–2.24) ⚑ |
| mobilenetv2 | 0.89 (0.82–0.91) ⚑ | 1.61 (1.09–1.84) ⚑ | 0.95 (0.87–0.97) ⚑ | 0.85 (0.84–1.73) ⚑ |
| densenet121 | 1.00 (0.99–1.01) | 1.33 (1.31–1.45) | 1.69 (1.68–1.71) | 1.31 (1.30–1.34) |

**CPU**: PyTorch time ÷ AiDotNet time, median of launches (range of launches). Above 1 means AiDotNet took less time. ⚑ marks a row with a flag in summary.csv.

| Model | Training, batch 64 | Inference, batch 1 | Inference, batch 8 | Inference, batch 16 |
| --- | ---: | ---: | ---: | ---: |
| mlp | 2.67 (2.67–3.04) | 4.69 (4.23–5.79) | 5.60 (4.55–6.33) | 3.72 (3.72–4.40) ⚑ |
| lstm | 2.08 (1.70–2.52) ⚑ | 4.01 (3.89–4.73) | 1.78 (1.64–2.28) ⚑ | 2.03 (2.02–2.43) |
| transformer | 1.43 (1.36–1.57) | 2.15 (1.92–2.67) | 2.76 (2.06–3.00) | 2.72 (2.55–2.83) |
| cnn | 1.07 (0.93–1.40) | 3.64 (3.56–5.71) ⚑ | 2.02 (1.88–2.70) | 1.50 (1.41–2.19) ⚑ |
| resnet18 | 4.73 (4.65–4.98) | 2.26 (2.20–3.16) | 1.83 (1.49–2.02) | 1.57 (1.38–2.20) |
| resnet50 | 3.03 (2.94–3.26) | 1.57 (1.54–1.64) | 1.61 (1.53–1.73) | 1.46 (1.25–1.52) |
| vgg11 | 1.32 (1.24–1.47) | 2.01 (1.74–2.34) | 2.98 (2.75–3.05) | 1.29 (1.19–1.33) |
| vit | 1.40 (1.23–1.41) ⚑ | 1.42 (1.07–1.44) ⚑ | 1.28 (0.92–1.30) ⚑ | 1.42 (1.29–1.46) ⚑ |
| gru | 1.49 (1.24–1.59) ⚑ | 9.17 (7.88–9.27) ⚑ | 5.09 (4.86–6.92) ⚑ | 4.12 (4.00–4.89) ⚑ |
| mobilenetv2 | 1.64 (1.34–1.84) ⚑ | 2.39 (2.15–2.84) ⚑ | 1.87 (1.82–1.96) ⚑ | 1.35 (1.31–1.44) ⚑ |
| densenet121 | 1.22 (1.03–1.26) | 1.38 (1.33–1.40) | 1.24 (1.13–1.25) | 1.28 (1.23–1.30) |

Where AiDotNet took longer, or the result is too close to call:

- **CUDA training.** MobileNetV2 is slower (0.89). DenseNet-121 (1.00) and ViT (1.02) are even.
- **CUDA inference.** The Transformer is slower at every batch size (0.42, 0.56, 0.72), and so is ViT (0.58, 0.98, 0.91). MobileNetV2 is slower at batches 8 and 16 (0.95, 0.85).
- **CPU.** AiDotNet's median is lower on every row. One CNN training launch (0.93) and one ViT batch-8 launch (0.92) went the other way.

Compiling the recurrent models changed two rows. PyTorch's compiled GRU is faster than its eager GRU on the CPU, so the CPU GRU rows fell from 2.35× to 1.49× (training) and from 15.40× to 9.17× (batch 1) against the [first November 20 run](#archived-runs), which compared against eager PyTorch only. On CUDA, the compiled GRU won every row, and the compiled LSTM won inference at every batch size. Eager cuDNN stayed PyTorch's fastest LSTM training on CUDA, and eager oneDNN the fastest CPU LSTM.

The fastest PyTorch configuration was `torch-cudagraphs-fused` for 10 of 11 CUDA training rows and 27 of 33 CUDA inference rows. On the CPU it was `torch-compile-fused` for 9 of 11 training rows.

Most `unstable` flags are on CUDA inference calls that take less than a millisecond. At that scale, one launch of either framework can take up to three times as long as another. The range column shows how far the launches spread. The four CUDA Transformer rows are flagged `gpu-clock-low` because AiDotNet's third launch ended with the GPU busy at a low clock. That launch's times lie inside the range of the other four. No row in the tables was flagged `battery`, `busy-cpu`, `busy-gpu`, `ci-job` or `missing-launches`.

Three sets of measurements were replaced, and all of them are published:

- In the CPU run, launch 5 first ran while the machine was also clearing disk space, and one of its processes ended on battery power. That launch was rerun on an idle machine on AC power, and then four of the run's CUDA rows whose receipts read 20–22% system CPU were rerun again. Every replaced process is kept under `superseded/` in its ZIP.
- The first CUDA rerun ran while a GitHub Actions job was using the machine, and every row was flagged `ci-job`. It was discarded and run again with the runner stopped.
- The ViT rows of the first two runs measured a ViT that did not learn. They remain in those runs' ZIPs, and the tables above use the reruns.

The CUDA rows ran at the GPU's normal clocks. This GPU computes wrong values at those clocks over long runs (see [the GPU clock lock](#the-gpu-clock-lock)). A wrong value changes what a model computes, not how long a step takes, so it does not move these timings, and a fault is rare within 20 updates. The large-data CUDA runs below, which train for hours and report accuracy, were made at a locked clock.

These results come from one Windows laptop. On Windows, PyTorch's GPU compiler uses the community `triton-windows` build, and Windows display-driver scheduling adds GPU launch overhead that Linux does not have. We have not run this on Linux yet, so these numbers do not show how the two frameworks compare there.

| Run | Summary | Every report, receipt and log |
| --- | --- | --- |
| CPU, November 21 | [CSV](evidence/2026-11-21-cpu-summary.csv) | [ZIP](evidence/2026-11-21-cpu.zip) |
| CUDA, November 21 | [CSV](evidence/2026-11-21-cuda-summary.csv) | [ZIP](evidence/2026-11-21-cuda.zip) |
| CPU ViT, November 24 | [CSV](evidence/2026-11-24-vit-cpu-summary.csv) | [ZIP](evidence/2026-11-24-vit-cpu.zip) |
| CUDA ViT, November 26 | [CSV](evidence/2026-11-26-vit-cuda-summary.csv) | [ZIP](evidence/2026-11-26-vit-cuda.zip) |
| Discarded CUDA run with a CI job on the machine, November 21 | | [ZIP](evidence/superseded-2026-11-21-cuda-ci-job.zip) |

The CPU run's ZIP also contains its CUDA rows, which were measured at the capped clock and are not used above. To rebuild a table from the raw reports, extract its ZIP and summarize it, for example `python -m zipfile -e benchmarks/AiDotNet.PyTorchParity/evidence/2026-11-21-cuda.zip benchmarks/AiDotNet.PyTorchParity/results/cuda-21`, then `python benchmarks/AiDotNet.PyTorchParity/run_matrix.py --summarize benchmarks/AiDotNet.PyTorchParity/results/cuda-21`.

### Archived runs

<a id="archived-runs"></a>The first November 20 run used this runner at commit `37597f19`, before PyTorch's LSTM and GRU were compiled. All three of its PyTorch configurations ran those two models eagerly, because `torch.compile` skips them unless `allow_rnn` is set. Its other rows match the run above within launch-to-launch spread. It is kept for comparison: [Table](evidence/2026-11-20.md) · [CSV](evidence/2026-11-20-summary.csv) · [Every launch](evidence/2026-11-20.csv) · [ZIP](evidence/2026-11-20.zip).

The November 13 and 19 runs came before this runner. Their reports are kept exactly as recorded. Each archive holds the 88 original framework reports: 11 models, two devices, two frameworks and two launches. Do not use them for performance claims. They differ from the current protocol in these ways:

- PyTorch ran in one configuration: `torch.compile` default mode with PyTorch's default AdamW. Inference at batch sizes 8 and 16 used the shape-generic (dynamic-shape) graph that `torch.compile` creates after it sees a second batch size.
- Each model ran in two launches, and AiDotNet always ran first.
- The power source, machine load and AiDotNet's GPU backend were not recorded for each process.

The November 13 run used public mainline commits. The November 19 run used a local integration of then-open pull requests that never shipped as a release. On November 19 the laptop was on battery and the first CUDA launch was throttled. Both frameworks were 5–10× slower in launch 1 than in launch 2: for example, PyTorch's ResNet-18 took 339 ms against 59 ms, and AiDotNet's ViT took 342 ms against 37 ms. The launch 1 CUDA ratios therefore say nothing about the frameworks. The integration build also contained an LSTM training regression. Its CPU training epoch took 0.80 s against 0.057 s on November 13, and its CUDA epoch was 7–28× slower, with identical losses. That regression is not in the code measured above.

| Run | Training and inference | Individual framework reports |
| --- | --- | --- |
| November 13 | [CSV](evidence/2026-11-13.csv) | [ZIP](evidence/2026-11-13.zip) |
| November 19 | [CSV](evidence/2026-11-19.csv) | [ZIP](evidence/2026-11-19.zip) |

For example, from the repository root, inspect the original CUDA GRU pair:

```shell
python -m zipfile -e benchmarks/AiDotNet.PyTorchParity/evidence/2026-11-19.zip benchmarks/AiDotNet.PyTorchParity/results/archive-19
python benchmarks/AiDotNet.PyTorchParity/pytorch/compare.py benchmarks/AiDotNet.PyTorchParity/results/archive-19/aidotnet-1-cuda-gru.json benchmarks/AiDotNet.PyTorchParity/results/archive-19/torch-compile-1-cuda-gru.json
```

The historical source revisions and their reproduction commands are in [AiDotNet.Tensors](https://github.com/ooples/AiDotNet.Tensors/tree/7b6f340ef8d006bc9331c9e7e9aa3717a1da3fec/parity/model-families). The [17-second LSTM video benchmark](https://github.com/ooples/AiDotNet.Tensors/tree/7b6f340ef8d006bc9331c9e7e9aa3717a1da3fec/parity/lstm-video) uses a separate fixture and training protocol.

## Large datasets (`--scale`)

The main comparison times 20 updates on random batches that fit in cache. The large-data mode checks that the results hold when each model trains for whole epochs over hundreds of thousands to a million real or non-repeating samples streamed from disk, and that both frameworks learn the task equally well.

```shell
python benchmarks/AiDotNet.PyTorchParity/scale/prepare_data.py --root D:/parity-data
python benchmarks/AiDotNet.PyTorchParity/run_matrix.py --scale D:/parity-data --scale-torch-from benchmarks/AiDotNet.PyTorchParity/evidence/2026-11-21-cuda-summary.csv --devices cuda --launches 3 --gpu-clock-lock 1200
```

`prepare_data.py` downloads two public datasets, refuses any archive whose SHA-256 differs from the one this benchmark was built from, and writes six datasets (about 5.5 GB) in one raw format that both harnesses read:

| Dataset | Training samples | Test samples | Sample | Models |
| --- | ---: | ---: | --- | --- |
| `emnist-digits`: NIST EMNIST Digits | 240,000 | 40,000 | 1×28×28 | MLP, CNN |
| `emnist-seq32`: the same digits padded to 32×32, one row per time step | 240,000 | 40,000 | 32 steps × 32 | LSTM, GRU, Transformer |
| `cinic10`: CINIC-10 (its train and valid splits for training) | 180,000 | 90,000 | 3×32×32 | ResNet-18, ResNet-50, VGG-11, MobileNetV2, DenseNet-121, ViT |
| `synth-mnist`, `synth-seq32`, `synth-rgb32`: synthetic | 1,000,000 each | 10,000 each | as above | as above |

Each synthetic sample is 25% a fixed random template for its class and 75% fresh uniform noise, so no two samples repeat and the models can still learn the classes.

What both sides do identically:

- **Data.** Both memory-map the same files and read each epoch's samples in the same shuffled order (a SplitMix64 Fisher–Yates shuffle implemented in both). Both normalise a byte of channel *c* to (*b*/255 − mean*c*)/std*c* with the training set's statistics, and both drop the last partial batch.
- **Training.** Batch 64, AdamW at learning rate 1e-3, gradient clipping at 1.0, seed 1234, eight CPU threads, the same model definitions as the main comparison. Before epoch 0, both run the same 20 untimed updates and one untimed test batch, so compilation and JIT stay out of the epoch times. Both read the training loss every 100 updates.
- **Scoring.** After every epoch, both score the whole test split in batches of 250, in inference mode. That time is reported separately and is not part of training time.
- **Throughput.** A launch's throughput is the samples it trained divided by its training time over all epochs. The table compares the median over launches.

Each framework feeds batches its own fastest way. AiDotNet gathers and normalises the next batch on a background thread. PyTorch normalises inside its compiled model, so the conversion is part of its compiled graph (and its CUDA graph), and copies only the raw bytes. It gathers each batch inline, into pinned memory on CUDA. That inline path was PyTorch's fastest of the options measured on an MLP over EMNIST Digits:

| PyTorch input path (MLP, 3,750 updates) | CUDA, ms per update | CPU, ms per update |
| --- | ---: | ---: |
| Inline gather into pinned memory (used) | 2.42 | — |
| Inline gather (used on CPU) | 2.64 | 3.58 |
| Background prefetch thread, as AiDotNet uses | 2.61 | 4.19 |
| `DataLoader`, 2 worker processes, pinned, prefetch 4 | 3.01 | 4.44 |
| `DataLoader`, 4 worker processes, pinned, prefetch 4 | 2.89 | — |
| Whole dataset already on the GPU (no streaming; a lower bound) | 2.22 | — |

The runner trains every model on its real dataset for three epochs and on its synthetic dataset for one. On the CPU, the six vision models train one epoch of CINIC-10 only: a CPU DenseNet-121 epoch over 180,000 images takes about 3.3 hours for the two frameworks together. Pass the CUDA main-comparison summary for `--devices cuda` and the CPU one for `--devices cpu`. `--scale-torch-from` runs, for each row, the PyTorch configuration that was fastest for that row's training in the given main-comparison summary. Without it, every applicable configuration runs. Run directories contain the same receipts as the main comparison, plus `scale-measurements.csv` (every epoch of every launch), `scale-summary.csv` and `scale-summary.md`.

### Large-data results, CUDA

Same laptop as above, on AC power, with the CI runner stopped. Commit `24b81597`, three launches, finished at 18:51 UTC on November 24. The ViT rows come from a rerun at commit `709c001a` after the ViT fix described below (three launches, finished 21:27 UTC on November 24). The GPU's SM clock was locked at 1200 MHz for both runs; [why](#the-gpu-clock-lock).

PyTorch time ÷ AiDotNet time is the median over launches of each launch's throughput ratio (range of launches). Above 1 means AiDotNet trained faster. Accuracy is the final test accuracy, median over launches.

| Model | Dataset | Samples per launch | AiDotNet samples/s | PyTorch samples/s | PyTorch time ÷ AiDotNet time | Test accuracy, AiDotNet / PyTorch | Flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| mlp | EMNIST Digits, 3 epochs | 720,000 | 150,635 | 23,884 | 6.17 (6.01–7.51) | 98.92% / 98.89% |  |
| mlp | synthetic, 1 epoch | 1,000,000 | 120,697 | 23,173 | 5.18 (4.83–11.52) | 99.99% / 100.00% | unstable |
| lstm | EMNIST rows, 3 epochs | 720,000 | 94,473 | 15,254 | 6.15 (5.67–7.10) | 98.80% / 98.80% |  |
| lstm | synthetic, 1 epoch | 1,000,000 | 80,707 | 17,189 | 4.66 (4.60–10.80) | 99.62% / 99.67% | unstable |
| gru | EMNIST rows, 3 epochs | 720,000 | 114,650 | 15,085 | 8.21 (7.17–8.53) | 98.92% / 98.87% |  |
| gru | synthetic, 1 epoch | 1,000,000 | 103,205 | 14,774 | 6.34 (1.89–13.66) | 99.59% / 99.52% | unstable |
| transformer | EMNIST rows, 3 epochs | 720,000 | 39,729 | 19,902 | 2.00 (1.96–2.04) | 94.66% / 94.83% |  |
| transformer | synthetic, 1 epoch | 1,000,000 | 39,764 | 19,653 | 2.03 (2.00–2.08) | 88.75% / 89.36% |  |
| cnn | EMNIST Digits, 3 epochs | 720,000 | 58,313 | 25,008 | 2.33 (2.33–2.35) | 99.16% / 99.19% |  |
| cnn | synthetic, 1 epoch | 1,000,000 | 58,246 | 24,835 | 2.35 (2.32–2.35) | 98.13% / 99.00% |  |
| resnet18 | CINIC-10, 3 epochs | 539,904 | 4,002 | 837 | 4.77 (4.73–4.83) | 64.82% / 65.50% |  |
| resnet18 | synthetic, 1 epoch | 1,000,000 | 4,034 | 834 | 4.84 (4.75–4.87) | 99.99% / 99.99% |  |
| resnet50 | CINIC-10, 3 epochs | 539,904 | 1,854 | 670 | 2.76 (2.76–2.77) | 61.29% / 59.72% |  |
| resnet50 | synthetic, 1 epoch | 1,000,000 | 1,850 | 670 | 2.76 (2.75–2.77) | 99.92% / 99.92% | ai-launches-disagree |
| vgg11 | CINIC-10, 3 epochs | 539,904 | 2,071 | 1,693 | 1.22 (1.22–1.22) | 62.39% / 63.71% | ai-launches-disagree |
| vgg11 | synthetic, 1 epoch | 1,000,000 | 2,088 | 1,692 | 1.23 (1.23–1.23) | 9.82% / 9.82% |  |
| vit | CINIC-10, 3 epochs | 539,904 | 1,330 | 1,328 | 1.00 (1.00–1.00) | 45.09% / 48.38% |  |
| vit | synthetic, 1 epoch | 1,000,000 | 1,329 | 1,327 | 1.00 (1.00–1.00) | 99.60% / 99.60% |  |
| mobilenetv2 | CINIC-10, 3 epochs | 539,904 | 4,753 | 5,735 | 0.83 (0.82–0.91) | 56.32% / 58.52% |  |
| mobilenetv2 | synthetic, 1 epoch | 1,000,000 | 4,753 | 5,789 | 0.82 (0.82–0.93) | 99.93% / 99.91% |  |
| densenet121 | CINIC-10, 3 epochs | 539,904 | 315 | 330 | 0.95 (0.95–0.95) | 76.81% / 76.90% |  |
| densenet121 | synthetic, 1 epoch | 1,000,000 | 317 | 330 | 0.96 (0.96–0.96) | 99.98% / 99.96% |  |

PyTorch's configuration on every row was `torch-cudagraphs-fused`, except the LSTM rows (`torch-compile-fused`).

What to read from this:

- **The main comparison's direction holds over whole epochs of real data.** Every model AiDotNet trained clearly faster in the main comparison's CUDA rows, it also trained faster here. MobileNetV2 is slower in AiDotNet here too (0.82–0.83, against 0.89 in the main comparison), DenseNet-121 went from even (1.00) to slightly slower (0.95–0.96), and ViT is even. On the real datasets, both frameworks reached test accuracies within 2.2 points of each other on every model except ViT (3.3 points, below).
- **Some ratios are larger than in the main comparison.** ResNet-18 trains 4.8× faster here against 1.36× in the main comparison's CUDA rows, and ResNet-50 2.76× against 1.59×. In the first large-data attempt, at the same full clock as the main comparison, AiDotNet's ResNet-18 update took 13.5 ms against 40.9 ms in the main comparison, while PyTorch's took about the same in both. We have not yet established why. Until we have, read the main comparison as the conservative number.
- **The `unstable` rows** are the small, host-bound models on the synthetic data, where one launch of either framework ran much slower than the others (AiDotNet's third GRU launch ran at 27,983 samples/s against 103,205 and 134,456). Their medians agree with the EMNIST rows, whose launches were steady.
- **VGG-11 does not learn the synthetic task in either framework.** Both stay at chance (loss 2.3029, 9.82%), in identical form, with the same recipe that trains VGG-11 to 62–64% on CINIC-10. We report the row as measured.

### Large-data results, CPU

Same laptop and conditions, CPU only (eight threads per framework). Commit `709c001a`, three launches, finished at 20:32 UTC on November 25. The small models trained three epochs on their real dataset and one on their synthetic dataset; each vision model trained one epoch of CINIC-10.

| Model | Dataset | Samples per launch | AiDotNet samples/s | PyTorch samples/s | PyTorch time ÷ AiDotNet time | Test accuracy, AiDotNet / PyTorch | Flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| mlp | EMNIST Digits, 3 epochs | 720,000 | 62,744 | 18,445 | 3.40 (3.13–3.43) | 98.77% / 98.88% |  |
| mlp | synthetic, 1 epoch | 1,000,000 | 59,152 | 18,019 | 3.26 (3.13–3.64) | 99.99% / 100.00% |  |
| lstm | EMNIST rows, 3 epochs | 720,000 | 30,436 | 10,234 | 2.94 (2.88–2.98) | 98.84% / 98.83% |  |
| lstm | synthetic, 1 epoch | 1,000,000 | 29,337 | 10,077 | 2.81 (2.56–3.04) | 99.62% / 99.78% |  |
| gru | EMNIST rows, 3 epochs | 720,000 | 16,149 | 7,985 | 2.02 (2.00–2.03) | 98.89% / 98.88% |  |
| gru | synthetic, 1 epoch | 1,000,000 | 16,616 | 7,992 | 2.08 (2.00–2.18) | 99.48% / 99.52% |  |
| transformer | EMNIST rows, 3 epochs | 720,000 | 11,448 | 4,120 | 2.78 (2.77–2.78) | 94.44% / 94.85% |  |
| transformer | synthetic, 1 epoch | 1,000,000 | 11,377 | 4,064 | 2.75 (2.74–2.86) | 89.03% / 89.75% |  |
| cnn | EMNIST Digits, 3 epochs | 720,000 | 11,775 | 6,241 | 1.89 (1.85–1.91) | 99.15% / 99.23% |  |
| cnn | synthetic, 1 epoch | 1,000,000 | 11,105 | 5,916 | 1.88 (1.85–1.89) | 98.27% / 99.20% |  |
| resnet18 | CINIC-10, 1 epoch | 179,968 | 732 | 139 | 5.27 (5.09–5.47) | 54.65% / 55.36% |  |
| resnet50 | CINIC-10, 1 epoch | 179,968 | 287 | 91 | 3.19 (3.05–3.27) | 42.85% / 45.64% |  |
| vgg11 | CINIC-10, 1 epoch | 179,968 | 158 | 111 | 1.43 (1.39–1.46) | 54.02% / 53.30% |  |
| vit | CINIC-10, 1 epoch | 179,968 | 192 | 137 | 1.40 (1.38–1.41) | 41.59% / 42.52% |  |
| mobilenetv2 | CINIC-10, 1 epoch | 179,968 | 548 | 273 | 2.01 (1.99–2.02) | 45.28% / 47.85% |  |
| densenet121 | CINIC-10, 1 epoch | 179,968 | 34 | 26 | 1.31 (1.29–1.35) | 64.91% / 62.02% |  |

PyTorch's configuration was `torch-compile-fused` on every row except GRU (`torch-compile-fused-rnn`) and DenseNet-121 (`torch-compile`).

AiDotNet trained faster on every CPU row, by 1.31× (DenseNet-121) to 5.27× (ResNet-18), and both frameworks reached test accuracies within 2.9 points of each other. No CPU row was flagged. GRU and ViT training is not bit-reproducible on the CPU either: AiDotNet's three launches reported different first-epoch losses for both, while every other model repeated exactly.

| Run | Summary | Every epoch of every launch, reports, receipts and logs |
| --- | --- | --- |
| CUDA, November 24 | [CSV](evidence/2026-11-24-scale-cuda-summary.csv) | [ZIP](evidence/2026-11-24-scale-cuda.zip) |
| CUDA ViT after the fix, November 24 | [CSV](evidence/2026-11-24-scale-vit-cuda-summary.csv) | [ZIP](evidence/2026-11-24-scale-vit-cuda.zip) |
| CPU, November 25 | [CSV](evidence/2026-11-25-scale-cpu-summary.csv) | [ZIP](evidence/2026-11-25-scale-cpu.zip) |
| Stopped CUDA attempt at full clock (GPU faults), November 21 | | [ZIP](evidence/superseded-2026-11-21-scale-cuda-gpu-fault.zip) |

### The ViT fix

The first run showed AiDotNet's ViT at chance: 10.00% on CINIC-10 after three epochs, with its loss pinned at ln 10, while PyTorch reached 48.85%. The main comparison had reported ViT training as even (1.00) because it times 20 updates and never checks what the model learns. The cause was in AiDotNet: ViT's classification head applied a softmax, so the cross-entropy-with-logits loss trained on probabilities and its gradients vanished. Commit `709c001a` makes the head emit logits (Predict still returns probabilities), and also brings the encoder in line with the ViT paper: pre-LayerNorm blocks with a GELU MLP and a final LayerNorm before the head, as the PyTorch reference already had. One epoch of CINIC-10 then reached 40.89% against PyTorch's 41.16%. After three epochs AiDotNet is still 3.3 points behind (45.09% against 48.38%). The remaining structural difference is that AiDotNet's attention has no query, key and value biases (2,690,122 parameters against PyTorch's 2,693,578). Throughput did not change. The main comparison's ViT rows were rerun on the fixed code, and the tables above use the reruns.

### The GPU clock lock

This laptop's GPU computes wrong values at its normal boost clocks. In the first large-data attempt, both frameworks diverged to NaN on VGG-11, ResNet-18 and ResNet-50, at different points in identical runs. PyTorch with deterministic algorithms, run several times on the same data, produced different weights at 1500 MHz and identical weights at 1200 MHz. AiDotNet's training of these models is deterministic on the GPU, so identical launches must report identical losses, and they did at a locked 1200 MHz in repeated checks.

The runner therefore takes `--gpu-clock-lock MHZ`. The run declares the lock (set with `nvidia-smi -lgc 1200,1200` as administrator), refuses to start unless a load probe sustains it, and flags any launch that ran below 90% of it. The summary flags `ai-launches-disagree` when AiDotNet's launches of a row report different losses or accuracies. That cannot happen on healthy hardware for a deterministic model, so it marks a launch that computed wrong values. AiDotNet's GRU and ViT training is not bit-reproducible, on the GPU or the CPU (identical runs differ from the first epoch), so the flag is not raised for them.

Even at 1200 MHz the flag fired twice in 20 hours: VGG-11 on CINIC-10 and ResNet-50 on synthetic data, both in launch 1. Launches 2 and 3 agreed exactly with each other and, for VGG-11, with the earlier checks. A wrong value changes what the model learns (VGG-11's launch 1 reached 64.09% against 62.39%), not how long an update takes, and both rows' launch-to-launch ratio ranges are within 1%. They are kept as measured.

Locking the clock slows both frameworks by the same hardware limit. It does not favour either side, but absolute samples per second are lower than this GPU reaches at its stock clocks.

## Model definitions

An in-repo twin of the [AIsEval](https://github.com/ooples/AIsEval)
`aidotnet-benchmarks` harness, with one decisive difference: this project
references the AiDotNet **source** (`<ProjectReference Include="..\..\src\AiDotNet.csproj" />`),
not a published NuGet package. So it measures the **current working tree** —
the exact thing you want when validating a perf change before it ships.

That makes it the validation harness for changes a released-package benchmark
can't yet see, e.g.:

- **PR #1469** — the default-Adam fused-training gate (set `AIDOTNET_FUSED_DIAG=1`
  to print whether the compiled fused step actually engages: `Hit=True`).
- **`FeedForwardNeuralNetwork.Predict` → `IEngine.MlpForward`** fused-inference
  wiring — compare the `mlp` row (high-level `Predict`) against the `mlp-fused`
  row (direct kernel call). With the wiring in place they should converge.

Both sides build the same four reference models with matching layer shapes
(MLP / CNN / LSTM / Transformer), run the same training + multi-batch-inference
loop, and emit the same JSON schema. `pytorch/compare.py` lines the two reports
up row-by-row.

Seven more families run against their standard PyTorch definitions: `gru` (a GRU twin of the LSTM row; see the note
below), and at CIFAR
shape `[3, 32, 32]` with 10 classes `resnet18`, `resnet50`, `vgg11`, `mobilenetv2`, `densenet121` and `vit` (patch 4,
d 192, 6 layers, 3 heads). The AiDotNet side builds each through its own configuration class (`ResNetNetwork`,
`VGGNetwork`, ...); the PyTorch side uses torchvision, adjusted only where AiDotNet's builder adjusts for a 32x32 input
(VGG's classifier reads the 512x1x1 map directly, DenseNet uses the CIFAR stem). These rows have no step-1 weight map,
so `--verify-step1` skips them; each report records both parameter counts instead (they agree within 1%).

Four time-series forecasting rows run the shipped tensor-interface forecasters at the ETT shape (96-step lookback,
24-step horizon, 7 channels; N-BEATS univariate) with each paper's default sizes and Adam learning rate:
`informer` (`Informer`, d 512, ProbSparse, distilling), `autoformer` (`Autoformer`, d 512, auto-correlation),
`deepar` (`DeepAR`, 3x40 LSTM, Gaussian likelihood, forecast = median of 200 sample paths) and `nbeats`
(`NBEATSFinance`, 30 generic blocks of 4x512). `TimeSeriesRows.cs` wraps them; benchmark.py's twins write out the
official architectures layer for layer, and the parameter counts match exactly (trainable: informer 11,324,935, autoformer
10,531,847, deepar 36,542, nbeats 26,976,360). Neither side feeds time-feature marks (AiDotNet's tensor-interface forecasters take the history alone).

**The `gru` row is a timing comparison, not a parity row.** Both sides apply the reset gate after the recurrent
product, `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))` (AiDotNet's `GRULayer` default `resetAfter: true`), with the
same weight matrices. PyTorch's `nn.GRU` has a second, recurrent bias per gate that AiDotNet's `GRULayer` folds into one,
so the parameter counts differ by 192 and the losses are not expected to agree bit for bit. The PyTorch report carries
this as the row's `note`.

Four more rows wrap shipped model classes (`MoreFamilyRows.cs`), each against a layer-for-layer twin in
`benchmark.py` with the same trainable parameter count, loss and optimizer:

| row | AiDotNet class | shape | parameters |
|-----|----------------|-------|-----------:|
| `efficientnet-b0` | `EfficientNetNetwork` (B0 scaling) | CIFAR `[3, 32, 32]`, 10 classes | 4,020,358 |
| `vae` | `VariationalAutoencoder` | MNIST 784 -> 400 -> 20 -> 400 -> 784 | 652,824 |
| `gat` | `GraphAttentionNetwork` | Cora-sized graph: 2,708 nodes, 1,433 features, 8x8 heads, 7 classes | 92,373 |
| `dcgan` | `DCGAN` | 64x64 images, latent 100, one D + one G update per step | 6,350,852 |

`gat` treats the whole graph as one sample: run it with `--batch-size 1 --inference-batches 1` on both sides. Under
`--compile`, a twin with its own objective (`training_loss`, or the GAN's `train_step` networks) is compiled too, so
compiled training is what gets timed.

Three audio and speech rows (`AudioRows.cs`, beside the existing `whisper-tiny`) wrap shipped models at their papers'
shapes, each against a hand-written twin in `benchmark.py` (no torchaudio, librosa or transformers needed):

| row | AiDotNet class | shape, step | trainable parameters |
|-----|----------------|-------------|---------------------:|
| `panns-cnn14` | `PANNs` (CNN14) | 10 s AudioSet clips as `[1, 1001, 64]` log-mel, 527 tags, BCE on logits, batch 32 | 80,761,679 |
| `wav2vec2-base` | `Wav2Vec2Model` (BASE + CTC head) | 2 s of raw 16 kHz audio (32,000 samples, 99 frames), 24-symbol transcripts, CTC, feature encoder frozen | 90,197,410 |
| `hifigan-v1` | `HiFiGAN` (V1) | 256-frame mel (about 3 s at 22.05 kHz); one D + one G update on an 8192-sample segment | 84,660,721 |

PANNs takes features, so both sides build the log-mel batch outside the timed region; wav2vec 2.0 reads the waveform.
`wav2vec2-base` runs LayerDrop 0 on both sides (it skips whole layers at random, a graph break per layer under
`torch.compile`). `hifigan-v1` trains one utterance per step on both sides, as AiDotNet's HiFiGAN does: run it with
`--batch-size 1` (the paper's batch is 16). Its parameter count covers the generator (13,936,130) and both
discriminators.

Each training report also records `FirstLoss` / `LastLoss` (`first_loss` / `last_loss`): the loss of the first and
the last training step, as evidence the timed step trains. The last is read after timing. The first is read during
epoch 0, once, which the steady-state figures exclude; the PyTorch side holds it on the device until timing ends.
The C# report preserves non-finite endpoint losses as JSON strings (`"NaN"`, `"Infinity"`,
`"-Infinity"`), so divergence does not discard the report. Finite losses remain JSON numbers.
The comparison reader accepts these reports; the C# report types also deserialize those strings.

## 1. Run the AiDotNet side

```bash
# From the repo root. Release is mandatory for meaningful numbers.
# Match the thread count to the PyTorch side for a fair head-to-head.
set AIDOTNET_CPU_THREADS=8           # PowerShell: $env:AIDOTNET_CPU_THREADS=8
set AIDOTNET_FUSED_DIAG=1            # optional: print fused-path Hit/Miss

dotnet run -c Release --project benchmarks/AiDotNet.PyTorchParity -- \
    --models mlp,cnn,lstm,transformer,mlp-fused \
    --epochs 3 --train-batches 20 --batch-size 64 \
    --inference-iterations 100 --warmup-iterations 10 \
    --output benchmarks/AiDotNet.PyTorchParity/results/aidotnet.json
```

With `--device cpu` (the default) the harness pins the CPU engine via
`AiDotNetEngine.ResetToCpu()`, so an integrated GPU detected at startup cannot
take over a CPU row. `--device cuda` adopts the DirectGpu engine and refuses to
run on anything else; set `AIDOTNET_DIRECTGPU_BACKENDS=cuda` so it cannot fall
back to OpenCL or HIP, as `run_matrix.py` does.

## 2. Run the PyTorch side

```bash
cd benchmarks/AiDotNet.PyTorchParity/pytorch
pip install -r requirements.txt
python benchmark.py --models mlp,cnn,lstm,transformer --device cpu --compile \
    --threads 8 --output ../results/pytorch.json
```

Without flags PyTorch runs eager. `--compile` wraps each model in
`torch.compile` with static shapes; `--compile-mode reduce-overhead` adds CUDA
graphs on a CUDA device; `--optimizer-impl fused` selects fused AdamW. The
report's `protocol` block records which of these ran. `run_matrix.py` runs the
three compiled configurations listed above and compares AiDotNet with the
fastest. Pin both sides to the same thread count (`--threads` ↔
`AIDOTNET_CPU_THREADS`) and device. Record peak CPU RAM and GPU VRAM separately
from allocation volume.

## 3. Compare

```bash
cd benchmarks/AiDotNet.PyTorchParity/pytorch
python compare.py ../results/aidotnet.json ../results/pytorch.json
```

Prints a per-model / per-batch table with the latency ratio and verdict. The
gate (from AIsEval `Reporting/findings.md`) is **p95(AiDotNet) < mean(PyTorch)**:
our worst-of-95% steady-state latency still beats their average.

## CLI options (both sides)

| flag | default | meaning |
|------|---------|---------|
| `--models` | `mlp,cnn,lstm,transformer` | comma-separated subset of `mlp,cnn,lstm,gru,transformer,resnet18,resnet50,vgg11,mobilenetv2,densenet121,vit,informer,autoformer,deepar,nbeats` (and the other family rows); C# side also accepts `mlp-fused` |
| `--inference-batches` | `1,8,32,128` | inference batch sizes; for vision rows on a 6 GB card, explicitly pass `--inference-batches 1,8,16` to both harnesses (the default does not change by model) |
| `--epochs` | `3` | training epochs |
| `--train-batches` | `20` | batches per epoch |
| `--batch-size` | `64` | training batch size |
| `--inference-iterations` | `100` | steady-state inference iterations per batch size |
| `--warmup-iterations` | `10` | warmup iterations before measuring |
| `--warmup-min-ms` | `1000` | also warm each batch size for at least this long. .NET runs tier-0 (unoptimized) code until a method has 30 calls and no new tier-0 code has run for 100 ms, so a count-only warm-up timed tier-0 code for whichever batch ran first (#2386) |
| `--seed` | `1234` | RNG seed |
| `--output` | `results/{aidotnet,pytorch}.json` | report path |
| `--threads` (PyTorch) | `0` (all cores) | pin CPU threads; match `AIDOTNET_CPU_THREADS` |
| `--compile` (PyTorch) | off | `torch.compile` each model with static shapes |
| `--compile-mode` (PyTorch) | `default` | `default`, `reduce-overhead` (CUDA graphs), `max-autotune`, `max-autotune-no-cudagraphs` |
| `--optimizer-impl` (PyTorch) | `default` | AdamW implementation: PyTorch's default choice or `fused` |
| `--compile-rnn` (PyTorch) | off | under `--compile`, also compile `nn.LSTM`/`nn.GRU` (`torch._dynamo.config.allow_rnn`); without it they run eagerly |

Run directly, both harnesses measure inference at batch sizes **1, 8, 32, 128** by default (`--inference-batches`). `run_matrix.py` passes `1,8,16` so the vision rows fit a 6 GB GPU.

## Time-series forecasters (`--timeseries`)

The forecasters (`TimeSeriesModelBase` models such as N-BEATS) train through their own `Train(x, y)` loop over
windows of one series, so they do not fit the per-batch Forward/Backward/Step rows above. `--timeseries` times that
real loop instead: the AiDotNet side writes `series.csv` (a seasonal series with trend and noise) and trains each
forecaster on it, and `pytorch/timeseries.py` trains a twin with the same architecture, window, batch, loss, optimizer
and clipping on the same numbers.

```bash
# AiDotNet side (CPU or CUDA engine, as for the main rows)
set AIDOTNET_CPU_THREADS=8
dotnet run -c Release --project benchmarks/AiDotNet.PyTorchParity -- --timeseries results/ts --device cuda --epochs 6

# PyTorch side, compiled (the comparison bar)
cd benchmarks/AiDotNet.PyTorchParity/pytorch
python timeseries.py ../results/ts --device cuda --compile --epochs 6 --threads 8
```

- **Timing:** wall time per epoch. The AiDotNet side reads it from the model's training callback, so it includes the
  model's own window shuffling, batch building and best-epoch snapshot. Epoch 0 carries compilation and warm-up on
  both sides and is excluded; the report gives the median and IQR of the remaining epochs.
- **PyTorch gets its best setup:** `torch.compile` on the model, fused Adam on CUDA, windows gathered on the device
  from one resident series tensor, and the epoch loss read back once per epoch.
- **Rows:** `nbeats`, at `NBEATSModelOptions` defaults: 30 blocks of 4 x 512 ReLU layers, fixed polynomial basis of
  degree 2, lookback 10, horizon 5, batch 1024, Adam lr 1e-3, global-norm clip 1.0, MSE.
- **Loss evidence:** both reports carry the first and last epoch loss. Weight initialization differs (AiDotNet uses
  He-normal), so the curves are comparable in trend, not value.
- Options: `--epochs` (default 6, minimum 2), `--series-length` (AiDotNet side, default 20000), `--seed`, `--rows`.
  Reports: `aidotnet_timeseries_<device>.json` and `pytorch_timeseries_<device>[_compiled].json`.

## Notes

- `results/` is git-ignored (machine-specific timings don't belong in version control).
- `mlp-fused` is an AiDotNet-only primitive variant (direct `MlpForward`); the
  PyTorch side maps it to the same `MLP` so a shared `--models` list won't error.
- This project is excluded from `dotnet test` (`IsTestProject=false`); it's a
  manual harness you run on demand.

### Matched LSTM classification

Run these commands from this benchmark directory. Generate the seed-1234 fixture once. It contains six state-dict fields, 20 training batches of 64 labeled sequences, and inference batches of 1, 8 and 16 sequences.

```powershell
python pytorch/lstm_fixture.py --output results/lstm-fixture
dotnet run -c Release -- --models lstm --device cpu --inference-batches 1,8,16 --lstm-fixture results/lstm-fixture --lstm-numerics results/lstm-ai-cpu
python pytorch/benchmark.py --models lstm --device cpu --inference-batches 1,8,16 --lstm-fixture results/lstm-fixture --compile --lstm-cpu-mkldnn disabled --lstm-numerics results/lstm-torch-cpu --output results/lstm-torch-cpu/report.json
```

The numerical run writes the logits, scalar loss, gradients and learned parameters from three successive AdamW updates, followed by inference logits for each batch size. The raw FP32 arrays are in `observations.safetensors`. For timing, omit `--lstm-numerics`, set `--output`, and pass `--lstm-workload-receipt` with a fresh path. The normal timing loop makes 60 updates and each receipt binds the tensors actually used. For CUDA, select `--device cuda` on both commands and omit the CPU setting.

On Windows with PyTorch 2.14.1+cu126, the default oneDNN CPU LSTM decomposition sets the recurrent primitive's training flag to false. Its absent workspace fails Inductor's alignment check and AOT autograd's saved-tensor check during training. The explicit `--lstm-cpu-mkldnn disabled` setting selects the installed ATen decomposition and preserves full-graph Inductor compilation and its assertions. Both the compiler and the backend settings are recorded in the workload receipt. Passing `enabled` reproduces the oneDNN control on the affected runtime. Omitting the option preserves the installed PyTorch setting.
