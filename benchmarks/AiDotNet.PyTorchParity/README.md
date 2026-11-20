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
- **Same function where it can be checked.** Before any timing, MLP, CNN, LSTM and Transformer must reproduce PyTorch's first update from the same weights on each device. The check covers the forward pass, loss, gradients and parameter change. The other seven rows are timing comparisons of the same architecture. Three have slightly different parameter counts: GRU (AiDotNet 19,274, PyTorch 19,466), ViT (2,689,738 and 2,693,578) and MobileNetV2 (2,253,738 and 2,236,682). Those rows are flagged `parameters-differ`. [benchmark.py](pytorch/benchmark.py) gives the PyTorch definitions and model notes.
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

### Results from November 20, 2026

The machine was a Ryzen 7 4800H laptop on AC power, with a 6 GB GTX 1660 Ti, NVIDIA driver 617.42, Windows 11, .NET 10 and PyTorch 2.14.1+cu126. The run used commit `37597f19` with AiDotNet.Tensors 0.151.0 from NuGet and five launches. Each row compares AiDotNet with PyTorch's fastest configuration for that row. GRU, ViT and MobileNetV2 carry ⚑ on every row because their parameter counts differ slightly (`parameters-differ`).

**CUDA**: PyTorch time ÷ AiDotNet time, median of launches (range of launches). Above 1 means AiDotNet took less time. ⚑ marks a row with a flag in summary.csv.

| Model | Training, batch 64 | Inference, batch 1 | Inference, batch 8 | Inference, batch 16 |
| --- | ---: | ---: | ---: | ---: |
| mlp | 6.53 (5.92–7.33) | 1.43 (0.92–1.81) ⚑ | 1.67 (1.60–1.78) | 1.63 (0.69–1.75) ⚑ |
| lstm | 6.55 (6.14–6.67) | 1.48 (1.01–1.60) ⚑ | 1.91 (1.75–1.97) | 1.83 (1.75–2.70) ⚑ |
| transformer | 2.06 (1.85–2.35) | 0.40 (0.34–0.54) ⚑ | 0.57 (0.53–0.67) | 0.71 (0.66–1.08) ⚑ |
| cnn | 2.37 (2.14–2.56) | 0.87 (0.30–1.30) ⚑ | 1.41 (1.34–2.12) ⚑ | 1.50 (1.25–1.58) |
| resnet18 | 1.30 (1.18–1.54) | 2.12 (1.95–2.51) | 1.99 (0.93–2.12) ⚑ | 2.00 (1.38–2.10) |
| resnet50 | 1.71 (1.51–1.80) | 1.42 (1.33–1.47) | 1.48 (1.33–1.51) | 1.32 (1.30–1.36) |
| vgg11 | 1.18 (1.16–1.18) | 2.21 (2.13–2.33) | 1.69 (1.54–1.77) | 1.41 (1.39–1.47) |
| vit | 1.02 (1.02–1.02) ⚑ | 0.62 (0.55–0.66) ⚑ | 1.01 (0.83–1.04) ⚑ | 0.93 (0.92–0.95) ⚑ |
| gru | 8.73 (7.99–10.06) ⚑ | 1.73 (1.47–2.49) ⚑ | 2.29 (2.24–3.17) ⚑ | 2.65 (1.67–2.82) ⚑ |
| mobilenetv2 | 0.61 (0.58–0.62) ⚑ | 1.77 (1.26–1.79) ⚑ | 0.96 (0.93–2.90) ⚑ | 0.86 (0.81–0.89) ⚑ |
| densenet121 | 1.00 (0.99–1.01) | 1.34 (1.27–1.49) | 1.70 (1.69–1.71) | 1.31 (1.30–1.31) |

**CPU**: PyTorch time ÷ AiDotNet time, median of launches (range of launches). Above 1 means AiDotNet took less time. ⚑ marks a row with a flag in summary.csv.

| Model | Training, batch 64 | Inference, batch 1 | Inference, batch 8 | Inference, batch 16 |
| --- | ---: | ---: | ---: | ---: |
| mlp | 2.65 (2.55–3.00) | 3.95 (3.73–4.10) | 4.28 (3.54–4.95) | 4.72 (4.17–4.82) |
| lstm | 2.28 (2.10–2.30) | 3.97 (3.85–4.09) | 1.58 (1.46–2.30) | 1.93 (1.75–2.09) |
| transformer | 1.65 (1.34–1.73) | 2.18 (2.11–2.28) | 3.00 (2.89–3.11) | 2.77 (2.31–3.05) |
| cnn | 1.14 (0.98–1.59) | 3.86 (3.49–5.18) | 2.53 (2.44–2.91) | 1.67 (1.36–2.52) ⚑ |
| resnet18 | 4.67 (4.02–4.67) | 2.18 (1.96–2.59) | 1.95 (1.72–2.05) | 1.63 (1.45–1.78) |
| resnet50 | 3.06 (2.86–3.22) | 1.68 (1.33–1.98) ⚑ | 1.55 (1.42–1.97) ⚑ | 1.60 (1.40–1.60) |
| vgg11 | 1.43 (1.39–1.50) | 2.01 (1.95–2.26) | 2.88 (2.74–3.48) | 1.26 (1.23–1.50) |
| vit | 1.24 (1.13–1.41) ⚑ | 1.28 (1.19–1.92) ⚑ | 1.09 (0.96–1.23) ⚑ | 1.34 (1.22–1.46) ⚑ |
| gru | 2.35 (1.90–2.56) ⚑ | 15.40 (11.49–16.09) ⚑ | 6.59 (6.47–7.33) ⚑ | 4.77 (4.66–5.77) ⚑ |
| mobilenetv2 | 1.73 (1.63–1.80) ⚑ | 2.54 (2.22–2.67) ⚑ | 1.94 (1.84–2.09) ⚑ | 1.36 (1.27–1.50) ⚑ |
| densenet121 | 1.24 (1.08–1.31) | 1.39 (1.09–1.58) | 1.21 (0.91–1.36) | 1.29 (1.17–1.49) |

Where AiDotNet took longer, or the result is too close to call:

- **CUDA training.** MobileNetV2 is slower (0.61). DenseNet-121 (1.00) and ViT (1.02) are even.
- **CUDA inference.** The Transformer is slower at every batch size (0.40–0.71). ViT is slower at batch 1 (0.62) and batch 16 (0.93). MobileNetV2 is slower at batches 8 and 16 (0.96, 0.86), and the CNN at batch 1 (0.87).
- **CPU.** AiDotNet's median is lower on every row. On some rows a single launch went the other way: CNN training (0.98), ViT at batch 8 (0.96) and DenseNet-121 at batch 8 (0.91).

Most `unstable` flags are on CUDA inference calls that take less than a millisecond. At that scale, one launch of either framework can take up to three times as long as another. The range column shows how far the launches spread. No row was flagged `battery`, `busy-cpu`, `busy-gpu`, `ci-job` or `missing-launches`. The run was interrupted once by a switch to battery power. Its one battery-flagged process is kept under `superseded/` in the ZIP, and the process was rerun on AC power.

The fastest PyTorch configuration for most CUDA rows was `torch-cudagraphs-fused` (10 of 11 training rows and 28 of 33 inference rows). On CPU it was `torch-compile-fused` for every training row.

These results come from one Windows laptop. On Windows, PyTorch's GPU compiler uses the community `triton-windows` build, and Windows display-driver scheduling adds GPU launch overhead that Linux does not have. We have not run this on Linux yet, so these numbers do not show how the two frameworks compare there.

| Run | Summary | Every launch | Every report, receipt and log |
| --- | --- | --- | --- |
| November 20 | [Table](evidence/2026-11-20.md) · [CSV](evidence/2026-11-20-summary.csv) | [CSV](evidence/2026-11-20.csv) | [ZIP](evidence/2026-11-20.zip) |

To rebuild the tables from the raw reports, run `python -m zipfile -e benchmarks/AiDotNet.PyTorchParity/evidence/2026-11-20.zip benchmarks/AiDotNet.PyTorchParity/results/archive-20`, then `python benchmarks/AiDotNet.PyTorchParity/run_matrix.py --summarize benchmarks/AiDotNet.PyTorchParity/results/archive-20`.

### Archived runs from November 13 and 19

These two runs came before this runner. Their reports are kept exactly as recorded. Each archive holds the 88 original framework reports: 11 models, two devices, two frameworks and two launches. Do not use them for performance claims. They differ from the current protocol in these ways:

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
python benchmark.py --models mlp,cnn,lstm,transformer --device cpu \
    --threads 8 --output ../results/pytorch.json
```

Without flags PyTorch runs eager. `--compile` wraps each model in
`torch.compile` with static shapes; `--compile-mode reduce-overhead` adds CUDA
graphs on a CUDA device; `--optimizer-impl fused` selects fused AdamW. The
report's `protocol` block records which of these ran. `run_matrix.py` runs the
three compiled configurations listed above and compares AiDotNet with the
fastest. Pin both sides to the same thread count (`--threads` ↔
`AIDOTNET_CPU_THREADS`).

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
