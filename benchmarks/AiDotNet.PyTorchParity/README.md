# AiDotNet ⇄ PyTorch parity benchmark

## Run all 11 model families

From the repository root, the following command builds this AiDotNet checkout in Release and runs its models against **Python PyTorch with `torch.compile`**. It covers MLP, CNN, LSTM, Transformer, GRU, ResNet-18, ResNet-50, VGG-11, MobileNetV2, DenseNet-121 and ViT. Each framework runs alone in a fresh process for every model, device and launch.

Use the .NET 10 SDK and a Python 3.13 virtual environment. On Windows, open the x64 Developer PowerShell for Visual Studio with the C++ build tools. For NVIDIA CUDA, install a compatible driver and CUDA development libraries. The published baseline used these wheels:

```shell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install psutil==7.2.2 numpy==2.5.4
```

On Windows CUDA also run `python -m pip install triton-windows==3.9.0.post30`. For a CPU-only machine, use the PyTorch CPU wheel index and pass `--devices cpu`.

```shell
python benchmarks/AiDotNet.PyTorchParity/run_matrix.py
```

The full comparison runs two launches on CUDA and CPU. Both frameworks use eight threads, seed 1234, and three epochs of 20 updates at batch size 64. Inference measures 100 calls at each of batches 1, 8 and 16 after warm-up. The runner alternates framework order between launches. For a quick execution check, run `python benchmarks/AiDotNet.PyTorchParity/run_matrix.py --models cnn --devices cpu --smoke`. Smoke timings only check execution.

The printed run directory contains the hardware and runtime versions, git revision, binary hashes, commands, exit codes, logs, raw JSON and a CSV with each launch's median training time per update and inference latency in milliseconds. In the CSV, `torch_over_ai` is PyTorch time divided by AiDotNet time. Failed pairs are listed in `completion.json` and the process exits nonzero. The runner first checks the supported models (MLP, CNN, LSTM and Transformer) against a same-weights PyTorch update on each requested device. The other seven rows compare timings and report parameter counts. The PyTorch implementation and its model-specific notes are in [benchmark.py](pytorch/benchmark.py).

By default the build uses the Tensors package pinned by this checkout. Use `--tensors /path/to/AiDotNet.Tensors` to test a source build and verify the loaded assembly's hash. Separate full historical runs are available for [November 13 and November 19](https://github.com/ooples/AiDotNet.Tensors/tree/bench/reproducible-model-families/parity/model-families). The [17-second LSTM recording](https://github.com/ooples/AiDotNet.Tensors/tree/bench/reproducible-model-families/parity/lstm-video) uses a different workload.

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
so `--verify-step1` skips them; each report records both parameter counts instead (they agree within 1.4%).

Four time-series forecasting rows run the shipped tensor-interface forecasters at the ETT shape (96-step lookback,
24-step horizon, 7 channels; N-BEATS univariate) with each paper's default sizes and Adam learning rate:
`informer` (`Informer`, d 512, ProbSparse, distilling), `autoformer` (`Autoformer`, d 512, auto-correlation),
`deepar` (`DeepAR`, 3x40 LSTM, Gaussian likelihood, forecast = median of 200 sample paths) and `nbeats`
(`NBEATSFinance`, 30 generic blocks of 4x512). `TimeSeriesRows.cs` wraps them; benchmark.py's twins write out the
official architectures layer for layer, and the parameter counts match exactly (trainable: informer 11,324,935, autoformer
10,531,847, deepar 36,542, nbeats 26,976,360). Neither side feeds time-feature marks (AiDotNet's tensor-interface forecasters take the history alone).

**The `gru` row is a timing comparison, not a parity row.** PyTorch's `nn.GRU` applies the reset gate after the recurrent
product, `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`; AiDotNet's `GRULayer` applies it before,
`n = tanh(W_in x + b_in + W_hn (r * h) + b_hn)` (Cho et al., Keras `reset_after=False`). The sizes and parameter shapes
match, but the two compute different functions, so their losses are not expected to agree. The PyTorch report carries
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

The harness pins the CPU engine via `AiDotNetEngine.ResetToCpu()` so the
comparison is CPU-vs-CPU (the integrated-GPU/OpenCL auto-detect path is slower
for these small/medium workloads and is not what the Tensors micro-benchmarks
beat PyTorch on).

## 2. Run the PyTorch side

```bash
cd benchmarks/AiDotNet.PyTorchParity/pytorch
pip install -r requirements.txt
python benchmark.py --models mlp,cnn,lstm,transformer --device cpu \
    --threads 8 --output ../results/pytorch.json
```

PyTorch runs **eager** here (no `torch.compile`) on purpose: the AiDotNet side
runs its own compiled/fused path, and pitting it against a separately compiled
PyTorch compares two compilation stacks rather than the kernels. Pin both sides
to the same thread count (`--threads` ↔ `AIDOTNET_CPU_THREADS`).

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

Inference is measured at batch sizes **1, 8, 32, 128** on both sides by default (`--inference-batches`).

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
