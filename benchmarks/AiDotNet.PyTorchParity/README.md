# AiDotNet ⇄ PyTorch parity benchmark

## Run the model families

From the repository root, the following command builds this AiDotNet checkout in Release and runs its models against **Python PyTorch with `torch.compile`**. It covers MLP, CNN, LSTM, Transformer, GRU, ResNet-18, ResNet-50, VGG-11, MobileNetV2, DenseNet-121 and ViT. Each framework runs alone in a fresh process for every model, device and launch.

Use the .NET 10 SDK and a Python 3.13 virtual environment. On Windows, open the x64 Developer PowerShell for Visual Studio with the C++ build tools. For NVIDIA CUDA, install a compatible driver and CUDA development libraries. The published baseline used these wheels:

```shell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install psutil==7.2.2 numpy==2.5.4
```

On Windows CUDA also run `python -m pip install triton-windows==3.9.0.post30`. For a CPU-only machine, use the PyTorch CPU wheel index and pass `--devices cpu`. The A-6 family rows also need `python -m pip install transformers` (the whisper-tiny twin), and `--classical` needs `python -m pip install scikit-learn xgboost lightgbm`.

```shell
python benchmarks/AiDotNet.PyTorchParity/run_matrix.py
```

The full comparison runs nine launches on CUDA and CPU, and every row reports the median and interquartile range (IQR) over them. Both frameworks use eight threads, seed 1234, and three epochs of 20 updates at batch size 64. Inference measures 100 calls at each of batches 1, 8 and 16 after warm-up. The runner rotates framework order between launches. For a quick execution check, run `python benchmarks/AiDotNet.PyTorchParity/run_matrix.py --models cnn --devices cpu --smoke`. Smoke timings only check execution.

The printed run directory contains the hardware and runtime versions, git revision, binary hashes, commands, exit codes, logs, raw JSON and a CSV with each launch's median training time per update and inference latency in milliseconds. In the CSV, `torch_over_ai` is PyTorch time divided by AiDotNet time. Failed pairs are listed in `completion.json` and the process exits nonzero. The runner first checks the supported models (MLP, CNN, LSTM and Transformer) against a same-weights PyTorch update on each requested device. The other seven rows compare timings and report parameter counts. The PyTorch implementation and its model-specific notes are in [benchmark.py](pytorch/benchmark.py).

### A-6 family rows (#2395)

`--models families` (or `all`, or any of the row names) adds eight rows. Each runs at its paper's input shape, so it carries its own batch sizes; the heavy rows time fewer updates and inference calls per launch:

| Row | Family | Training batch | Inference batches | Updates per epoch | Inference calls |
| --- | --- | ---: | --- | ---: | ---: |
| `ddpm` | diffusion (Ho et al. CIFAR-10 U-Net, one DDPM step) | 4 | 1, 4 | 5 | 10 |
| `decoder-lm` | decoder-only (GPT-style) language model | 64 | 1, 8, 16 | 20 | 100 |
| `whisper-tiny` | ASR, encoder-decoder (vs transformers' Whisper) | 4 | 1, 4 | 5 | 10 |
| `wav2vec2-base` | ASR, CTC | 4 | 1, 4 | 5 | 10 |
| `hifigan-v1` | TTS vocoder (GAN) | 1 | 1 | 5 | 10 |
| `faster-rcnn` | detection (ResNet-50 FPN, vs torchvision) | 1 | 1 | 10 | 10 |
| `gcn` | GNN, graph convolution | 64 | 1, 8, 16 | 20 | 100 |
| `gat` | GNN, graph attention (Cora-sized graph) | 1 | 1 | 20 | 100 |

Every row's table entry carries both parameter counts and the median first and last training loss of both sides. The summary flags `parameters-differ` and `loss-not-falling`. One family twin differs by design: faster-rcnn's RPN objectness is Ren et al.'s 2k-way softmax in AiDotNet and torchvision's k sigmoid scores (771 parameters).

### Locked GPU clocks

A CUDA launch samples the GPU every 0.5 s with `nvidia-smi -lms 500` into `<launch>.clocks.csv`. Its receipt records the SM and memory clocks under load (samples at 50% utilization or more). A launch is flagged `clock-varied` when the loaded SM clock moved more than 3%, and `clock-unlocked` when it missed a requested lock by more than 3%. From an elevated prompt, `--lock-gpu-clock MHZ` and `--lock-memory-clock MHZ` run `nvidia-smi -lgc MHZ,MHZ` and `-lmc MHZ,MHZ` before the launches and `-rgc` / `-rmc` after them. A laptop driver can refuse a lock. The refusal is then recorded in `protocol.json`, the run continues, and the per-launch clocks show what ran.

```shell
python benchmarks/AiDotNet.PyTorchParity/run_matrix.py --models all --devices cuda --lock-gpu-clock 1800 --lock-memory-clock 7001
```

### Classical ML (`--classical`)

`run_matrix.py --classical` runs `ClassicalBench.cs` against `pytorch/classical.py` (scikit-learn, XGBoost and LightGBM) for nine launches, alternating which side goes first. Both sides fit the same 20,000 x 20 CSV, scored on 5,000 held-out rows. Each launch fits every row `--classical-repeats` times (default 3) and reports the median. The summary gives the median and IQR of those per-launch medians, and the fastest Python library's fit time divided by AiDotNet's. The all-core rows use every core on both sides and the `_1job` rows use one thread. Classical ML runs on the CPU only.

### A-6 CPU results (2026-11-25)

Intel Core i7-13850HX, 8 threads, PyTorch 2.14.1 with `torch.compile`. Nine launches each, unlocked laptop (no CI job during any launch). Most rows carry the `unstable` flag: AiDotNet's slowest launch was over 1.5x its fastest on those rows, while PyTorch's stayed within a few percent. faster-rcnn's `parameters-differ` is the RPN objectness difference above. The CUDA table waits for a locked-clock run.

**CPU**: milliseconds per training update or inference call, median (interquartile range) over 9 launches. The ratio is PyTorch time ÷ AiDotNet time (IQR of the per-launch ratios); above 1 means AiDotNet took less time. PyTorch is the fastest configuration for each row. ⚑ marks a row with a flag in summary.csv.

| Model | Operation | Batch | AiDotNet ms | PyTorch ms | PyTorch ÷ AiDotNet | Parameters (AiDotNet / PyTorch) | Loss first → last (AiDotNet; PyTorch) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ddpm | train | 4 | 296 (231–304) | 486 (475–505) torch-compile-fused | 1.64 (1.58–2.13) ⚑ | 30,734,851 / 30,734,851 | 0.9907 → 0.2891; 0.9812 → 0.4611 |
| ddpm | inference | 1 | 35.6 (24.9–39.1) | 41.9 (41.6–43.7) torch-compile | 1.18 (1.12–1.69) ⚑ | 30,734,851 / 30,734,851 |  |
| ddpm | inference | 4 | 88.5 (57.5–93.1) | 141 (138–153) torch-compile | 1.60 (1.55–2.36) ⚑ | 30,734,851 / 30,734,851 |  |
| decoder-lm | train | 64 | 63 (60.3–84) | 60.5 (59.8–61.5) torch-compile-fused | 0.96 (0.73–1.00) ⚑ | 526,976 / 526,976 | 6.383 → 0.0559; 6.358 → 0.105 |
| decoder-lm | inference | 1 | 0.558 (0.526–1.08) | 0.716 (0.579–0.749) torch-compile | 1.28 (0.67–1.34) ⚑ | 526,976 / 526,976 |  |
| decoder-lm | inference | 8 | 2.09 (2.01–3.7) | 2.58 (2.49–2.65) torch-compile | 1.24 (0.71–1.24) ⚑ | 526,976 / 526,976 |  |
| decoder-lm | inference | 16 | 4 (3.87–7.09) | 4.49 (4.31–4.63) torch-compile | 1.12 (0.65–1.16) ⚑ | 526,976 / 526,976 |  |
| whisper-tiny | train | 4 | 1,840 (1,232–1,986) | 2,055 (2,038–2,065) torch-compile-fused | 1.12 (1.04–1.65) ⚑ | 37,184,640 / 37,184,640 | 10.92 → 7.305; 10.97 → 7.367 |
| whisper-tiny | inference | 1 | 118 (112–185) | 168 (165–172) torch-compile | 1.42 (0.94–1.52) ⚑ | 37,184,640 / 37,184,640 |  |
| whisper-tiny | inference | 4 | 660 (451–721) | 649 (643–659) torch-compile | 0.98 (0.87–1.45) ⚑ | 37,184,640 / 37,184,640 |  |
| wav2vec2-base | train | 4 | 1,488 (1,239–1,596) | 1,352 (1,329–1,365) torch-compile-fused | 0.91 (0.84–1.09) | 90,197,410 / 90,197,410 | 284.5 → 85.33; 258.9 → 86.4 |
| wav2vec2-base | inference | 1 | 195 (140–221) | 123 (122–130) torch-compile-fused | 0.63 (0.56–0.89) ⚑ | 90,197,410 / 90,197,410 |  |
| wav2vec2-base | inference | 4 | 614 (500–730) | 415 (406–416) torch-compile-fused | 0.68 (0.57–0.80) | 90,197,410 / 90,197,410 |  |
| hifigan-v1 | train | 1 | 2,276 (1,580–2,363) | 2,144 (2,120–2,175) torch-compile | 0.94 (0.93–1.36) ⚑ | 84,660,721 / 84,660,721 | 133.2 → 45.99; 339.6 → 55.65 |
| hifigan-v1 | inference | 1 | 387 (363–596) | 1,017 (975–1,036) torch-compile-fused | 2.63 (1.62–2.85) ⚑ | 84,660,721 / 84,660,721 |  |
| faster-rcnn | train | 1 | 522 (453–536) | 1,162 (1,155–1,206) torch-compile | 2.23 (2.21–2.85) ⚑ | 41,809,177 / 41,808,406 | 6.393 → 0.903; 5.42 → 1.804 |
| faster-rcnn | inference | 1 | 209 (143–213) | 355 (343–376) torch-compile | 1.70 (1.67–2.47) ⚑ | 41,809,177 / 41,808,406 |  |
| gcn | train | 64 | 1.28 (1.26–1.34) | 1.87 (1.65–1.89) torch-compile-fused | 1.46 (1.28–1.50) | 5,898 / 5,898 | 2.313 → 2.195; 2.306 → 2.227 |
| gcn | inference | 1 | 0.019 (0.019–0.02) | 0.062 (0.062–0.14) torch-compile | 3.26 (3.26–7.30) ⚑ | 5,898 / 5,898 |  |
| gcn | inference | 8 | 0.067 (0.065–0.071) | 0.084 (0.083–0.161) torch-compile-fused | 1.25 (1.24–2.27) ⚑ | 5,898 / 5,898 |  |
| gcn | inference | 16 | 0.128 (0.075–0.13) | 0.178 (0.101–0.181) torch-compile | 1.39 (1.33–1.44) ⚑ | 5,898 / 5,898 |  |
| gat | train | 1 | 16 (15.7–16.3) | 11.1 (10.9–11.2) torch-compile-fused | 0.70 (0.68–0.70) ⚑ | 92,373 / 92,373 | 2.154 → 1.922; 1.958 → 1.88 |
| gat | inference | 1 | 2.54 (2.44–2.85) | 3.9 (3.82–4.36) torch-compile-fused | 1.53 (1.36–1.61) ⚑ | 92,373 / 92,373 |  |

**CPU, classical ML**: fit and predict milliseconds, median (interquartile range) over 9 launches of each launch's median. The ratio is the fastest Python library's fit time ÷ AiDotNet's; above 1 means AiDotNet fit faster. Quality is held-out accuracy, R² or k-means inertia. ⚑ marks a row with a flag in classical_summary.csv.

| Row | AiDotNet fit | scikit-learn fit | XGBoost fit | LightGBM fit | Fastest ÷ AiDotNet | AiDotNet predict | scikit-learn predict | Quality (AiDotNet / scikit-learn / XGBoost / LightGBM) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| random_forest | 41,471 (41,003–41,548) | 4,183 (4,154–4,195) | – | – | 0.10 (scikit-learn) | 70.2 (69.7–71) | 53.2 (52.5–53.4) | accuracy: 0.8176 / 0.816 / – / – |
| random_forest_all_cores | 14,828 (14,700–15,162) | 384 (383–392) | – | – | 0.03 (scikit-learn) | 24.6 (24.5–25.7) | 32.1 (31.3–32.4) | accuracy: 0.8176 / 0.816 / – / – |
| hist_gradient_boosting | 286 (271–318) | 245 (235–255) | 126 (124–129) | 129 (128–132) | 0.44 (xgboost) | 5.22 (4.6–6.51) | 2.83 (2.73–2.84) | accuracy: 0.8252 / 0.8246 / 0.8274 / 0.8316 |
| hist_gradient_boosting_1job | 289 (283–292) | 331 (330–334) | 424 (422–427) | 216 (213–219) | 0.75 (lightgbm) | 10.9 (10.6–11.2) | 13.3 (13.2–13.6) | accuracy: 0.8252 / 0.8246 / 0.8274 / 0.8316 |
| logistic_regression | 3.55 (3.35–3.65) | 4.4 (4.15–4.64) | – | – | 1.24 (scikit-learn) | 0.173 (0.172–0.176) | 0.222 (0.214–0.226) | accuracy: 0.642 / 0.642 / – / – |
| ridge | 2.8 (2.8–2.91) | 2.45 (2.37–2.47) | – | – | 0.87 (scikit-learn) ⚑ | 0.0872 (0.0849–0.0922) | 0.155 (0.145–0.158) | r2: 0.997 / 0.997 / – / – |
| kmeans | 660 (633–668) | 60.2 (59.5–63.2) | – | – | 0.09 (scikit-learn) | – | – | inertia: 3.502e+05 / 3.502e+05 / – / – |

By default the build uses the Tensors package pinned by this checkout. Use `--tensors /path/to/AiDotNet.Tensors` to test a source build and verify the loaded assembly's hash.

### Compare to the archived runs

Each archive contains 88 original framework reports covering every model, device and launch. The CSVs give absolute times and parameter counts for each launch. The November 13 baseline used public mainline commits. The November 19 experiment used an integration of open PRs and the laptop's power state varied. Both include losses and wins. [Historical source revisions and reproduction commands](https://github.com/ooples/AiDotNet.Tensors/tree/a2577f0b1/parity/model-families)

| Run | Training and inference | Individual framework reports |
| --- | --- | --- |
| November 13 | [CSV](evidence/2026-11-13.csv) | [ZIP](evidence/2026-11-13.zip) |
| November 19 | [CSV](evidence/2026-11-19.csv) | [ZIP](evidence/2026-11-19.zip) |

For example, from the repository root, inspect the original CUDA GRU pair:

```shell
python -m zipfile -e benchmarks/AiDotNet.PyTorchParity/evidence/2026-11-19.zip benchmarks/AiDotNet.PyTorchParity/results/archive-19
python benchmarks/AiDotNet.PyTorchParity/pytorch/compare.py benchmarks/AiDotNet.PyTorchParity/results/archive-19/aidotnet-1-cuda-gru.json benchmarks/AiDotNet.PyTorchParity/results/archive-19/torch-compile-1-cuda-gru.json
```

The [17-second LSTM video benchmark](https://github.com/ooples/AiDotNet.Tensors/tree/a2577f0b1/parity/lstm-video) uses a separate fixture and training protocol.

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

`gat` treats the whole graph as one sample: run it with `--batch-size 1 --inference-batches 1` on both sides. Both sides evaluate attention over the graph's 13,540 stored entries: AiDotNet's layer through its CSR kernel, the twin through `EdgeGATLayer` (gather, per-row softmax by scatter, aggregation by `index_add`, as PyG's `GATConv`). `test_gat_twin.py` checks the edge layer's outputs and gradients against the dense masked reference. The first A-6 run used that dense reference as the twin, which computed every one of the 2,708² node pairs (542x the attention work), and reported a 45.5x training ratio that measured dense against sparse. The gat rows above are a re-run on 2026-11-25 with the edge twin (flagged `busy-cpu`: other processes held the CPU during some launches; no CI job ran). Under
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

The harness pins the CPU engine via `AiDotNetEngine.ResetToCpu()` so the
comparison is CPU-vs-CPU (the integrated-GPU/OpenCL auto-detect path is slower
for these small/medium workloads and is not what the Tensors micro-benchmarks
beat PyTorch on).

## 2. Run the PyTorch side

```bash
cd benchmarks/AiDotNet.PyTorchParity/pytorch
pip install -r requirements.txt
python benchmark.py --models mlp,cnn,lstm,transformer --device cpu --compile \
    --threads 8 --output ../results/pytorch.json
```

The performance target is **`torch.compile`**, including its first epoch and
steady-state training and inference. Pass `--compile` explicitly; omitting it
measures an eager baseline and does not establish a win against compiled PyTorch.
Pin both sides to the same thread count (`--threads` ↔ `AIDOTNET_CPU_THREADS`)
and device. Record peak CPU RAM and GPU VRAM separately from allocation volume.

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

### Matched LSTM classification

Run these commands from this benchmark directory. Generate the seed-1234 fixture once. It contains six state-dict fields, 20 training batches of 64 labeled sequences, and inference batches of 1, 8 and 16 sequences.

```powershell
python pytorch/lstm_fixture.py --output results/lstm-fixture
dotnet run -c Release -- --models lstm --device cpu --inference-batches 1,8,16 --lstm-fixture results/lstm-fixture --lstm-numerics results/lstm-ai-cpu
python pytorch/benchmark.py --models lstm --device cpu --inference-batches 1,8,16 --lstm-fixture results/lstm-fixture --compile --lstm-cpu-mkldnn disabled --lstm-numerics results/lstm-torch-cpu --output results/lstm-torch-cpu/report.json
```

The numerical run writes the logits, scalar loss, gradients and learned parameters from three successive AdamW updates, followed by inference logits for each batch size. The raw FP32 arrays are in `observations.safetensors`. For timing, omit `--lstm-numerics`, set `--output`, and pass `--lstm-workload-receipt` with a fresh path. The normal timing loop makes 60 updates and each receipt binds the tensors actually used. For CUDA, select `--device cuda` on both commands and omit the CPU setting.

On Windows with PyTorch 2.14.1+cu126, the default oneDNN CPU LSTM decomposition sets the recurrent primitive's training flag to false. Its absent workspace fails Inductor's alignment check and AOT autograd's saved-tensor check during training. The explicit `--lstm-cpu-mkldnn disabled` setting selects the installed ATen decomposition and preserves full-graph Inductor compilation and its assertions. Both the compiler and the backend settings are recorded in the workload receipt. Passing `enabled` reproduces the oneDNN control on the affected runtime. Omitting the option preserves the installed PyTorch setting.
