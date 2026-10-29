# AiDotNet ⇄ PyTorch parity benchmark

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

**The `gru` row is a timing comparison, not a parity row.** PyTorch's `nn.GRU` applies the reset gate after the recurrent
product, `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`; AiDotNet's `GRULayer` applies it before,
`n = tanh(W_in x + b_in + W_hn (r * h) + b_hn)` (Cho et al., Keras `reset_after=False`). The sizes and parameter shapes
match, but the two compute different functions, so their losses are not expected to agree. The PyTorch report carries
this as the row's `note`.

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
| `--models` | `mlp,cnn,lstm,transformer` | comma-separated subset of `mlp,cnn,lstm,gru,transformer,resnet18,resnet50,vgg11,mobilenetv2,densenet121,vit`; C# side also accepts `mlp-fused` |
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

## Notes

- `results/` is git-ignored (machine-specific timings don't belong in version control).
- `mlp-fused` is an AiDotNet-only primitive variant (direct `MlpForward`); the
  PyTorch side maps it to the same `MLP` so a shared `--models` list won't error.
- This project is excluded from `dotnet test` (`IsTestProject=false`); it's a
  manual harness you run on demand.
