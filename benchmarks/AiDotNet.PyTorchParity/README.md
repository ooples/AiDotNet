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

## 1. Run the AiDotNet side

```bash
# From the repo root. Release is mandatory for meaningful numbers.
# Match the thread count to the PyTorch side for a fair head-to-head.
set AIDOTNET_BLAS_THREADS=8          # PowerShell: $env:AIDOTNET_BLAS_THREADS=8
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

By default PyTorch runs **eager** (no `torch.compile`, PyTorch's default AdamW):
the kernel-vs-kernel comparison. `--compile` (with `--compile-backend` /
`--compile-mode`) and `--optimizer-impl fused` select the other modes; the
scoreboard below runs all of them and scores against the fastest. Pin both sides
to the same thread count (`--threads` ↔ `AIDOTNET_BLAS_THREADS`) for a
kernel-level comparison.

## 3. Compare

```bash
cd benchmarks/AiDotNet.PyTorchParity/pytorch
python compare.py ../results/aidotnet.json ../results/pytorch.json
```

Prints a per-model / per-batch table with the latency ratio and verdict. The
gate (from AIsEval `Reporting/findings.md`) is **p95(AiDotNet) < mean(PyTorch)**:
our worst-of-95% steady-state latency still beats their average.

## 4. Scoreboard: every family × device vs the best PyTorch mode

`pytorch/scoreboard.py` is the trustworthy training number: AiDotNet vs PyTorch,
training **ms/step**, for MLP / CNN / LSTM / Transformer on **cpu and cuda**, with
PyTorch in its **fastest mode for each cell**. It writes `results/scoreboard.md`
and `results/scoreboard.json` (every raw run, the mode-selection table, skipped
modes with reasons, machine + DLL provenance). These two files are the one
deliberate exception to `results/` being git-ignored: commit them when you
refresh the scoreboard so the history shows how the gap moves.

```powershell
# From benchmarks/AiDotNet.PyTorchParity/pytorch, with a python that has torch (+CUDA) and psutil.
# --ours-bin is a BUILT harness directory: the shared baseline, a perf track's B side, or
# benchmarks/AiDotNet.PyTorchParity/bin/Release/net10.0 after `dotnet build -c Release`.
python scoreboard.py --ours-bin C:\Users\yolan\source\repos\_bench\baseline-bin
# Subsets while iterating (same protocol, fewer cells):
python scoreboard.py --ours-bin <dir> --models cnn,lstm --devices cpu
```

Protocol:

- **One run = one fresh process**: 5 epochs × 20 steps × batch 64 (epoch 0 is
  warmup); the run's value is the median steady-state epoch / steps, i.e. the
  same training row `compare.py` prints. AiDotNet runs with one inference
  iteration (the C# harness has no training-only switch); PyTorch runs with
  `--skip-inference`.
- **Every run holds `Global\AiDotNetBenchLock`** (the mutex `_bench/bench.ps1`
  takes) for its own duration only, so other benchmark tracks interleave between
  runs and no two timed runs ever overlap.
- **Machine noise**: builds and test hosts of other tracks do not take the lock
  (on the shared box they inflated CPU steps up to 6x), so three more defences
  apply to both sides alike. Every timed process starts at `--priority`
  (`above-normal`), so normal-priority builds yield the CPUs to it. Before a run,
  with the lock held, the system CPU load is sampled; above
  `--max-background-pct` (25% of all logical CPUs) the run waits with the lock
  released, up to `--quiet-wait` seconds. After the run, the CPU used by OTHER
  processes (system busy time minus the run's whole process tree, read from a
  Win32 job object so TorchInductor compile workers count as the run's own) is
  computed; a run above the threshold is discarded and redone up to
  `--max-retries` (2) times, then kept and flagged. The *Machine noise* table
  lists every contender's background load and discarded runs.
- **Mode selection**: every PyTorch mode available on the device runs 3× —
  `eager`, `compile[inductor]` (CPU needs MSVC `cl.exe`; found via vswhere /
  `--vcvars`; GPU needs `triton-windows`), and on CUDA
  `compile[inductor/reduce-overhead]` (CUDA graphs) and `compile[cudagraphs]`
  (CUDA graphs without Inductor), each with PyTorch's default AdamW and with
  `fused` AdamW. Unavailable or failing modes are listed with the reason.
  `max-autotune` and `aot_eager` are not run (reasons in the output).
- **Scoring**: AiDotNet, the 2 best selection modes and eager each run **9×**,
  interleaved round by round; the finalist with the lowest median is the bar.
  Re-measuring from scratch keeps a lucky selection run from biasing it.
- **Verdict** as in `compare.py`: `WIN`/`LOSE` only when the p25–p75 ranges of
  the 9 runs do not overlap; lowercase means the medians differ within noise.

Flags: `--runs` (9), `--select-runs` (3), `--finalists` (2), `--modes` (subset of
mode labels; eager is always included), `--epochs` (5), `--threads` (0 = each side
at its default; N pins torch `--threads` and `AIDOTNET_BLAS_THREADS`), `--python`,
`--vcvars`, `--output-dir` (`results/`), `--priority`, `--max-background-pct`,
`--max-retries`, `--quiet-wait`, `--quiet-poll`, `--timeout` (1800 s per run),
`--snapshot` / `--no-snapshot` (default on: AiDotNet runs from a private copy of
`--ours-bin` taken at the start, so a shared baseline refreshed mid-sweep cannot
mix two builds into one scoreboard; provenance hashes that copy).
Windows only (the lock is a Win32 named
mutex).

## CLI options (both sides)

| flag | default | meaning |
|------|---------|---------|
| `--models` | `mlp,cnn,lstm,transformer` | comma-separated subset; C# side also accepts `mlp-fused` |
| `--epochs` | `3` | training epochs |
| `--train-batches` | `20` | batches per epoch |
| `--batch-size` | `64` | training batch size |
| `--inference-iterations` | `100` | steady-state inference iterations per batch size |
| `--warmup-iterations` | `10` | warmup iterations before measuring |
| `--seed` | `1234` | RNG seed |
| `--output` | `results/{aidotnet,pytorch}.json` | report path |
| `--threads` (PyTorch) | `0` (all cores) | pin CPU threads; match `AIDOTNET_BLAS_THREADS` |
| `--compile` (PyTorch) | off | wrap the model in `torch.compile` |
| `--compile-backend` (PyTorch) | `inductor` | `inductor`, `cudagraphs` (no triton), `aot_eager` |
| `--compile-mode` (PyTorch) | `default` | `reduce-overhead` (CUDA graphs), `max-autotune`, `max-autotune-no-cudagraphs` |
| `--optimizer-impl` (PyTorch) | `default` | AdamW implementation: `default`, `foreach`, `fused` |
| `--skip-inference` (PyTorch) | off | training rows only |

Inference is measured at batch sizes **1, 8, 32, 128** on both sides.

## Notes

- `results/` is git-ignored (machine-specific timings don't belong in version control),
  except the committed `results/scoreboard.{md,json}` (section 4).
- `mlp-fused` is an AiDotNet-only primitive variant (direct `MlpForward`); the
  PyTorch side maps it to the same `MLP` so a shared `--models` list won't error.
- This project is excluded from `dotnet test` (`IsTestProject=false`); it's a
  manual harness you run on demand.
