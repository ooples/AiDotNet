# Recurrent foundation measurements

Measured 2026-10-27 on Windows, Ryzen 9 3950X and Radeon RX 5500 XT. AiDotNet uses CPU/OpenCL; PyTorch 2.4.1 uses native `nn.LSTM` on CPU and an explicit tensor recurrence on DirectML. Native `nn.LSTM` on this DirectML version attempts an unsupported CPU fallback; it is **not** the GPU comparator. These are single-machine microbenchmarks, not end-to-end model-training or CUDA/cuDNN claims.

Baseline: AiDotNet `04e44e859446307322f06043a11023f9ebc822d8` (the #2378 merge), Tensors NuGet 0.143.0. Updated: AiDotNet production tree `e00ec8266c`, Tensors `3c52f2918393eb067479e0238cbf0b5e583803b1`. The results measure the two PRs together. Exact environment and raw median/min/max, .NET allocation counts, transfer counts, and checksums are in [results](results).

## Method

- Release/net10.0, one CPU thread (`AIDOTNET_BLAS_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `DOTNET_PROCESSOR_COUNT` = 1; PyTorch intra/inter-op = 1), `DOTNET_TieredCompilation=0`. Five warmups, 21 timed iterations, median reported. Processes run sequentially without concurrent builds/tests.
- LSTM shapes are `[batch, time, input, hidden]`; inputs are `sin(index * .01) * .1`, weights and the single bias are `.01`, initial states zero. PyTorch's second bias is zero and frozen. Each training measurement computes the sum of squared sequence outputs and gradients of inputs plus all trainable parameters; no optimizer update. Float32 except explicitly labeled float64.
- Projected gates use the reported cell shape; the gate's last dimension is four times larger. Before = equivalent composed sigmoid/tanh cell, after = `LstmCellFromGates`; the loss includes both hidden and cell outputs. This isolates the new primitive used by ConvLSTM/DNC rather than measuring their full projections/controllers.
- Checkpoints use eight 256×256 float32 tensors (2 MiB). Before = packed flatten-and-restore microbenchmark; after = reusable `TensorCheckpoint` capture/restore; PyTorch = preallocated `copy_` snapshots. This isolates state-transfer overhead rather than timing a whole model. The first result is explicitly downloaded at each GPU timing boundary.
- GPU timings include completed work and the same result download in both frameworks: sequence output for forward, input gradient for training, gate gradient for gates, first live tensor for checkpoints. Transfer diagnostics run separately without the timing-boundary download. They are not mixed into wall time. All updated measured paths report zero internal downloads.
- PyTorch warnings are errors so a DirectML CPU fallback fails the run. Tensor dtypes are explicit. `collect.py` verifies matching output/gradient checksums (1e-5 relative/absolute tolerance); the numerical regression suites provide the stronger elementwise/finite-difference checks.

| Device / workload | Shape | Before ms | After ms | PyTorch ms | Before/after |
|---|---|---:|---:|---:|---:|
| CPU LSTMLayer.forward | 4×8×16×32 | 0.0422 | 0.0300 | 0.1797 | 1.41× |
| CPU LSTMLayer.forward_backward | 4×8×16×32 | 0.1290 | 0.1278 | 0.5759 | 1.01× |
| CPU LSTMLayer.forward | 16×32×32×64 | 0.6439 | 0.6205 | 0.8163 | 1.04× |
| CPU LSTMLayer.forward_backward | 16×32×32×64 | 3.0865 | 2.9423 | 2.4463 | 1.05× |
| CPU LSTMLayer.float64.forward | 16×32×32×64 | 7.0773 | 3.6445 | 2.7808 | 1.94× |
| CPU LSTMLayer.float64.forward_backward | 16×32×32×64 | 26.0280 | 7.7681 | 11.0791 | 3.35× |
| CPU projected_gates.forward_backward | 8×64 | 0.1243 | 0.0392 | 0.2447 | 3.17× |
| CPU projected_gates.forward_backward | 2×16×16×32 | 0.7134 | 0.6947 | 0.5395 | 1.03× |
| CPU checkpoint.capture_restore | 8×256×256 | 1.3798 | 0.1120 | 0.1381 | 12.32× |
| GPU LSTMLayer.forward | 4×8×16×32 | 0.4880 | 0.4037 | 3.6093 | 1.21× |
| GPU LSTMLayer.forward_backward | 4×8×16×32 | 5.3838 | 5.2676 | 11.8951 | 1.02× |
| GPU LSTMLayer.forward | 16×32×32×64 | 1.7628 | 1.6388 | 14.7776 | 1.08× |
| GPU LSTMLayer.forward_backward | 16×32×32×64 | 95.1895 | 92.1742 | 52.3870 | 1.03× |
| GPU projected_gates.forward_backward | 8×64 | 1.0867 | 1.0353 | 1.2194 | 1.05× |
| GPU projected_gates.forward_backward | 2×16×16×32 | 1.8947 | 1.3279 | 1.2249 | 1.43× |
| GPU checkpoint.capture_restore | 8×256×256 | 4.2332 | 0.5347 | 0.6094 | 7.92× |

The new float64 and checkpoint paths improve substantially. Already-fused float32 training changes little; ratios near 1× should not be treated as significant. AiDotNet remains slower than PyTorch at the larger float32 training shape (both CPU and GPU), and at the spatial gate shape compared with PyTorch. These results do not establish that AiDotNet is faster than PyTorch across workloads.

## Reproduce

Build this project in the PR checkout against the companion Tensors checkout:

```powershell
dotnet build benchmarks/FoundationPerformance -c Release -p:UseLocalTensors=true -p:LocalTensorsRoot=C:/path/to/tensors -p:GeneratePackageOnBuild=false
```

Copy this harness to an isolated checkout of the baseline revision and build there without `UseLocalTensors`. Set the environment variables above before running each DLL with `cpu` or `gpu`. Run `pytorch.py` with Python 3.12 / torch 2.4.1 / torch-directml 0.2.5.dev240914, with and without `--gpu`. Save outputs as `foundation-{before,after,pytorch}-{cpu,gpu}.json`, then run `python collect.py <output-directory>`. The collector strips OpenCL initialization logging and validates checksums before producing the table.

