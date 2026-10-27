# Recurrent foundation measurements

Measured 2026-10-27 on Windows, Ryzen 9 3950X and Radeon RX 5500 XT. AiDotNet uses CPU/OpenCL; PyTorch 2.4.1 uses native `nn.LSTM` on CPU and an explicit tensor recurrence on DirectML. Native `nn.LSTM` on this DirectML version attempts an unsupported CPU fallback; it is **not** the GPU comparator. These are single-machine microbenchmarks, not end-to-end model-training or CUDA/cuDNN claims.

Baseline: AiDotNet `04e44e859446307322f06043a11023f9ebc822d8` (the #2378 merge), Tensors NuGet 0.143.0. Updated: AiDotNet production tree `b714fe19696a1a7b75a4cb7840f344cca4217e1f`, Tensors `b67e720f86c0cc39d7d7f847bbe961a3c29807d6`. The results measure the two PRs together. Exact environment and raw samples and median/min/max, .NET allocation counts, transfer counts, and checksums are in [results](results).

## Method

- Release/net10.0, one CPU thread (`AIDOTNET_BLAS_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `DOTNET_PROCESSOR_COUNT` = 1; PyTorch intra/inter-op = 1), `DOTNET_TieredCompilation=0`. Twenty warmups, 101 timed iterations, median reported. Processes run sequentially without concurrent builds/tests.
- LSTM shapes are `[batch, time, input, hidden]`; inputs are `sin(index * .01) * .1`, weights and the single bias are `.01`, initial states zero. PyTorch's second bias is zero and frozen. Each training measurement computes the sum of squared sequence outputs and gradients of inputs plus all trainable parameters; no optimizer update. Float32 except explicitly labeled float64.
- Projected gates use the reported cell shape; the gate's last dimension is four times larger. Before = equivalent composed sigmoid/tanh cell, after = `LstmCellFromGates`; the loss includes both hidden and cell outputs. This isolates the new primitive used by ConvLSTM/DNC rather than measuring their full projections/controllers.
- Checkpoints use eight 256Ã—256 float32 tensors (2 MiB). Before = packed flatten-and-restore microbenchmark; after = reusable `TensorCheckpoint` capture/restore; PyTorch = preallocated `copy_` snapshots. This isolates state-transfer overhead rather than timing a whole model. The first result is explicitly downloaded at each GPU timing boundary.
- GPU timings include completed work and the same result download in both frameworks: sequence output for forward, input gradient for training, gate gradient for gates, first live tensor for checkpoints. Transfer diagnostics run separately without the timing-boundary download. They are not mixed into wall time. All updated measured paths report zero internal downloads.
- PyTorch warnings are errors so a DirectML CPU fallback fails the run. Tensor dtypes are explicit. `collect.py` verifies matching output/gradient checksums (1e-5 relative/absolute tolerance); the numerical regression suites provide the stronger elementwise/finite-difference checks.

| Device / workload | Shape | Before ms | After ms | PyTorch ms | Before/after | PyTorch/after |
|---|---|---:|---:|---:|---:|---:|
| CPU LSTMLayer.forward | 1×1×16×32 | 0.0139 | 0.0048 | 0.1465 | 2.90× | 30.52× |
| CPU LSTMLayer.forward_backward | 1×1×16×32 | 0.0622 | 0.0593 | 0.4925 | 1.05× | 8.31× |
| CPU LSTMLayer.forward | 4×8×16×32 | 0.0345 | 0.0262 | 0.1780 | 1.32× | 6.79× |
| CPU LSTMLayer.forward_backward | 4×8×16×32 | 0.1248 | 0.1190 | 0.5749 | 1.05× | 4.83× |
| CPU LSTMLayer.forward | 2×7×13×37 | 0.0477 | 0.0388 | 0.1804 | 1.23× | 4.65× |
| CPU LSTMLayer.forward_backward | 2×7×13×37 | 0.1296 | 0.1307 | 0.5935 | 0.99× | 4.54× |
| CPU LSTMLayer.forward | 16×32×32×64 | 0.6424 | 0.6107 | 0.8205 | 1.05× | 1.34× |
| CPU LSTMLayer.forward_backward | 16×32×32×64 | 2.8898 | 1.6904 | 2.3957 | 1.71× | 1.42× |
| CPU LSTMLayer.forward | 8×16×64×128 | 0.7429 | 0.6266 | 0.6943 | 1.19× | 1.11× |
| CPU LSTMLayer.forward_backward | 8×16×64×128 | 1.8399 | 1.7750 | 1.9136 | 1.04× | 1.08× |
| CPU LSTMLayer.float64.forward | 16×32×32×64 | 10.1137 | 2.1232 | 2.7997 | 4.76× | 1.32× |
| CPU LSTMLayer.float64.forward_backward | 16×32×32×64 | 32.0420 | 5.3498 | 10.9305 | 5.99× | 2.04× |
| CPU projected_gates.forward_backward | 8×64 | 0.1691 | 0.0340 | 0.2391 | 4.97× | 7.03× |
| CPU projected_gates.forward_backward | 3×37 | 0.1009 | 0.0471 | 0.2346 | 2.14× | 4.98× |
| CPU projected_gates.forward_backward | 2×16×16×32 | 0.7614 | 0.5034 | 0.5390 | 1.51× | 1.07× |
| CPU projected_gates.forward_backward | 4×8×8×64 | 0.6885 | 0.3433 | 0.5027 | 2.01× | 1.46× |
| CPU checkpoint.capture_restore | 8×256×256 | 1.2645 | 0.1258 | 0.1492 | 10.05× | 1.19× |
| GPU LSTMLayer.forward | 1×1×16×32 | 0.3695 | 0.2812 | 0.5285 | 1.31× | 1.88× |
| GPU LSTMLayer.forward_backward | 1×1×16×32 | 1.2577 | 0.7494 | 1.5154 | 1.68× | 2.02× |
| GPU LSTMLayer.forward | 4×8×16×32 | 0.4884 | 0.3714 | 3.3445 | 1.32× | 9.01× |
| GPU LSTMLayer.forward_backward | 4×8×16×32 | 5.1483 | 1.1509 | 12.0767 | 4.47× | 10.49× |
| GPU LSTMLayer.forward | 2×7×13×37 | 0.5589 | 0.3671 | 3.3936 | 1.52× | 9.24× |
| GPU LSTMLayer.forward_backward | 2×7×13×37 | 5.3901 | 1.0303 | 12.6397 | 5.23× | 12.27× |
| GPU LSTMLayer.forward | 16×32×32×64 | 1.7852 | 1.5786 | 13.8542 | 1.13× | 8.78× |
| GPU LSTMLayer.forward_backward | 16×32×32×64 | 83.1160 | 4.4647 | 55.0027 | 18.62× | 12.32× |
| GPU LSTMLayer.forward | 8×16×64×128 | 3.0493 | 2.8731 | 8.2998 | 1.06× | 2.89× |
| GPU LSTMLayer.forward_backward | 8×16×64×128 | 198.5795 | 5.5686 | 27.8366 | 35.66× | 5.00× |
| GPU projected_gates.forward_backward | 8×64 | 1.1837 | 0.6971 | 1.3016 | 1.70× | 1.87× |
| GPU projected_gates.forward_backward | 3×37 | 1.0852 | 0.7207 | 1.4265 | 1.51× | 1.98× |
| GPU projected_gates.forward_backward | 2×16×16×32 | 1.8940 | 1.0205 | 1.3540 | 1.86× | 1.33× |
| GPU projected_gates.forward_backward | 4×8×8×64 | 1.4252 | 1.0214 | 1.5801 | 1.40× | 1.55× |
| GPU checkpoint.capture_restore | 8×256×256 | 5.0127 | 0.4228 | 0.4694 | 11.86× | 1.11× |

The final medians favor AiDotNet in all 17 CPU and 15 GPU cases in this matrix. The smallest PyTorch margin is 1.07x; small margins and before/after ratios near 1x remain sensitive to machine and run variability. One odd-width CPU training case is approximately 1% slower than baseline. These results establish the measured cases on this machine, not every operation, shape, thread count, GPU backend, or end-to-end model. GPU results compare OpenCL against DirectML explicit recurrence, not CUDA/cuDNN.

## Reproduce

Build this project in the PR checkout against the companion Tensors checkout:

```powershell
dotnet build benchmarks/FoundationPerformance -c Release -p:UseLocalTensors=true -p:LocalTensorsRoot=C:/path/to/tensors -p:GeneratePackageOnBuild=false
```

Copy this harness to an isolated checkout of the baseline revision and build there without `UseLocalTensors`. Set the environment variables above before running each DLL with `cpu` or `gpu`. Run `pytorch.py` with Python 3.12 / torch 2.4.1 / torch-directml 0.2.5.dev240914, with and without `--gpu`. Save outputs as `foundation-{before,after,pytorch}-{cpu,gpu}.json`, then run `python collect.py <output-directory>`. The collector strips OpenCL initialization logging and validates checksums before producing the table.


## Profiling and fixes

ETW collection required elevation and did not complete. These findings come from `dotnet-trace collect --profile dotnet-sampled-thread-time`, exported with PerfView `userCommand NetperfToSpeedScope`; they are sampled managed/native stacks, not GPU kernel timestamps. [Profile summaries](results/profile-summary.json) restrict attribution to the measured workload.

- Float64 forward spent 23.76% of sampled time clearing buffers, 23.21% in the scalar recurrence core, and 13.26% in streaming GEMM. Use overwrite small recurrent GEMMs and full-precision vector exponential activation loops.
- Float32 CPU training spent 33.51% materializing GEMM transposes. Route full-gradient products through the existing direct-transpose GEMM implementation.
- GPU training spent 98.04% waiting at the result boundary. Inspection found the old OpenCL backward assigned the entire sequential BPTT workload to one work-item per batch element. Convert its interleaved gate cache on device and use the shared parallel backward kernels.
- Spatial projected-gate CPU backward spent 28.03% in scalar backward equations. Vectorize those equations and the cell update. Remove redundant GPU gate synchronizations while retaining queued buffer ownership.
- Expanded float32 inference profiling exposed clearing and scalar weight transpose overhead. Use the existing vectorized weight transpose and a serial write-first recurrent GEMM; poisoned-output tests check full and tail writes.

To reproduce a sampled profile, set `FOUNDATION_PROFILE_CASE` to a workload and shape (for example `LSTMLayer.forward:8x16x64x128`) and `FOUNDATION_PROFILE_SECONDS=20`, then run the harness under dotnet-trace. Unset the profile selector for timing. Keep profiling separate from benchmarks.

Execution order for the published measurements was fixed: before CPU, before GPU, after CPU, after GPU, PyTorch CPU, PyTorch GPU. It was not randomized or interleaved, so time-dependent thermal or clock drift may affect close margins. Future comparisons should alternate or randomize implementations and record the actual order.

When running the copied harness against the baseline without TensorCheckpoint, also pass `--legacy-checkpoint-baseline`. The updated checkout requires both Capture and Restore and rejects that baseline flag; a missing API fails before timing starts.
