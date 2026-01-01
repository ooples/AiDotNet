# Benchmarks

This repository includes GPU benchmark suites for the Phase B GPU acceleration work.

## GPU acceleration benchmarks

Location:
- `tests/AiDotNet.Tests/Benchmarks/GpuAccelerationBenchmarks.cs`

Run:
```bash
dotnet run -c Release --project tests/AiDotNet.Tests --filter "*GpuAccelerationBenchmarks*"
```

## TensorFlow.NET comparison benchmarks

Location:
- `AiDotNetBenchmarkTests/TensorFlowComparisonBenchmarks.cs`

Run:
```bash
cd AiDotNetBenchmarkTests
dotnet run -c Release -f net8.0 -- --filter *TensorFlowComparison*
```

Latest run (2025-12-28, short-run config: IterationCount=5, WarmupCount=3, LaunchCount=1):
- Host: AMD Ryzen 7 4800H, Windows 11, .NET 8.0.22
- GPU: legacy ILGPU reported gfx902 during AiDotNet runs (DirectGpu now replaces this path)
- TensorFlow.NET logs indicated oneDNN CPU backend active

Mean time (ms):

| Operation | Size | AiDotNet | TensorFlow.NET |
|---|---:|---:|---:|
| ReLU | 1,000,000 elems | 19.596 | 1.116 |
| Sigmoid | 1,000,000 elems | 20.100 | 1.518 |
| ReduceSum | 1,000,000 elems | 12.078 | 0.354 |
| ReduceMean | 1,000,000 elems | 8.441 | 0.224 |
| Conv2D | [1,16,64,64] x [32,16,3,3] | 2.782 | 0.572 |
| MatMul | 256x256 | 2.121 | 0.385 |
| MatMul | 512x512 | 40.753 | 1.802 |
| Add | 100,000 elems | 3.088 | 0.261 |
| Multiply | 100,000 elems | 1.628 | 0.181 |
| Add | 1,000,000 elems | 7.953 | 1.445 |
| Multiply | 1,000,000 elems | 8.052 | 1.398 |
## TorchSharp comparison benchmarks

Location:
- `AiDotNetBenchmarkTests/TorchSharpComparisonBenchmarks.cs`

Run:
```bash
cd AiDotNetBenchmarkTests
dotnet run -c Release -f net8.0 -- --filter *TorchSharpComparison*
```

Latest run (2025-12-28, short-run config: IterationCount=5, WarmupCount=3, LaunchCount=1):
- Host: AMD Ryzen 7 4800H, Windows 11, .NET 8.0.22
- GPU: legacy ILGPU reported gfx902 during AiDotNet runs (DirectGpu now replaces this path)
- TorchSharp device: CPU (torch.cuda.is_available() returned false with TorchSharp-cuda-windows)

Mean time (ms):

| Operation | Size | AiDotNet | TorchSharp |
|---|---:|---:|---:|
| ReLU | 1,000,000 elems | 21.174 | 0.090 |
| Sigmoid | 1,000,000 elems | 22.408 | 0.368 |
| ReduceSum | 1,000,000 elems | 11.190 | 0.052 |
| ReduceMean | 1,000,000 elems | 6.272 | 0.058 |
| Conv2D | [1,16,64,64] x [32,16,3,3] | 2.662 | 0.293 |
| MatMul | 256x256 | 2.429 | 0.127 |
| MatMul | 512x512 | 10.166 | 1.037 |
| Add | 100,000 elems | 1.574 | 0.011 |
| Multiply | 100,000 elems | 1.758 | 0.010 |
| Add | 1,000,000 elems | 8.260 | 0.661 |
| Multiply | 1,000,000 elems | 8.142 | 0.655 |

Notes:
- 2025-12-28: Re-ran after removing duplicate kernel launches and explicit accelerator synchronizations; results were within noise of the prior run (transfer overhead still dominates).


## Expected trends

- Small operations (below adaptive thresholds) are usually faster on CPU due to GPU launch overhead.
- Large matrix and convolution operations should show significant GPU speedups.

## Comparison notes vs PyTorch/TensorFlow

AiDotNet now uses DirectGpu backends; the legacy ILGPU kernels had the following relative performance expectations:
- GEMM: roughly 0.5x to 0.8x of cuBLAS on comparable hardware
- Conv2D: roughly 0.3x to 0.6x of cuDNN on comparable hardware
- Pooling: roughly 0.5x to 0.9x of cuDNN on comparable hardware

See `docs/GPU_PERFORMANCE_BENCHMARKS.md` for detailed benchmark coverage and methodology.
