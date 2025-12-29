# Benchmarks

This repository includes GPU and CPU benchmark suites for the Phase B GPU acceleration work.

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
- GPU: ILGPU reported gfx902 during AiDotNet runs
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

## Expected trends

- Small operations (below adaptive thresholds) are usually faster on CPU due to GPU launch overhead.
- Large matrix and convolution operations should show significant GPU speedups.

## Comparison notes vs PyTorch/TensorFlow

AiDotNet uses ILGPU kernels rather than vendor-tuned cuBLAS/cuDNN. Expected relative performance:
- GEMM: roughly 0.5x to 0.8x of cuBLAS on comparable hardware
- Conv2D: roughly 0.3x to 0.6x of cuDNN on comparable hardware
- Pooling: roughly 0.5x to 0.9x of cuDNN on comparable hardware

See `docs/GPU_PERFORMANCE_BENCHMARKS.md` for detailed benchmark coverage and methodology.
