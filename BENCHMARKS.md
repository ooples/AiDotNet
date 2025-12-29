# Benchmarks

This repository includes GPU and CPU benchmark suites for the Phase B GPU acceleration work.

## GPU acceleration benchmarks

Location:
- `tests/AiDotNet.Tests/Benchmarks/GpuAccelerationBenchmarks.cs`

Run:
```bash
dotnet run -c Release --project tests/AiDotNet.Tests --filter "*GpuAccelerationBenchmarks*"
```

## Expected trends

- Small operations (below adaptive thresholds) are usually faster on CPU due to GPU launch overhead.
- Large matrix and convolution operations should show significant GPU speedups.

## Comparison notes vs PyTorch/TensorFlow

AiDotNet uses ILGPU kernels rather than vendor-tuned cuBLAS/cuDNN. Expected relative performance:
- GEMM: roughly 0.5x to 0.8x of cuBLAS on comparable hardware
- Conv2D: roughly 0.3x to 0.6x of cuDNN on comparable hardware
- Pooling: roughly 0.5x to 0.9x of cuDNN on comparable hardware

See `docs/GPU_PERFORMANCE_BENCHMARKS.md` for detailed benchmark coverage and methodology.
