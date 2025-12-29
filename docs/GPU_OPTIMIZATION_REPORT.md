# GPU Performance Optimization Report (Issue #496)

## Scope
- Date: 2025-12-28
- Benchmarks: `AiDotNetBenchmarkTests/TensorFlowComparisonBenchmarks.cs`, `AiDotNetBenchmarkTests/TorchSharpComparisonBenchmarks.cs`
- Environment: AMD Ryzen 7 4800H, Windows 11, .NET 8.0.22
- GPU: ILGPU reported gfx902 during AiDotNet runs
- TorchSharp device: CPU (torch.cuda.is_available() returned false with TorchSharp-cuda-windows)
- Results: `AiDotNetBenchmarkTests/BenchmarkDotNet.Artifacts/results/AiDotNetBenchmarkTests.TorchSharpComparisonBenchmarks-report-github.md`

## Summary of findings
- AiDotNet GPU is slower than TorchSharp CPU and TensorFlow.NET CPU across all tested micro-ops.
- Largest gaps are on elementwise and reduction ops (10x-235x slower vs TorchSharp CPU). MatMul and Conv2D are still 9x-20x slower.
- Forcing GPU execution on small/medium workloads amplifies launch and transfer overhead, masking any kernel-level speedups.
- GC allocations for AiDotNet ops are high (BDN shows Gen0/Gen1 pressure), indicating significant managed allocation overhead in hot paths.

## Changes applied in this branch
- Removed duplicate kernel launches that were executing once before and once inside the `_gpuLock` in `GpuEngine`.
- Removed explicit `_accelerator.Synchronize()` calls; rely on blocking CPU copies to avoid redundant device-wide barriers.

## Benchmark highlights (TorchSharp CPU vs AiDotNet GPU, mean time)
- ReLU: 21.174 ms vs 0.090 ms (235x slower)
- Sigmoid: 22.408 ms vs 0.368 ms (61x slower)
- ReduceSum: 11.190 ms vs 0.052 ms (215x slower)
- ReduceMean: 6.272 ms vs 0.058 ms (108x slower)
- MatMul 256: 2.429 ms vs 0.127 ms (19x slower)
- MatMul 512: 10.166 ms vs 1.037 ms (9.8x slower)
- Conv2D: 2.662 ms vs 0.293 ms (9.1x slower)
- Add 1,000,000: 8.260 ms vs 0.661 ms (12.5x slower)
- Multiply 1,000,000: 8.142 ms vs 0.655 ms (12.4x slower)

## Post-change benchmark observation
- After removing duplicate kernel launches and explicit synchronizations, TorchSharp comparison results were statistically unchanged. Host-device transfers still dominate.

## Likely bottlenecks (code evidence)
- Per-op host/device transfers: `GpuEngine` allocates GPU buffers and copies for each call (`Allocate1D`, `CopyFromCPU`, `CopyToCPU`) in `src/AiDotNet.Tensors/Engines/GpuEngine.cs`.
- Synchronization and serialization: `_gpuLock` serializes kernel launches; explicit synchronizations were removed but transfers still block.
- Kernel launch overhead: even with precompilation, many operations still dispatch separate kernels and cannot amortize the cost for smaller ops.
- Naive kernel implementations: Conv2D and MatMul are not using tiling/shared memory or vendor-tuned kernels, leaving bandwidth and occupancy on the table.
- Managed allocations: new `Tensor<T>` objects and arrays are created per operation, driving GC activity.

## Recommended optimizations (prioritized)

### P0 - Immediate (biggest ROI)
1. Reduce host/device roundtrips: introduce GPU-resident tensor paths to keep data on device across chained ops, only copying to CPU at the end.
2. Expand memory pooling usage: ensure pooled buffers are used in all hot paths (vector, matrix, tensor ops) and reuse output buffers where safe.
3. Avoid per-op synchronization: use async streams and defer `Synchronize` until data is needed on CPU; shrink or remove `_gpuLock` contention.

### P1 - Near term
4. Finish kernel caching coverage: ensure all hot paths use cached kernels and remove `LoadAutoGroupedStreamKernel` calls from runtime paths.
5. Kernel fusion for common sequences (bias + activation, add + activation, normalization + activation) to cut launch count.
6. Optimize GEMM/Conv2D kernels with tiling/shared memory and vectorized memory access patterns.

### P2 - Mid term
7. Optional vendor library interop: route GEMM/Conv2D to cuBLAS/cuDNN or rocBLAS/MIOpen when available.
8. Re-tune adaptive thresholds using benchmark-driven crossover points once the above changes land.
9. Add GPU profiling instrumentation (kernel time vs copy time) to validate improvements and catch regressions early.

## Validation plan
- Add microbenchmarks for copy-only vs kernel-only cost to quantify transfer overhead.
- Re-run TorchSharp/TensorFlow comparisons after each major optimization bucket.
- Add size-sweep benchmarks to re-derive thresholds for CPU vs GPU routing.
