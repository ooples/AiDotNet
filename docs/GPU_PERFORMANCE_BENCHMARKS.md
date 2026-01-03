# GPU Performance Benchmarks - Phase B

## Overview

This document describes the performance benchmarking infrastructure for AiDotNet's GPU acceleration (Phase B implementation).

## Benchmark Suite

### Location
`/tests/AiDotNet.Tests/Benchmarks/GpuAccelerationBenchmarks.cs`

### Operations Tested

The benchmark suite validates GPU acceleration across all Phase B epics:

#### Epic 2: Matrix Operations
- **GEMM** (General Matrix Multiply) - `MatrixMultiply`
  - Small: 128×128 matrices
  - Large: 2048×2048 matrices
- **GEMV** (General Matrix-Vector) - `MatrixVectorMultiply`
  - Large: 2048×2048 matrix × 2048 vector

#### Epic 3: Tensor Operations
- **Conv2D** (2D Convolution)
  - Input: 4×64×56×56 (batch, channels, height, width)
  - Kernels: 128×64×3×3
- **MaxPool2D** (Max Pooling)
  - Input: 4×128×28×28
  - Pool size: 2×2, stride: 2

#### Epic 4: Integration
- **ConvolutionalLayer** forward pass (combines Conv2D + bias + activation)
- **PoolingLayer** forward pass (direct pooling operation)
- **Vector Operations** (optimizer updates)
  - Element-wise add, multiply on 1M element vectors

## Running Benchmarks

### Method 1: BenchmarkDotNet CLI

```bash
cd tests/AiDotNet.Tests
dotnet run -c Release --filter "*GpuAccelerationBenchmarks*"
```

### Method 2: Programmatic Execution

```csharp
using BenchmarkDotNet.Running;
using AiDotNet.Tests.Benchmarks;

var summary = BenchmarkRunner.Run<GpuAccelerationBenchmarks>();
Console.WriteLine(summary);
```

### Method 3: Specific Benchmark

```bash
dotnet run -c Release --filter "*GEMM_Large_GPU*"
```

## Expected Results

### Small Operations (Below Adaptive Thresholds)

| Operation | Size | CPU | GPU | Winner |
|-----------|------|-----|-----|--------|
| GEMM | 128×128 | ~0.5ms | ~2ms | **CPU** |
| Reason | | No overhead | GPU launch overhead | Adaptive routing |

**Key Insight**: Adaptive execution automatically routes small operations to CPU, avoiding GPU overhead.

### Large Operations (Above Adaptive Thresholds)

| Operation | Size | CPU Time | GPU Time | Speedup | Notes |
|-----------|------|----------|----------|---------|-------|
| **GEMM** | 2048×2048 | ~5000ms | ~10ms | **500x** | O(n³) benefits most from parallelism |
| **GEMV** | 2048×2048 | ~50ms | ~0.5ms | **100x** | O(n²) moderate parallelism |
| **Conv2D** | 4×64×56×56 | ~3000ms | ~15ms | **200x** | Most critical CNN operation |
| **MaxPool2D** | 4×128×28×28 | ~200ms | ~5ms | **40x** | Simpler operation, less speedup |
| **Vector Ops** | 1M elements | ~10ms | ~0.2ms | **50x** | Limited by memory bandwidth |

### Layer Forward Passes

| Layer | Operation | CPU | GPU | Speedup |
|-------|-----------|-----|-----|---------|
| ConvolutionalLayer | Forward | ~3000ms | ~15ms | **200x** |
| PoolingLayer | Forward | ~200ms | ~5ms | **40x** |

**Impact**: Training a CNN is 100-200x faster on GPU due to convolution dominance.

### Memory Usage

- **GPU Operations**: Minimal allocations due to memory pooling (Gen 0: <10 collections)
- **CPU Operations**: More frequent allocations (Gen 0: 100+ collections)

## Comparison to PyTorch/TensorFlow

### Methodology

Direct benchmarking against PyTorch/TensorFlow requires Python interop. However, we can estimate comparative performance:

#### DirectGpu vs cuBLAS/cuDNN
Run the DirectGpu benchmarks to compare against vendor libraries on your hardware.
#### Key Differences

**AiDotNet Advantages**:
- ✅ Pure .NET implementation (no Python interop)
- ✅ Adaptive CPU/GPU execution (automatic optimization)
- ✅ Memory pooling (reduces allocation overhead)
- ✅ Type-safe at compile time

**PyTorch/TensorFlow Advantages**:
- ✅ Highly optimized cuDNN implementations (years of optimization)
- ✅ Larger ecosystem and community
- ✅ More mature profiling tools

**Expected Performance**: AiDotNet should achieve **30-80% of PyTorch/TensorFlow performance** for equivalent operations, which is excellent for a pure .NET implementation.

## Performance Optimization Tips

### 1. Use Appropriate Thresholds

```csharp
// For high-end GPU (RTX 4090, A100)
var engine = new GpuEngine(AdaptiveThresholds.HighEndGpu);

// For integrated graphics
var engine = new GpuEngine(AdaptiveThresholds.LowEndGpu);

// For testing (force all operations to GPU)
var engine = new GpuEngine(AdaptiveThresholds.AlwaysGpu);
```

### 2. Batch Operations

```csharp
// BAD: Sequential single operations
for (int i = 0; i < 100; i++)
{
    var result = engine.MatrixMultiply(a, b);
}

// GOOD: Batched operations
var results = engine.BatchMatMul(batchedA, batchedB);
```

### 3. Reuse Engines

```csharp
// BAD: Creating new engine per operation
var result1 = new GpuEngine().MatrixMultiply(a, b);
var result2 = new GpuEngine().MatrixMultiply(c, d); // Expensive!

// GOOD: Reuse single engine instance
var engine = new GpuEngine();
var result1 = engine.MatrixMultiply(a, b);
var result2 = engine.MatrixMultiply(c, d);
```

## Interpreting Results

### Speedup Calculation

```
Speedup = CPU Time / GPU Time
```

### When GPU is Slower

If GPU shows slower performance:
1. **Check operation size**: Below adaptive threshold?
2. **Check memory transfers**: Are you transferring data in a loop?
3. **Check GPU health**: Is GPU overheating or throttling?
4. **Check GPU availability**: Is CUDA/OpenCL properly installed?

### Memory Diagnostics

BenchmarkDotNet automatically reports:
- **Gen 0/1/2 collections**: Lower is better
- **Allocated bytes**: Lower is better
- **Memory pools**: Check that GPU engine reuses buffers

## Continuous Benchmarking

### Regression Detection

Run benchmarks before/after changes:

```bash
# Baseline
dotnet run -c Release --filter "*GpuAccelerationBenchmarks*" > baseline.txt

# After changes
dotnet run -c Release --filter "*GpuAccelerationBenchmarks*" > current.txt

# Compare
diff baseline.txt current.txt
```

### CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Run GPU Benchmarks
  run: |
    dotnet run -c Release --project tests/AiDotNet.Tests --filter "*GpuAccelerationBenchmarks*"
```

## Troubleshooting

### GPU Not Available

If GPU benchmarks return `null`:
1. Verify CUDA/OpenCL installation
2. Check GPU drivers are up to date
3. Verify `GpuEngine` initialization succeeds
4. Check DirectGpu backend availability (CUDA/OpenCL) for your GPU

### Unexpected Performance

If results differ from expected:
1. **Ensure Release build**: Debug builds are 10-100x slower
2. **Close background applications**: GPU might be busy
3. **Check CPU/GPU temperature**: Thermal throttling affects performance
4. **Verify adaptive thresholds**: Might be routing incorrectly

## References

- [BenchmarkDotNet Documentation](https://benchmarkdotnet.org/)
- [OpenCL Documentation](https://www.khronos.org/opencl/)
- [CUDA Documentation](https://developer.nvidia.com/cuda-toolkit)
- Phase B Implementation: See Epic 1-4 user stories in issue #496

---

**Last Updated**: 2025-01-17
**Phase**: B - GPU Production Implementation
**Status**: US-GPU-017 Complete

