# Inference Optimization Benchmarks

This directory contains benchmark tests for the inference optimization components.

## Running Benchmarks

### All Benchmarks

```bash
cd AiDotNetBenchmarkTests
dotnet run -c Release --project InferenceOptimization
```

### Individual Benchmarks

```bash
# GEMM benchmark
dotnet run -c Release --filter "*GemmBenchmark*"

# SIMD benchmark
dotnet run -c Release --filter "*SimdBenchmark*"

# Attention benchmark
dotnet run -c Release --filter "*AttentionBenchmark*"
```

## Benchmark Descriptions

### GemmBenchmark
Tests matrix multiplication performance:
- **NaiveGemm**: Baseline triple-nested loop implementation
- **OptimizedGemm**: Cache-blocked SIMD-optimized implementation
- **OptimizedGemmTranspose**: Optimized implementation for transposed matrices

**Matrix sizes tested**: 64x64, 128x128, 256x256, 512x512, 1024x1024

**Expected results**:
- 2-3x speedup on AVX2 systems
- 2.5x speedup on ARM NEON systems
- Better speedup for larger matrices

### SimdBenchmark
Tests SIMD-optimized vector operations:
- **Vector Addition**: Element-wise addition
- **Vector Multiplication**: Element-wise multiplication
- **Dot Product**: Inner product with FMA optimization
- **ReLU**: Activation function
- **Sum**: Reduction operation

**Array sizes tested**: 1K, 10K, 100K, 1M elements

**Expected results**:
- 4-8x speedup on AVX2 systems (processes 8 floats at once)
- 2-4x speedup on SSE systems (processes 4 floats at once)
- 2-4x speedup on NEON systems (processes 4 floats at once)

### AttentionBenchmark
Tests fused attention kernel performance:
- **NaiveAttention**: Standard three-step implementation (QK^T, softmax, V)
- **OptimizedAttention**: Fused implementation with SIMD
- **MultiHeadAttention**: Multi-head variant (8 heads)

**Parameters tested**:
- Sequence lengths: 64, 128, 256
- Feature dimensions: 32, 64

**Expected results**:
- 2-2.5x speedup from memory traffic reduction
- Better performance for longer sequences

## Interpreting Results

BenchmarkDotNet produces detailed reports including:

### Timing Metrics
- **Mean**: Average execution time
- **Error**: Half of 99.9% confidence interval
- **StdDev**: Standard deviation
- **Median**: 50th percentile

### Memory Metrics
- **Gen0/Gen1/Gen2**: Garbage collection frequency
- **Allocated**: Total memory allocated

### Speedup Calculation
```
Speedup = Baseline Time / Optimized Time
```

Example output:
```
|                Method | MatrixSize |      Mean |    Error |   StdDev | Ratio |
|---------------------- |----------- |----------:|---------:|---------:|------:|
|             NaiveGemm |        256 | 27.45 ms  | 0.421 ms | 0.394 ms |  1.00 |
|         OptimizedGemm |        256 |  9.12 ms  | 0.142 ms | 0.133 ms |  0.33 |

Speedup = 27.45 / 9.12 = 3.01x
```

## Performance Targets

### GEMM
- ✅ Target: 2-5x speedup
- Platform dependent:
  - AVX2: 2.5-3x
  - AVX-512: 3-4x
  - NEON: 2-2.5x

### SIMD Operations
- ✅ Target: 2-8x speedup
- Depends on:
  - Instruction set (AVX2 > SSE > scalar)
  - Array size (larger = better amortization)
  - Operation type (simple ops get higher speedup)

### Attention
- ✅ Target: 2-3x speedup
- Benefits:
  - Reduced memory traffic
  - Fused operations
  - Cache efficiency

## Platform-Specific Results

Your benchmark results will vary based on:

1. **CPU Architecture**
   - Intel/AMD x86_64: Best with AVX2/AVX-512
   - ARM: Good with NEON
   - Older CPUs: Falls back to SSE or scalar

2. **Cache Hierarchy**
   - Larger caches = Better performance for blocked algorithms
   - L1/L2/L3 sizes affect optimal tile sizes

3. **Memory Bandwidth**
   - DDR4/DDR5 speed affects large matrix operations
   - Memory channels matter for parallel operations

4. **Thermal Throttling**
   - Sustained benchmarks may hit thermal limits
   - Use adequate cooling

## Comparing to Reference Implementations

To compare against Intel MKL or OpenBLAS:

```csharp
// Add reference implementation
[Benchmark]
public Tensor<float> MKL_SGEMM()
{
    // Call Intel MKL cblas_sgemm
    // ...
}
```

## Contributing

To add new benchmarks:

1. Create a new class inheriting benchmark attributes
2. Add `[Params]` for different sizes/configurations
3. Implement baseline (naive) version
4. Implement optimized version
5. Add `[GlobalSetup]` for initialization
6. Mark baseline with `[Benchmark(Baseline = true)]`
7. Add memory diagnostics: `[MemoryDiagnoser]`

## CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Run Benchmarks
  run: |
    cd AiDotNetBenchmarkTests
    dotnet run -c Release --filter "*InferenceOptimization*"

- name: Upload Results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: BenchmarkDotNet.Artifacts/results/
```

## Troubleshooting

### Benchmark takes too long
- Reduce parameter ranges
- Use `[SimpleJob]` instead of full job
- Reduce warmup/iteration counts

### Inconsistent results
- Close other applications
- Disable CPU frequency scaling
- Run multiple iterations
- Check for thermal throttling

### Out of memory
- Reduce test sizes
- Add `[MemoryDiagnoser]` to track allocations
- Consider streaming benchmarks for large data

## References

- [BenchmarkDotNet Documentation](https://benchmarkdotnet.org/)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [ARM NEON Documentation](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)
