# GPU Stress Testing and Memory Leak Detection - Phase B

## Overview

This document describes the stress testing and memory leak detection infrastructure for AiDotNet's GPU acceleration (Phase B: US-GPU-018).

## Test Suites

### Location
- **Stress Tests**: `/tests/AiDotNet.Tests/StressTests/GpuStressTests.cs`
- **Memory Leak Tests**: `/tests/AiDotNet.Tests/StressTests/MemoryLeakTests.cs`

## Stress Test Categories

### 1. Matrix Operation Stress Tests

**Purpose**: Validate GPU stability over extended matrix operations

#### Tests:
- **10K Matrix Multiplications** (`MatrixMultiply_LongRun_10KIterations_NoMemoryLeak`)
  - Runs 10,000 consecutive 256×256 matrix multiplications
  - Monitors memory growth (should be < 10MB)
  - Validates performance consistency (avg < 1ms per operation)
  - **Pass Criteria**: No memory leaks, stable performance

- **Concurrent Matrix Operations** (`MatrixMultiply_Concurrent_8Threads_NoRaceConditions`)
  - Executes 8 concurrent threads performing matrix operations
  - Each thread runs 100 iterations
  - Tests thread safety and race condition prevention
  - **Pass Criteria**: No exceptions, all threads complete successfully

### 2. Tensor Operation Stress Tests

**Purpose**: Validate CNN operations under sustained load

#### Tests:
- **1K Conv2D Operations** (`Conv2D_LongRun_1KIterations_StablePerformance`)
  - Runs 1,000 convolution operations (4×32×28×28 → 4×64×28×28)
  - Monitors performance drift between first and last quartile
  - Memory growth should be < 20MB
  - **Pass Criteria**: Performance drift < 20%, no memory leaks

- **High-Frequency Pooling** (`Pooling_HighFrequency_1KIterations_NoLeaks`)
  - Alternates between MaxPool2D and AvgPool2D for 1,000 iterations
  - Large batch size (8×64×56×56) for stress testing
  - Memory growth should be < 15MB
  - **Pass Criteria**: No memory leaks, stable operation

### 3. Neural Network Layer Stress Tests

**Purpose**: Validate layer operations in realistic training scenarios

#### Tests:
- **ConvolutionalLayer 1K Forward Passes** (`ConvolutionalLayer_LongRun_1KForwardPasses_Stable`)
  - Runs 1,000 forward passes through convolutional layer
  - Tests layer state management and memory cleanup
  - **Pass Criteria**: Memory growth < 10MB

- **Full CNN Pipeline** (`FullCNNPipeline_100Iterations_NoMemoryLeaks`)
  - Simulates realistic CNN: Conv → ReLU → Pool → Conv → ReLU → Pool
  - Runs 100 complete pipeline iterations
  - Memory growth should be < 30MB
  - Average pipeline time should be < 100ms
  - **Pass Criteria**: No memory leaks, acceptable performance

### 4. GPU Memory Pool Stress Tests

**Purpose**: Validate memory pool behavior under varied allocation patterns

#### Tests:
- **Variable Size Allocations** (`MemoryPool_VariableSizeAllocations_ReuseBuffers`)
  - Allocates matrices of varying sizes (64, 128, 256, 512, 256, 128, 64)
  - Runs 100 cycles through size variations
  - Tests buffer reuse in memory pool
  - **Pass Criteria**: Memory growth < 5MB (indicates effective pooling)

- **Rapid Allocation/Deallocation** (`MemoryPool_RapidAllocDealloc_1KCycles_Stable`)
  - Performs 1,000 rapid allocation/deallocation cycles
  - Tests GC integration and buffer cleanup
  - **Pass Criteria**: Memory growth < 5MB

### 5. GPU Recovery Stress Tests

**Purpose**: Validate graceful error handling and recovery

#### Tests:
- **Invalid Operations** (`GPU_InvalidOperations_GracefulErrorHandling`)
  - Tests error handling for incompatible matrix sizes
  - Validates engine continues operating after errors
  - **Pass Criteria**: Proper exception handling, engine remains functional

- **Multiple Error Recovery** (`GPU_MultipleErrors_ContinuesOperating`)
  - Causes 10 consecutive errors
  - Tests operation between each error
  - **Pass Criteria**: Engine remains stable after multiple errors

## Memory Leak Detection

### Memory Leak Criteria

A memory leak is suspected if:
1. **Linear Growth**: Memory grows linearly with iterations (correlation > 0.8)
2. **Excessive Growth**: Total growth > 15MB over 5,000 iterations
3. **No Plateau**: Memory does not stabilize after warmup period
4. **High GC Pressure**: Excessive Gen 2 collections (> 5 per 5K iterations)

### Memory Analysis Tests

#### 1. Growth Pattern Analysis (`MatrixOperations_5KIterations_LinearGrowthCheck`)
- Runs 5,000 matrix operations
- Samples memory every 500 iterations
- Uses correlation analysis to detect linear growth
- **Pass Criteria**: No linear growth pattern, total growth < 15MB

#### 2. Plateau Detection (`TensorOperations_5KIterations_PlateauCheck`)
- Runs 5,000 Conv2D operations
- Validates memory stabilizes after initial allocations
- **Pass Criteria**: Memory plateaus with < 10% variance in final quarter

#### 3. GC Pressure Analysis (`GpuOperations_GCPressure_BoundedCollections`)
- Monitors garbage collection frequency
- Validates memory pooling reduces GC pressure
- **Expected Results**:
  - Gen 0 collections: < 100 (pooling minimizes allocations)
  - Gen 1 collections: < 20
  - Gen 2 collections: < 5 (indicates no leaks)

#### 4. Optimizer Memory Leak Detection (`OptimizerVectorUpdates_5KIterations_NoLeak`)
- Simulates 5,000 optimizer updates on 10K parameter vector
- Tests vectorized operations for leaks
- **Pass Criteria**: Memory growth < 10MB

#### 5. Mixed Precision Operations (`MixedPrecisionOperations_5KIterations_NoLeak`)
- Cycles through different operation types (matrix, vector, tensor)
- Tests memory pool with varied allocation patterns
- **Pass Criteria**: Growth rate between halves < 20%

### Resource Cleanup Tests

#### 1. Engine Disposal (`GpuEngine_MultipleCreateDispose_NoResourceLeak`)
- Creates and destroys 100 GpuEngine instances
- Tests resource cleanup and finalizer behavior
- **Pass Criteria**: Memory growth < 20MB

#### 2. Tensor Lifecycle (`Tensor_CreateUseDiscard_5KCycles_NoLeak`)
- Creates, uses, and discards tensors 5,000 times
- Tests tensor memory management
- **Pass Criteria**: Memory growth < 15MB

## Running Stress Tests

### Run All Stress Tests

```bash
dotnet test --filter "FullyQualifiedName~StressTests"
```

### Run Specific Test Categories

```bash
# Stress tests only
dotnet test --filter "FullyQualifiedName~GpuStressTests"

# Memory leak tests only
dotnet test --filter "FullyQualifiedName~MemoryLeakTests"
```

### Run Individual Tests

```bash
# Specific stress test
dotnet test --filter "FullyQualifiedName~MatrixMultiply_LongRun_10KIterations"

# Specific memory leak test
dotnet test --filter "FullyQualifiedName~MatrixOperations_5KIterations_LinearGrowthCheck"
```

### Test Execution Time

| Test Category | Estimated Time |
|---------------|----------------|
| Matrix Stress Tests | 2-5 minutes |
| Tensor Stress Tests | 3-7 minutes |
| Layer Stress Tests | 2-4 minutes |
| Memory Pool Tests | 1-2 minutes |
| Recovery Tests | < 1 minute |
| Memory Leak Detection | 5-10 minutes |
| **Total** | **~20-30 minutes** |

## Interpreting Results

### Memory Growth Indicators

**Healthy Memory Profile**:
```
Initial Memory: 50 MB
After 1000 iterations: 52 MB (+2 MB)
After 3000 iterations: 53 MB (+1 MB)
After 5000 iterations: 53 MB (+0 MB) ← Plateau reached
```

**Memory Leak Profile**:
```
Initial Memory: 50 MB
After 1000 iterations: 55 MB (+5 MB)
After 3000 iterations: 65 MB (+10 MB)
After 5000 iterations: 75 MB (+10 MB) ← Linear growth (LEAK)
```

### Performance Stability Indicators

**Healthy Performance**:
```
First quartile avg: 0.45 ms
Last quartile avg: 0.47 ms
Drift: 4.4% ✓ (< 20% threshold)
```

**Performance Degradation**:
```
First quartile avg: 0.45 ms
Last quartile avg: 0.70 ms
Drift: 55.6% ✗ (> 20% threshold - indicates resource exhaustion)
```

### GC Collection Indicators

**Healthy GC Profile** (5,000 iterations):
```
Gen 0: 45 collections  ✓ (< 100)
Gen 1: 8 collections   ✓ (< 20)
Gen 2: 2 collections   ✓ (< 5)
```

**Problematic GC Profile**:
```
Gen 0: 850 collections ✗ (poor memory pooling)
Gen 1: 120 collections ✗ (excessive allocations)
Gen 2: 35 collections  ✗ (likely memory leak)
```

## Troubleshooting

### Test Failures

#### "Memory leaked: X MB growth"

**Possible Causes**:
1. GPU memory pool not reusing buffers
2. Native resource leaks (GPU buffers not freed)
3. Managed object retention (event handlers, closures)

**Debug Steps**:
```bash
# Run with memory profiler
dotnet-trace collect --process-id <pid> --providers Microsoft-Windows-DotNETRuntime:0xC0000001

# Analyze with dotMemory, PerfView, or Visual Studio Profiler
```

#### "Performance degraded by X%"

**Possible Causes**:
1. GPU thermal throttling
2. GPU memory fragmentation
3. Kernel compilation not cached
4. Background GPU applications

**Debug Steps**:
1. Check GPU temperature: `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD)
2. Monitor GPU utilization during test
3. Close background applications using GPU
4. Verify kernel pre-compilation in GpuEngine initialization

#### "Excessive Gen 2 collections"

**Possible Causes**:
1. Large object heap (LOH) fragmentation
2. Long-lived objects not being released
3. Finalizer queue buildup

**Debug Steps**:
```csharp
// Add diagnostic logging
GC.RegisterForFullGCNotification(10, 10);
// Monitor finalizer queue
Console.WriteLine($"Finalizers pending: {GC.GetTotalMemory(false)}");
```

### GPU Not Available

If GPU is not available, stress tests will gracefully skip:
```
GPU not available - skipping test
```

This is expected on systems without CUDA/OpenCL support.

### Test Timeouts

Default xUnit timeout: 5 minutes per test

If tests timeout:
1. Reduce iteration counts for initial testing
2. Run tests individually instead of in bulk
3. Check GPU is responsive (`nvidia-smi` or Task Manager)

## Continuous Integration

### CI/CD Integration

Add to your CI pipeline (with caution - GPU may not be available):

```yaml
- name: Run GPU Stress Tests
  if: has-gpu == true
  run: |
    dotnet test --filter "FullyQualifiedName~GpuStressTests" --logger "trx;LogFileName=stress-tests.trx"
  timeout-minutes: 30

- name: Run Memory Leak Tests
  if: has-gpu == true
  run: |
    dotnet test --filter "FullyQualifiedName~MemoryLeakTests" --logger "trx;LogFileName=leak-tests.trx"
  timeout-minutes: 15
```

### Nightly Builds

Recommended: Run stress tests in nightly builds rather than on every commit
- Tests are long-running (20-30 minutes)
- GPU availability may vary across build agents
- Memory leak detection requires extended runtime

### Regression Detection

Track memory growth and performance metrics over time:

```bash
# Baseline
dotnet test --filter "FullyQualifiedName~StressTests" > baseline-stress.log

# After changes
dotnet test --filter "FullyQualifiedName~StressTests" > current-stress.log

# Compare
diff baseline-stress.log current-stress.log
```

## Best Practices

### Writing New Stress Tests

1. **Use Appropriate Iteration Counts**:
   - Long-running: 5,000-10,000 iterations (leak detection)
   - Medium-running: 1,000 iterations (stability)
   - Short-running: 100 iterations (concurrency)

2. **Monitor Multiple Metrics**:
   - Managed memory (GC.GetTotalMemory)
   - GC collection counts
   - Performance timings
   - Resource handles (if applicable)

3. **Force GC Periodically**:
   ```csharp
   if (i % 1000 == 0)
   {
       GC.Collect();
       GC.WaitForPendingFinalizers();
   }
   ```

4. **Establish Baselines**:
   - Measure initial memory before loop
   - Force full GC to get accurate baseline
   - Account for JIT/warmup overhead

5. **Use Statistical Analysis**:
   - Don't rely on single snapshots
   - Calculate trends, averages, variances
   - Detect patterns (linear growth, plateau)

### Memory Leak Prevention

1. **Use Memory Pooling** (already implemented in GpuEngine)
2. **Avoid Closures in Loops** (captures variables, prevents GC)
3. **Dispose IDisposable Objects** (GPU buffers, streams)
4. **Clear Event Handlers** when done
5. **Use Weak References** for caches

## Expected Results

All stress tests and memory leak tests should **PASS** on a properly functioning GPU acceleration system.

**Key Success Metrics**:
- ✅ Memory growth < 15MB over 5,000 iterations
- ✅ Performance drift < 20% between first and last quartiles
- ✅ Gen 2 GC collections < 5 per 5,000 iterations
- ✅ No exceptions in concurrent operations
- ✅ Memory plateaus after warmup period
- ✅ Engine remains functional after errors

## References

- Phase B Implementation: See Epic 4 user stories in issue #496
- GPU Performance Benchmarks: See `GPU_PERFORMANCE_BENCHMARKS.md`
- xUnit Documentation: https://xunit.net/
- .NET Memory Profiling: https://learn.microsoft.com/en-us/dotnet/core/diagnostics/

---

**Last Updated**: 2025-01-17
**Phase**: B - GPU Production Implementation
**Status**: US-GPU-018 Complete
