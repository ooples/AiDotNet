# GPU Thread Safety - Phase B

## Overview

This document describes the thread safety implementation for AiDotNet's GPU acceleration (Phase B: US-GPU-019).

> Note: ILGPU/GpuEngine has been removed. The details below describe legacy behavior and should be refreshed for DirectGpu backends.

## Thread Safety Guarantees

**GpuEngine is fully thread-safe for concurrent operations**:
- ✅ Multiple threads can call operations simultaneously
- ✅ No race conditions in kernel execution
- ✅ GPU health tracking uses atomic operations
- ✅ Memory pools are thread-safe
- ✅ No deadlocks under high concurrency

## Implementation Details

### 1. GPU Health Tracking (Volatile)

```csharp
// Thread-safe GPU health flag
private volatile bool _gpuHealthy = true;
```

**Why volatile?**
- Ensures all threads see the latest value
- Prevents compiler optimizations that could cache the value
- Provides atomic read/write without full locking overhead

**Usage**:
```csharp
if (SupportsGpu && _gpuHealthy)
{
    // Safe to use GPU
}
```

### 2. Kernel Execution Synchronization

```csharp
// Synchronization lock for GPU operations
private readonly object _gpuLock = new object();
```

**Why locking?**
- Legacy ILGPU accelerator was **not thread-safe**
- Concurrent kernel launches can corrupt GPU state
- Lock ensures only one kernel executes at a time

**Pattern used throughout GpuEngine**:
```csharp
// Thread-safe kernel execution (Phase B: US-GPU-019)
lock (_gpuLock)
{
    _someKernelFloat!(size, inputView, outputView);
    _accelerator!.Synchronize();
}
```

### 3. Memory Pool Thread Safety

The legacy `GpuMemoryPool<T>` class was thread-safe:

```csharp
// Thread-safe collections
private readonly ConcurrentDictionary<int, ConcurrentBag<MemoryBuffer1D<T, Stride1D.Dense>>> _pools;
```

**Features**:
- `ConcurrentDictionary` for bucket management
- `ConcurrentBag` for buffer storage within each bucket
- Lock-free rent/return operations
- Safe for concurrent access from multiple threads

### 4. Lock Scope and Granularity

**What's locked**:
- GPU kernel launches
- Accelerator synchronization

**What's NOT locked** (to maximize parallelism):
- CPU-to-GPU memory transfers (`CopyFromCPU`)
- GPU-to-CPU memory transfers (`CopyToCPU`)
- Memory pool rent/return (already thread-safe)
- CPU fallback operations

**Rationale**:
- Memory transfers can overlap with CPU work
- Only the actual kernel execution requires serialization
- Minimizes lock contention

## Performance Characteristics

### Concurrent Operation Overhead

**Small operations (< 10K elements)**:
- Lock overhead: ~50-100 nanoseconds
- Operation time: ~10 microseconds
- **Impact**: ~1% overhead

**Large operations (> 100K elements)**:
- Lock overhead: ~50-100 nanoseconds
- Operation time: ~1-10 milliseconds
- **Impact**: <0.01% overhead

### Scalability

**Concurrent small operations**:
- Limited by lock serialization
- Throughput: ~10,000 ops/second
- Recommendation: Use CPU engine for concurrent small ops

**Concurrent large operations**:
- Lock overhead negligible
- Throughput limited by GPU compute capacity
- Recommendation: Use GPU engine, serialization overhead is minimal

### Multi-GPU Scenarios

**Current implementation**: Single GPU only
- All operations use the same `_accelerator` instance
- Lock ensures thread-safe access to single GPU

**Future enhancement**: Multi-GPU support
- Create separate `GpuEngine` instance per GPU device
- Each instance has independent lock
- Enables true parallel execution across GPUs

## Testing

### Thread Safety Test Suite

Location: `/tests/AiDotNet.Tests/Concurrency/ThreadSafetyTests.cs`

**Test Coverage**:
1. **Concurrent Vector Operations** (16 threads, 50 ops each)
2. **Concurrent Matrix Multiply** (16 threads, 10 ops each)
3. **Mixed Operation Types** (16 threads, 20 mixed ops each)
4. **Concurrent Conv2D** (16 threads, 5 ops each)
5. **Concurrent Pooling** (16 threads, 10 ops each)
6. **GPU Health Tracking** (16 threads, 1000 reads each)
7. **Deadlock Prevention** (32 threads, 100 ops each with timeout)

**Running tests**:
```bash
dotnet test --filter "FullyQualifiedName~ThreadSafetyTests"
```

**Expected results**:
- ✅ All operations complete successfully
- ✅ No exceptions or race conditions
- ✅ All results mathematically correct
- ✅ No deadlocks (completes within timeout)

### Stress Testing

For extended stress testing under high concurrency:
```bash
dotnet test --filter "FullyQualifiedName~ThreadSafetyTests.HighConcurrencyLoad"
```

This runs 32 threads × 100 operations with a 30-second timeout to detect deadlocks.

## Common Patterns

### Safe Concurrent Usage

```csharp
// Multiple threads can safely call GPU operations
var gpuEngine = new GpuEngine();

Parallel.For(0, 100, i =>
{
    var vector = CreateVector(1000);
    var result = gpuEngine.Add(vector, vector);
    // Thread-safe!
});
```

### Shared Engine Instance

```csharp
// Single shared engine instance (recommended)
public class MyNeuralNetwork
{
    private readonly IEngine _engine = new GpuEngine();

    public void TrainBatch(Batch[] batches)
    {
        // Process batches in parallel - thread-safe
        Parallel.ForEach(batches, batch =>
        {
            var gradients = ComputeGradients(batch);
            var update = _engine.Multiply(gradients, learningRate);
            // Safe!
        });
    }
}
```

### Per-Thread Engine Instances (Advanced)

```csharp
// For maximum throughput with many small operations
// (Avoids lock contention but uses more GPU memory)

[ThreadStatic]
private static GpuEngine? _threadLocalEngine;

public static IEngine GetThreadLocalEngine()
{
    if (_threadLocalEngine == null)
        _threadLocalEngine = new GpuEngine();
    return _threadLocalEngine;
}

// Each thread gets its own engine instance
// Note: Only beneficial if you have multiple GPUs!
```

## Troubleshooting

### "Operation Timed Out" or Suspected Deadlock

**Symptoms**:
- Operations hang indefinitely
- Threads blocked waiting for lock

**Potential causes**:
1. GPU driver issue causing kernel hang
2. Very long-running operation
3. Actual deadlock (should not happen with current implementation)

**Debug steps**:
```csharp
// Enable lock diagnostics (requires custom build)
// Add to GpuEngine.cs:

private readonly object _gpuLock = new object();

lock (_gpuLock)
{
    Console.WriteLine($"[Thread {Thread.CurrentThread.ManagedThreadId}] Acquired GPU lock");
    try
    {
        _kernel!(args);
        _accelerator!.Synchronize();
    }
    finally
    {
        Console.WriteLine($"[Thread {Thread.CurrentThread.ManagedThreadId}] Released GPU lock");
    }
}
```

### High Lock Contention

**Symptoms**:
- Many threads waiting for GPU lock
- Poor concurrent throughput

**Solutions**:
1. **Batch operations**: Combine many small operations into larger ones
2. **Use CPU engine**: For small concurrent ops, CPU may be faster
3. **Reduce thread count**: Match concurrency to GPU capacity
4. **Profile lock wait time**: Use concurrency profilers

### Race Conditions Despite Locking

**Should not happen** - all kernel executions are serialized

**If you see non-deterministic results**:
1. Check if `_gpuHealthy` flag is being set correctly
2. Verify all `Synchronize()` calls are inside locks
3. Ensure memory pool returns are happening correctly

## Best Practices

### ✅ DO

- **Share a single GpuEngine instance** across threads
- **Use GPU for large operations** (> 100K elements)
- **Batch small operations** before sending to GPU
- **Let the engine handle synchronization** automatically
- **Use CPU fallback for small concurrent ops**

### ❌ DON'T

- **Don't create many GpuEngine instances** (wastes GPU memory)
- **Don't use GPU for tiny concurrent operations** (lock overhead dominates)
- **Don't manually synchronize** the accelerator outside of GpuEngine
- **Don't assume operations are parallel** (they're serialized by the lock)

## Future Enhancements

### Planned Improvements

1. **Multi-stream execution**:
   - Use CUDA streams for true concurrent kernel execution
   - Requires legacy ILGPU stream support
   - Would enable parallel execution of independent operations

2. **Lock-free fast path**:
   - Use Interlocked for simple operations
   - Reserve locks for complex multi-kernel operations
   - Could improve small operation throughput

3. **Per-operation locks**:
   - Separate locks for different operation types
   - Enable parallel Conv2D + MatMul, etc.
   - Requires careful deadlock prevention

4. **Multi-GPU load balancing**:
   - Distribute operations across multiple GPUs
   - Automatic work stealing
   - Requires GPU topology discovery

## References

- **Legacy ILGPU Threading Model**: https://github.com/m4rs-mt/ILGPU/wiki/FAQ#is-ilgpu-thread-safe
- **.NET Thread Safety**: https://learn.microsoft.com/en-us/dotnet/standard/threading/
- **Volatile Keyword**: https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/volatile
- **ConcurrentBag**: https://learn.microsoft.com/en-us/dotnet/api/system.collections.concurrent.concurrentbag-1

---

**Last Updated**: 2025-01-17
**Phase**: B - GPU Production Implementation
**Status**: US-GPU-019 Complete
