# ILGPU Cheat Sheet for AiDotNet GPU Acceleration

## Overview
ILGPU is a JIT (just-in-time) compiler for high-performance GPU programs in .NET, completely written in C# without native dependencies, making it truly portable across CUDA, OpenCL, and CPU.

**Current Version**: 1.5.3+
**Target**: GPU acceleration and vectorization for entire AI library

---

## Core Concepts

### 1. Kernel Loading Methods

#### LoadAutoGroupedKernel
- Loads implicitly grouped kernel with auto-determined group size
- **Requires** passing an accelerator stream explicitly
- Use when you need stream control

```csharp
var kernel = accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, float>(MyKernel);
kernel(someStream, buffer.Extent, buffer.View, value);
```

#### LoadAutoGroupedStreamKernel
- Loads implicitly grouped kernel with auto-determined group size
- **Uses default accelerator stream** automatically
- Simpler API when stream control not needed

```csharp
var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float>(MyKernel);
kernel(buffer.Extent, buffer.View, value);
```

**Key Difference**: `LoadAutoGroupedStreamKernel` = `LoadAutoGroupedKernel` + default stream

---

### 2. Memory Buffer Views and Strides

#### As2DView Method
Converts 1D buffer/view to 2D view by specifying dimensions:

```csharp
// From MemoryBuffer1D
var buffer1D = accelerator.Allocate1D<float>(width * height);
var view2D = buffer1D.View.As2DView<Stride2D.DenseY>(new Index2D(height, width));

// From SharedMemory
var shared1D = SharedMemory.Allocate<byte>(totalLength);
var shared2D = shared1D.As2DView(width, height);
```

#### Stride Types (CRITICAL)
**Always use these defaults when unsure:**
- `Stride1D.Dense` - matches C# 1D array layout
- `Stride2D.DenseY` - **most efficient for .NET interop** (recommended)
- `Stride2D.DenseX` - alternative layout
- `Stride3D.DenseZY` - for 3D arrays

**DenseY** = row-major order = C# default = optimal for CPU↔GPU transfer

---

### 3. Shared Memory (Critical for Performance)

Shared memory is **much faster** than global memory - essential for matrix operations.

#### Static Allocation (compile-time known size)
```csharp
var staticMemory = SharedMemory.Allocate<float>(1024);
```

#### Dynamic Allocation (runtime size)
```csharp
var dynamicMemory = SharedMemory.GetDynamic<float>();
```

#### Shared Memory + Explicit Grouping
⚠️ **IMPORTANT**: Using shared memory with **implicitly grouped kernels** leads to undefined behavior!

**Always use explicitly grouped kernels** with shared memory:
```csharp
var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<float>>(MyKernel);
// Launch with explicit group size
kernel((dataSize, groupSize), buffer.View);
```

---

## Matrix Multiplication Best Practices

### Tiled Matrix Multiply (Essential Pattern)

```csharp
static void TiledMatMulKernel(
    Index2D index,
    ArrayView2D<float, Stride2D.DenseY> a,
    ArrayView2D<float, Stride2D.DenseY> b,
    ArrayView2D<float, Stride2D.DenseY> c)
{
    const int TILE_SIZE = 16; // Adjust for your GPU

    // Allocate shared memory tiles
    var aTile = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
        new Index2D(TILE_SIZE, TILE_SIZE),
        new Stride2D.DenseX(TILE_SIZE));
    var bTile = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
        new Index2D(TILE_SIZE, TILE_SIZE),
        new Stride2D.DenseX(TILE_SIZE));

    float sum = 0.0f;

    // Tile loop over inner dimension (K)
    int numTiles = (a.IntExtent.X + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++)
    {
        // Load tiles into shared memory (coalesced access)
        int tiledCol = t * TILE_SIZE + Group.IdxX;
        int tiledRow = t * TILE_SIZE + Group.IdxY;

        aTile[Group.IdxY, Group.IdxX] = (index.Y < a.IntExtent.Y && tiledCol < a.IntExtent.X)
            ? a[index.Y, tiledCol]
            : 0.0f;

        bTile[Group.IdxY, Group.IdxX] = (tiledRow < b.IntExtent.Y && index.X < b.IntExtent.X)
            ? b[tiledRow, index.X]
            : 0.0f;

        Group.Barrier(); // Synchronize tile loading

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++)
            sum += aTile[Group.IdxY, k] * bTile[k, Group.IdxX];

        Group.Barrier(); // Synchronize before next tile
    }

    if (index.Y < c.IntExtent.Y && index.X < c.IntExtent.X)
        c[index] = sum;
}
```

**Key Points**:
- Tile size typically 16x16 or 32x32 (GPU-dependent)
- Shared memory tiles reduce global memory access
- `Group.Barrier()` ensures synchronization
- Coalesced memory access critical for performance

---

## Performance Optimization

### 1. Compilation Optimization
```csharp
var context = Context.Create(builder => builder
    .Optimize(OptimizationLevel.O2)); // Best performance (use in Release)
```

**OptimizationLevel.O2**: Additional transformations, longer compile time, better GPU code.

### 2. Memory Access Patterns

#### Coalesced Global Memory Access
✅ **Good**: Threads in warp access contiguous memory
```csharp
// Thread 0 → data[0], Thread 1 → data[1], etc.
var value = globalData[Grid.GlobalIndex.X];
```

❌ **Bad**: Strided or random access
```csharp
var value = globalData[Grid.GlobalIndex.X * stride]; // Slower!
```

#### Bank Conflict Avoidance
Use padding to avoid shared memory bank conflicts:
```csharp
// 16×17 instead of 16×16 (padding prevents conflicts)
var shared = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
    new Index2D(16, 17), new Stride2D.DenseX(17));
```

### 3. Vectorization Principles
Neural networks benefit massively from vectorization:
- Eliminate `for` loops over batches - compute all in parallel
- Use matrix operations instead of element-wise loops
- **10,000%+ speedup** vs sequential processing

---

## Type Constraints and Errors

### CS0315 / CS0311 Errors
These occur when generic type constraints don't match:

#### Problem
```csharp
public class MyClass<T>  // No constraint
{
    void UseGpu()
    {
        var buffer = accelerator.Allocate1D<T>(100); // ERROR! Needs 'unmanaged'
    }
}
```

#### Solution: Add `unmanaged` constraint
```csharp
public class GpuClass<T> where T : unmanaged
{
    void UseGpu()
    {
        var buffer = accelerator.Allocate1D<T>(100); // ✓ OK
    }
}
```

### Stride Constraints
ArrayView2D requires stride type constraints:
```csharp
void ProcessView<TStride>(ArrayView2D<float, TStride> view)
    where TStride : struct, IStride<Index2D>
{
    // Must constrain TStride properly
}
```

---

## Architecture Recommendations

### CPU Fallback Pattern
```csharp
public class HybridEngine : IEngine
{
    private readonly Accelerator? _gpu;
    private readonly IEngine _cpuFallback;

    public Vector<T> Multiply<T>(Vector<T> a, Vector<T> b) where T : unmanaged
    {
        if (_gpu != null && _gpuHealthy && a.Length > threshold)
            return MultiplyGpu(a, b);

        return _cpuFallback.Multiply(a, b); // CPU fallback
    }
}
```

### GPU Health Monitoring
```csharp
private bool _gpuHealthy = true;
private int _consecutiveFailures = 0;
const int MaxRecoveryAttempts = 3;

private void HandleGpuError(Exception ex)
{
    _consecutiveFailures++;
    if (_consecutiveFailures >= MaxRecoveryAttempts)
    {
        _gpuHealthy = false; // Permanent fallback to CPU
        Console.WriteLine($"GPU permanently disabled after {MaxRecoveryAttempts} failures");
    }
}
```

---

## Debugging Tips

### 1. Use CPUAccelerator for Debugging
```csharp
#if DEBUG
var accelerator = context.CreateCPUAccelerator(0); // Full C# debugging
#else
var accelerator = context.CreateCudaAccelerator(0); // Production GPU
#endif
```

CPUAccelerator is slow but allows full C# debugging features.

### 2. Verify GPU Capabilities
```csharp
Console.WriteLine($"Name: {accelerator.Name}");
Console.WriteLine($"Memory: {accelerator.MemorySize / (1024 * 1024)} MB");
Console.WriteLine($"Max Group Size: {accelerator.MaxNumThreadsPerGroup}");
Console.WriteLine($"Warp Size: {accelerator.WarpSize}");
```

---

## Common Patterns for AI Library

### 1. Element-wise Operations
```csharp
static void ElementWiseKernel(
    Index1D index,
    ArrayView<float> input,
    ArrayView<float> output,
    float scalar)
{
    output[index] = input[index] * scalar + 0.5f; // Fast!
}
```

### 2. Reduction Operations (Sum, Max, etc.)
```csharp
static void SumReductionKernel(
    ArrayView<float> input,
    ArrayView<float> output)
{
    var shared = SharedMemory.Allocate<float>(Group.DimX);
    var localIdx = Group.IdxX;
    var globalIdx = Grid.GlobalIndex.X;

    // Load into shared memory
    shared[localIdx] = globalIdx < input.Length ? input[globalIdx] : 0;
    Group.Barrier();

    // Tree reduction
    for (int stride = Group.DimX / 2; stride > 0; stride >>= 1)
    {
        if (localIdx < stride)
            shared[localIdx] += shared[localIdx + stride];
        Group.Barrier();
    }

    if (localIdx == 0)
        Atomic.Add(ref output[0], shared[0]);
}
```

### 3. Activation Functions (ReLU, Sigmoid, etc.)
```csharp
static void ReLUKernel(Index1D index, ArrayView<float> data)
{
    data[index] = XMath.Max(0.0f, data[index]);
}

static void SigmoidKernel(Index1D index, ArrayView<float> data)
{
    data[index] = 1.0f / (1.0f + XMath.Exp(-data[index]));
}
```

---

## AiDotNet Specific Integration

### Global Engine Pattern
```csharp
public static class AiDotNetEngine
{
    public static IEngine Current { get; set; } = new CpuEngine();
}

public abstract class LayerBase<T>
{
    protected IEngine Engine => AiDotNetEngine.Current;

    // No GPU-specific code here - keep CPU-compatible
}

public abstract class GpuLayerBase<T> : LayerBase<T> where T : unmanaged
{
    // GPU-specific members here
    protected GpuContext<T> GpuContext { get; set; }
}
```

### INumericOperations Compatibility
Keep CPU path unconstrained for maximum flexibility:
```csharp
// CPU: Works with ANY T supporting INumericOperations<T>
public class DenseLayer<T> : LayerBase<T> { }

// GPU: Requires T : unmanaged (float, double, int, etc.)
public class GpuDenseLayer<T> : GpuLayerBase<T> where T : unmanaged { }
```

---

## Error Resolution Quick Reference

| Error | Cause | Solution |
|-------|-------|----------|
| CS0029 | LoadAutoGroupedKernel delegate signature | Add `AcceleratorStream` as first parameter |
| CS0315 | Type doesn't satisfy constraint | Add `where T : unmanaged` |
| CS0311 | Generic argument doesn't match | Check stride/constraint types |
| CS1593 | Delegate signature mismatch | Match kernel parameter types |
| CS7036 | Missing required arguments | Add missing kernel parameters |
| CS1061: LoadAutoGroupedStreamKernel | Method doesn't exist in ILGPU 1.5.3 | Use `LoadAutoGroupedKernel` instead |
| CS1061: ViewAs2DView | Method doesn't exist | Use `As2DView<TStride>` instead |

### CS0029: LoadAutoGroupedKernel Delegate Signature Mismatch

**❌ WRONG:**
```csharp
private readonly Action<Index1D, ArrayView<float>, ArrayView<float>>? _myKernel;

_myKernel = _accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>>(MyKernelMethod);
// ERROR CS0029: Cannot implicitly convert
// Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>
// to Action<Index1D, ArrayView<float>, ArrayView<float>>
```

**✅ CORRECT:**
```csharp
private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _myKernel;

_myKernel = _accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>>(MyKernelMethod);
// ✓ OK - delegate signature matches

// Then invoke with default stream:
_myKernel(_accelerator.DefaultStream, dataLength, inputView, outputView);
```

**Key Point**: `LoadAutoGroupedKernel` requires `AcceleratorStream` as the FIRST parameter in the delegate signature.

---

## Performance Checklist

✅ Use `OptimizationLevel.O2` in Release builds
✅ Implement tiling for matrix operations
✅ Use shared memory for frequently accessed data
✅ Ensure coalesced global memory access
✅ Add `Group.Barrier()` after shared memory writes
✅ Use explicitly grouped kernels with shared memory
✅ Pad shared memory arrays to avoid bank conflicts
✅ Implement CPU fallback for small inputs
✅ Monitor GPU health and handle failures gracefully
✅ Use `Stride2D.DenseY` for CPU↔GPU transfers

---

## Version-Specific Notes (ILGPU 1.5.3+)

- ✅ Nullable annotations on all APIs
- ✅ New Vector data types for .NET 7+
- ✅ Specialized sparse-matrix extensions
- ✅ Improved stride system with explicit type info
- ⚠️ Use `LoadAutoGroupedKernel` not `LoadAutoGroupedStreamKernel`
- ⚠️ Use `As2DView` not `As2DDenseX` or `ViewAs2DView`

---

## Next Steps for AiDotNet

1. ✅ Fix non-GPU errors (COMPLETED)
2. 🔄 Fix CS1061 ILGPU API errors (LoadAutoGroupedStreamKernel, ViewAs2DView)
3. 🔄 Fix CS0315/CS0311 constraint errors (add `unmanaged` where needed)
4. 🔄 Fix CS1593 delegate signature mismatches
5. 🔄 Fix CS7036 missing kernel arguments
6. ⏳ Implement matrix multiplication with tiling
7. ⏳ Optimize activation functions with GPU kernels
8. ⏳ Add vectorized batch processing
9. ⏳ Performance benchmarking and profiling

---

**Last Updated**: 2025-01-18
**Based on**: ILGPU 1.5.3 official documentation and 2024 best practices
