# ILGPU Implementation Guide for AiDotNet

## Architecture Summary

### The Problem
- **INumericOperations<T> Flexibility**: CPU logic works with any numeric type (no constraints)
- **ILGPU Requirements**: GPU operations require `where T : unmanaged` constraint
- **Conflict**: Cannot force `unmanaged` constraint on entire codebase without breaking CPU-only users

### The Solution: Dual-Path Architecture

**Option B: Separate GPU-Specific Class Variants**

1. **Keep existing classes UNCONSTRAINED** - CPU-only logic
   - `DenseLayer<T> : LayerBase<T>` (no constraint)
   - `AdamOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>` (no constraint)
   - These contain ONLY CPU logic - no GPU code

2. **Create NEW GPU-specific variants WITH constraints**
   - `GpuDenseLayer<T> : GpuLayerBase<T> where T : unmanaged`
   - `GpuAdamOptimizer<T, TInput, TOutput> : GpuGradientBasedOptimizerBase<T, TInput, TOutput> where T : unmanaged`
   - These contain GPU-accelerated implementations

3. **Base class hierarchy**
   ```
   LayerBase<T>  (no constraint)
       ↓
   GpuLayerBase<T> where T : unmanaged
       ↓
   GpuDenseLayer<T> where T : unmanaged

   vs.

   LayerBase<T> (no constraint)
       ↓
   DenseLayer<T> (no constraint) - CPU only
   ```

## Implementation Steps

### Step 1: Constrain GPU-Specific Infrastructure
```csharp
// src/Gpu/GpuTensor.cs
public class GpuTensor<T> : IDisposable
    where T : unmanaged  // ADD THIS
{
    // ...
}

// src/Gpu/ExecutionContext.cs
public class ExecutionContext<T> : IDisposable
    where T : unmanaged  // ADD THIS
{
    // ...
}
```

### Step 2: Keep Base Classes Unconstrained
```csharp
// src/NeuralNetworks/Layers/LayerBase.cs
public abstract class LayerBase<T> : ILayer<T>
    // NO constraint - stays flexible for CPU
{
    // Contains only CPU logic and interface implementations
}

// src/Optimizers/GradientBasedOptimizerBase.cs
public abstract class GradientBasedOptimizerBase<T, TInput, TOutput>
    : OptimizerBase<T, TInput, TOutput>
    // NO constraint
{
    // Contains only CPU logic
}
```

### Step 3: Create GPU-Specific Base Classes
```csharp
// src/NeuralNetworks/Layers/GpuLayerBase.cs (NEW FILE)
public abstract class GpuLayerBase<T> : LayerBase<T>
    where T : unmanaged  // GPU classes need this
{
    protected ExecutionContext<T> GpuContext { get; private set; }

    protected GpuLayerBase(/* parameters */)
        : base(/* base parameters */)
    {
        // Initialize GPU context
        if (AiDotNetEngine.Current is GpuEngine gpuEngine)
        {
            GpuContext = new ExecutionContext<T>(gpuEngine.Accelerator);
        }
    }

    // GPU-specific helper methods can go here
}

// src/Optimizers/GpuGradientBasedOptimizerBase.cs (NEW FILE)
public abstract class GpuGradientBasedOptimizerBase<T, TInput, TOutput>
    : GradientBasedOptimizerBase<T, TInput, TOutput>
    where T : unmanaged
{
    protected ExecutionContext<T> GpuContext { get; private set; }

    protected GpuGradientBasedOptimizerBase(/* parameters */)
        : base(/* base parameters */)
    {
        if (AiDotNetEngine.Current is GpuEngine gpuEngine)
        {
            GpuContext = new ExecutionContext<T>(gpuEngine.Accelerator);
        }
    }
}
```

### Step 4: Revert Existing Classes to CPU-Only
```csharp
// src/NeuralNetworks/Layers/DenseLayer.cs
public class DenseLayer<T> : LayerBase<T>  // Revert to LayerBase<T>
    // NO where T : unmanaged constraint
    // NO GpuContext field
{
    // Contains ONLY CPU implementation
    // Remove all GPU-specific code

    public override Tensor<T> Forward(Tensor<T> input)
    {
        // CPU implementation using INumericOperations<T>
    }

    public override Tensor<T> Backward(Tensor<T> gradient)
    {
        // CPU implementation
    }
}
```

### Step 5: Create GPU-Specific Implementations
```csharp
// src/NeuralNetworks/Layers/GpuDenseLayer.cs (NEW FILE)
public class GpuDenseLayer<T> : GpuLayerBase<T>
    where T : unmanaged  // Inherits from GpuLayerBase
{
    // Same interface as DenseLayer but GPU-accelerated

    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Check if GPU is available
        if (GpuContext != null && AiDotNetEngine.Current is GpuEngine)
        {
            return ForwardGpu(input);
        }

        // Fallback to CPU
        return ForwardCpu(input);
    }

    private Tensor<T> ForwardGpu(Tensor<T> input)
    {
        // GPU implementation using GpuContext
        // Can use ExecutionContext<T> because we have unmanaged constraint
    }

    private Tensor<T> ForwardCpu(Tensor<T> input)
    {
        // CPU fallback
    }
}
```

### Step 6: Factory Pattern for User Convenience
```csharp
// User code can use factory to get appropriate implementation
public static class LayerFactory
{
    public static ILayer<T> CreateDenseLayer<T>(int inputSize, int outputSize)
    {
        // Check if GPU is available and T is unmanaged
        if (AiDotNetEngine.Current is GpuEngine && IsUnmanaged<T>())
        {
            return CreateGpuDenseLayer<T>(inputSize, outputSize);
        }

        return new DenseLayer<T>(inputSize, outputSize);
    }

    private static ILayer<T> CreateGpuDenseLayer<T>(int inputSize, int outputSize)
        where T : unmanaged
    {
        return new GpuDenseLayer<T>(inputSize, outputSize);
    }

    private static bool IsUnmanaged<T>()
    {
        // Runtime check for unmanaged constraint
        return typeof(T).IsValueType &&
               !typeof(T).IsPrimitive &&
               typeof(T).GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                   .All(f => IsUnmanaged(f.FieldType));
    }
}
```

## ILGPU Memory Operations

### Problem: ILGPU Copy Methods
The correct way to copy data between CPU and GPU in ILGPU:

```csharp
// ❌ WRONG - Extension methods don't work with constraints
gpuBuffer.CopyFromCPU(data);
gpuBuffer.View.CopyFromCPU(data);
_accelerator.Copy(data, gpuBuffer);  // This method doesn't exist

// ✅ CORRECT - Use MemoryBuffer methods directly
using var gpuBuffer = _accelerator.Allocate1D<T>(length);

// CPU → GPU
gpuBuffer.CopyFromCPU(data.AsSpan());

// GPU → CPU
gpuBuffer.CopyToCPU(result.AsSpan());

// GPU → GPU
_accelerator.MemCopy(sourceBuffer.View, destBuffer.View);
```

### Complete ILGPU Operation Example
```csharp
public Vector<T> AddGpu<T>(Vector<T> a, Vector<T> b)
    where T : unmanaged
{
    // Allocate GPU memory
    using var gpuA = _accelerator.Allocate1D<T>(a.Length);
    using var gpuB = _accelerator.Allocate1D<T>(b.Length);
    using var gpuResult = _accelerator.Allocate1D<T>(a.Length);

    // Copy to GPU
    gpuA.CopyFromCPU(a.AsSpan());
    gpuB.CopyFromCPU(b.AsSpan());

    // Execute kernel
    _addKernel(a.Length, gpuA.View, gpuB.View, gpuResult.View);
    _accelerator.Synchronize();

    // Copy result back
    var result = new Vector<T>(a.Length);
    gpuResult.CopyToCPU(result.AsWritableSpan());

    return result;
}

// Kernel definition
private static void AddKernel<T>(
    Index1D index,
    ArrayView1D<T, Stride1D.Dense> a,
    ArrayView1D<T, Stride1D.Dense> b,
    ArrayView1D<T, Stride1D.Dense> result)
    where T : unmanaged
{
    result[index] = a[index] + b[index];
}
```

## Migration Path

### For Classes That Need GPU Acceleration:
1. Keep original class (e.g., `DenseLayer<T>`) with CPU-only logic
2. Create new GPU variant (e.g., `GpuDenseLayer<T>`)
3. Update factory/builder to choose appropriate implementation

### For Classes That Don't Need GPU:
- No changes needed! They stay unconstrained and CPU-only

### Benefits:
- ✅ Preserves `INumericOperations<T>` flexibility for CPU users
- ✅ No breaking changes for existing CPU code
- ✅ Explicit opt-in for GPU acceleration
- ✅ Compile-time safety for GPU operations
- ✅ Clear separation of concerns

## Common ILGPU Types
- `Accelerator` - GPU device abstraction
- `MemoryBuffer1D<T>` - GPU memory allocation
- `ArrayView1D<T>` - View into GPU memory (from `.View` property)
- `Index1D` - 1D kernel index
- `Stride1D.Dense` - Contiguous memory layout

## Unmanaged Types in .NET
- ✅ `float`, `double`, `int`, `long`, `short`, `byte`, `uint`, `ulong`, `ushort`, `sbyte`
- ✅ Custom structs containing only unmanaged types
- ❌ `decimal` (not blittable)
- ❌ `BigInteger` (reference type)
- ❌ Any class or reference type
- ❌ Structs containing reference types

## Summary
**The key insight**: Don't force `unmanaged` constraint on classes that don't need it. Create parallel GPU-specific implementations that inherit the constraint from GPU-specific base classes. This preserves flexibility while enabling GPU acceleration where needed.
