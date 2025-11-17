using AiDotNet.LinearAlgebra;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace AiDotNet.Engines;

/// <summary>
/// GPU-based execution engine using ILGPU for hardware acceleration.
/// </summary>
/// <remarks>
/// <para>
/// GpuEngine provides GPU acceleration for supported numeric types (currently float).
/// Operations on unsupported types automatically fallback to CpuEngine.
/// </para>
/// <para><b>For Beginners:</b> This is the "turbo mode" for your calculations!
///
/// GpuEngine characteristics:
/// - 10-100x faster for large operations (> 100K elements)
/// - Works with float (more types coming soon)
/// - Automatically falls back to CPU for unsupported types
/// - Requires compatible GPU (NVIDIA CUDA, AMD OpenCL, or Intel)
///
/// When to use:
/// - Large neural networks (millions of parameters)
/// - Big datasets (100K+ samples)
/// - Float precision is sufficient
/// - You have a compatible GPU
///
/// The engine handles all the complexity - you just write normal code!
/// </para>
/// </remarks>
public class GpuEngine : IEngine, IDisposable
{
    private readonly Context? _context;
    private readonly Accelerator? _accelerator;
    private readonly CpuEngine _cpuFallback;
    private bool _disposed;

    /// <inheritdoc/>
    public string Name => _accelerator != null
        ? $"GPU Engine ({_accelerator.Name})"
        : "GPU Engine (Not Available)";

    /// <inheritdoc/>
    public bool SupportsGpu => _accelerator != null;

    /// <summary>
    /// Initializes a new instance of the GpuEngine class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The constructor attempts to initialize GPU acceleration. If no compatible GPU
    /// is found, the engine will still work but operations will fallback to CPU.
    /// </para>
    /// </remarks>
    public GpuEngine()
    {
        _cpuFallback = new CpuEngine();

        try
        {
            // Create ILGPU context
            _context = Context.CreateDefault();

            // Try to get preferred device (GPU over CPU)
            var device = _context.GetPreferredDevice(preferCPU: false);

            if (device.AcceleratorType != AcceleratorType.CPU)
            {
                _accelerator = device.CreateAccelerator(_context);
                Console.WriteLine($"[GpuEngine] Initialized: {_accelerator.Name}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuEngine] GPU initialization failed: {ex.Message}");
            Console.WriteLine("[GpuEngine] Operations will fallback to CPU");
        }
    }

    /// <inheritdoc/>
    public Vector<T> Add<T>(Vector<T> a, Vector<T> b)
    {
        // Runtime type check - only float supported on GPU currently
        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)AddGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
        }

        // Fallback to CPU for unsupported types
        return _cpuFallback.Add(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Subtract<T>(Vector<T> a, Vector<T> b)
    {
        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)SubtractGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
        }

        return _cpuFallback.Subtract(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Multiply<T>(Vector<T> a, Vector<T> b)
    {
        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)MultiplyGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
        }

        return _cpuFallback.Multiply(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Multiply<T>(Vector<T> vector, T scalar)
    {
        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)MultiplyScalarGpu((Vector<float>)(object)vector, (float)(object)scalar!);
        }

        return _cpuFallback.Multiply(vector, scalar);
    }

    /// <inheritdoc/>
    public Vector<T> Divide<T>(Vector<T> a, Vector<T> b)
    {
        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)DivideGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
        }

        return _cpuFallback.Divide(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Divide<T>(Vector<T> vector, T scalar)
    {
        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)DivideScalarGpu((Vector<float>)(object)vector, (float)(object)scalar!);
        }

        return _cpuFallback.Divide(vector, scalar);
    }

    /// <inheritdoc/>
    public Vector<T> Sqrt<T>(Vector<T> vector)
    {
        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)SqrtGpu((Vector<float>)(object)vector);
        }

        return _cpuFallback.Sqrt(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Power<T>(Vector<T> vector, T exponent)
    {
        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)PowerGpu((Vector<float>)(object)vector, (float)(object)exponent!);
        }

        return _cpuFallback.Power(vector, exponent);
    }

    #region GPU Kernels (Float Implementation)

    // Note: These are simple, unoptimized kernels for the prototype.
    // Production implementation would use optimized ILGPU.Algorithms or custom kernels.

    private Vector<float> AddGpu(Vector<float> a, Vector<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<float>(a.Length);

        // Allocate GPU memory
        using var gpuA = _accelerator!.Allocate1D<float>(a.Length);
        using var gpuB = _accelerator.Allocate1D<float>(b.Length);
        using var gpuResult = _accelerator.Allocate1D<float>(a.Length);

        // Copy to GPU
        gpuA.CopyFromCPU(a.ToArray());
        gpuB.CopyFromCPU(b.ToArray());

        // Define and launch kernel
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
            (index, aView, bView, resultView) => resultView[index] = aView[index] + bView[index]);

        kernel(a.Length, gpuA.View, gpuB.View, gpuResult.View);
        _accelerator.Synchronize();

        // Copy back to CPU
        gpuResult.CopyToCPU(result.ToArray());

        return result;
    }

    private Vector<float> SubtractGpu(Vector<float> a, Vector<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<float>(a.Length);

        using var gpuA = _accelerator!.Allocate1D<float>(a.Length);
        using var gpuB = _accelerator.Allocate1D<float>(b.Length);
        using var gpuResult = _accelerator.Allocate1D<float>(a.Length);

        gpuA.CopyFromCPU(a.ToArray());
        gpuB.CopyFromCPU(b.ToArray());

        var kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
            (index, aView, bView, resultView) => resultView[index] = aView[index] - bView[index]);

        kernel(a.Length, gpuA.View, gpuB.View, gpuResult.View);
        _accelerator.Synchronize();

        gpuResult.CopyToCPU(result.ToArray());

        return result;
    }

    private Vector<float> MultiplyGpu(Vector<float> a, Vector<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<float>(a.Length);

        using var gpuA = _accelerator!.Allocate1D<float>(a.Length);
        using var gpuB = _accelerator.Allocate1D<float>(b.Length);
        using var gpuResult = _accelerator.Allocate1D<float>(a.Length);

        gpuA.CopyFromCPU(a.ToArray());
        gpuB.CopyFromCPU(b.ToArray());

        var kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
            (index, aView, bView, resultView) => resultView[index] = aView[index] * bView[index]);

        kernel(a.Length, gpuA.View, gpuB.View, gpuResult.View);
        _accelerator.Synchronize();

        gpuResult.CopyToCPU(result.ToArray());

        return result;
    }

    private Vector<float> MultiplyScalarGpu(Vector<float> vector, float scalar)
    {
        var result = new Vector<float>(vector.Length);

        using var gpuVector = _accelerator!.Allocate1D<float>(vector.Length);
        using var gpuResult = _accelerator.Allocate1D<float>(vector.Length);

        gpuVector.CopyFromCPU(vector.ToArray());

        var kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, float, ArrayView<float>>(
            (index, vecView, scalarVal, resultView) => resultView[index] = vecView[index] * scalarVal);

        kernel(vector.Length, gpuVector.View, scalar, gpuResult.View);
        _accelerator.Synchronize();

        gpuResult.CopyToCPU(result.ToArray());

        return result;
    }

    private Vector<float> DivideGpu(Vector<float> a, Vector<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<float>(a.Length);

        using var gpuA = _accelerator!.Allocate1D<float>(a.Length);
        using var gpuB = _accelerator.Allocate1D<float>(b.Length);
        using var gpuResult = _accelerator.Allocate1D<float>(a.Length);

        gpuA.CopyFromCPU(a.ToArray());
        gpuB.CopyFromCPU(b.ToArray());

        var kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
            (index, aView, bView, resultView) => resultView[index] = aView[index] / bView[index]);

        kernel(a.Length, gpuA.View, gpuB.View, gpuResult.View);
        _accelerator.Synchronize();

        gpuResult.CopyToCPU(result.ToArray());

        return result;
    }

    private Vector<float> DivideScalarGpu(Vector<float> vector, float scalar)
    {
        var result = new Vector<float>(vector.Length);

        using var gpuVector = _accelerator!.Allocate1D<float>(vector.Length);
        using var gpuResult = _accelerator.Allocate1D<float>(vector.Length);

        gpuVector.CopyFromCPU(vector.ToArray());

        var kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, float, ArrayView<float>>(
            (index, vecView, scalarVal, resultView) => resultView[index] = vecView[index] / scalarVal);

        kernel(vector.Length, gpuVector.View, scalar, gpuResult.View);
        _accelerator.Synchronize();

        gpuResult.CopyToCPU(result.ToArray());

        return result;
    }

    private Vector<float> SqrtGpu(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);

        using var gpuVector = _accelerator!.Allocate1D<float>(vector.Length);
        using var gpuResult = _accelerator.Allocate1D<float>(vector.Length);

        gpuVector.CopyFromCPU(vector.ToArray());

        var kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(
            (index, vecView, resultView) => resultView[index] = XMath.Sqrt(vecView[index]));

        kernel(vector.Length, gpuVector.View, gpuResult.View);
        _accelerator.Synchronize();

        gpuResult.CopyToCPU(result.ToArray());

        return result;
    }

    private Vector<float> PowerGpu(Vector<float> vector, float exponent)
    {
        var result = new Vector<float>(vector.Length);

        using var gpuVector = _accelerator!.Allocate1D<float>(vector.Length);
        using var gpuResult = _accelerator.Allocate1D<float>(vector.Length);

        gpuVector.CopyFromCPU(vector.ToArray());

        var kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, float, ArrayView<float>>(
            (index, vecView, exp, resultView) => resultView[index] = XMath.Pow(vecView[index], exp));

        kernel(vector.Length, gpuVector.View, exponent, gpuResult.View);
        _accelerator.Synchronize();

        gpuResult.CopyToCPU(result.ToArray());

        return result;
    }

    #endregion

    /// <summary>
    /// Disposes GPU resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;

        _accelerator?.Dispose();
        _context?.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
