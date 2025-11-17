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
    private readonly GpuMemoryPool<float>? _memoryPoolFloat;
    private bool _disposed;

    // Kernel cache for float operations (pre-compiled in constructor)
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _addKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _subtractKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _multiplyKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, float, ArrayView<float>>? _multiplyScalarKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _divideKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, float, ArrayView<float>>? _divideScalarKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>>? _sqrtKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, float, ArrayView<float>>? _powerKernelFloat;

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

                // Pre-compile all kernels for float operations (Phase B: US-GPU-001)
                Console.WriteLine("[GpuEngine] Pre-compiling GPU kernels...");

                _addKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);

                _subtractKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = a[index] - b[index]);

                _multiplyKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = a[index] * b[index]);

                _multiplyScalarKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, float, ArrayView<float>>(
                    (index, vec, scalar, result) => result[index] = vec[index] * scalar);

                _divideKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = a[index] / b[index]);

                _divideScalarKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, float, ArrayView<float>>(
                    (index, vec, scalar, result) => result[index] = vec[index] / scalar);

                _sqrtKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, vec, result) => result[index] = XMath.Sqrt(vec[index]));

                _powerKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, float, ArrayView<float>>(
                    (index, vec, exp, result) => result[index] = XMath.Pow(vec[index], exp));

                Console.WriteLine("[GpuEngine] Kernel pre-compilation complete");

                // Initialize memory pool for float operations (Phase B: US-GPU-002)
                _memoryPoolFloat = new GpuMemoryPool<float>(_accelerator);
                Console.WriteLine("[GpuEngine] Memory pool initialized");
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

        // Rent GPU memory from pool (Phase B: US-GPU-002)
        var gpuA = _memoryPoolFloat!.Rent(a.Length);
        var gpuB = _memoryPoolFloat.Rent(b.Length);
        var gpuResult = _memoryPoolFloat.Rent(a.Length);

        try
        {
            // Copy to GPU
            gpuA.CopyFromCPU(a.ToArray());
            gpuB.CopyFromCPU(b.ToArray());

            // Use pre-compiled cached kernel (Phase B: US-GPU-001)
            _addKernelFloat!(a.Length, gpuA.View, gpuB.View, gpuResult.View);
            _accelerator!.Synchronize();

            // Copy back to CPU
            gpuResult.CopyToCPU(result.ToArray());

            return result;
        }
        finally
        {
            // Return buffers to pool for reuse
            _memoryPoolFloat.Return(gpuA);
            _memoryPoolFloat.Return(gpuB);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> SubtractGpu(Vector<float> a, Vector<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<float>(a.Length);

        var gpuA = _memoryPoolFloat!.Rent(a.Length);
        var gpuB = _memoryPoolFloat.Rent(b.Length);
        var gpuResult = _memoryPoolFloat.Rent(a.Length);

        try
        {
            gpuA.CopyFromCPU(a.ToArray());
            gpuB.CopyFromCPU(b.ToArray());

            _subtractKernelFloat!(a.Length, gpuA.View, gpuB.View, gpuResult.View);
            _accelerator!.Synchronize();

            gpuResult.CopyToCPU(result.ToArray());

            return result;
        }
        finally
        {
            _memoryPoolFloat.Return(gpuA);
            _memoryPoolFloat.Return(gpuB);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> MultiplyGpu(Vector<float> a, Vector<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<float>(a.Length);
        var gpuA = _memoryPoolFloat!.Rent(a.Length);
        var gpuB = _memoryPoolFloat.Rent(b.Length);
        var gpuResult = _memoryPoolFloat.Rent(a.Length);

        try
        {
            gpuA.CopyFromCPU(a.ToArray());
            gpuB.CopyFromCPU(b.ToArray());
            _multiplyKernelFloat!(a.Length, gpuA.View, gpuB.View, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.ToArray());
            return result;
        }
        finally
        {
            _memoryPoolFloat.Return(gpuA);
            _memoryPoolFloat.Return(gpuB);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> MultiplyScalarGpu(Vector<float> vector, float scalar)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = _memoryPoolFloat!.Rent(vector.Length);
        var gpuResult = _memoryPoolFloat.Rent(vector.Length);

        try
        {
            gpuVector.CopyFromCPU(vector.ToArray());
            _multiplyScalarKernelFloat!(vector.Length, gpuVector.View, scalar, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.ToArray());
            return result;
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> DivideGpu(Vector<float> a, Vector<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<float>(a.Length);
        var gpuA = _memoryPoolFloat!.Rent(a.Length);
        var gpuB = _memoryPoolFloat.Rent(b.Length);
        var gpuResult = _memoryPoolFloat.Rent(a.Length);

        try
        {
            gpuA.CopyFromCPU(a.ToArray());
            gpuB.CopyFromCPU(b.ToArray());
            _divideKernelFloat!(a.Length, gpuA.View, gpuB.View, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.ToArray());
            return result;
        }
        finally
        {
            _memoryPoolFloat.Return(gpuA);
            _memoryPoolFloat.Return(gpuB);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> DivideScalarGpu(Vector<float> vector, float scalar)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = _memoryPoolFloat!.Rent(vector.Length);
        var gpuResult = _memoryPoolFloat.Rent(vector.Length);

        try
        {
            gpuVector.CopyFromCPU(vector.ToArray());
            _divideScalarKernelFloat!(vector.Length, gpuVector.View, scalar, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.ToArray());
            return result;
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> SqrtGpu(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = _memoryPoolFloat!.Rent(vector.Length);
        var gpuResult = _memoryPoolFloat.Rent(vector.Length);

        try
        {
            gpuVector.CopyFromCPU(vector.ToArray());
            _sqrtKernelFloat!(vector.Length, gpuVector.View, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.ToArray());
            return result;
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> PowerGpu(Vector<float> vector, float exponent)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = _memoryPoolFloat!.Rent(vector.Length);
        var gpuResult = _memoryPoolFloat.Rent(vector.Length);

        try
        {
            gpuVector.CopyFromCPU(vector.ToArray());
            _powerKernelFloat!(vector.Length, gpuVector.View, exponent, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.ToArray());
            return result;
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    #endregion

    /// <summary>
    /// Disposes GPU resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;

        _memoryPoolFloat?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
