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
    private readonly AdaptiveThresholds _thresholds;
    private bool _disposed;
    private bool _gpuHealthy = true; // Track GPU health (Phase B: US-GPU-006)

    // Memory pools (Phase B: US-GPU-002, US-GPU-005)
    private readonly GpuMemoryPool<float>? _memoryPoolFloat;
    private readonly GpuMemoryPool<double>? _memoryPoolDouble;
    private readonly GpuMemoryPool<int>? _memoryPoolInt;
    private readonly GpuMemoryPool<long>? _memoryPoolLong;

    // Kernel cache for float operations (Phase B: US-GPU-001)
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _addKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _subtractKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _multiplyKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, float, ArrayView<float>>? _multiplyScalarKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _divideKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, float, ArrayView<float>>? _divideScalarKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>>? _sqrtKernelFloat;
    private readonly Action<Index1D, ArrayView<float>, float, ArrayView<float>>? _powerKernelFloat;

    // Kernel cache for double operations (Phase B: US-GPU-005)
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _addKernelDouble;
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _subtractKernelDouble;
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _multiplyKernelDouble;
    private readonly Action<Index1D, ArrayView<double>, double, ArrayView<double>>? _multiplyScalarKernelDouble;
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _divideKernelDouble;
    private readonly Action<Index1D, ArrayView<double>, double, ArrayView<double>>? _divideScalarKernelDouble;
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>>? _sqrtKernelDouble;
    private readonly Action<Index1D, ArrayView<double>, double, ArrayView<double>>? _powerKernelDouble;

    // Kernel cache for int operations (Phase B: US-GPU-005)
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>? _addKernelInt;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>? _subtractKernelInt;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>? _multiplyKernelInt;
    private readonly Action<Index1D, ArrayView<int>, int, ArrayView<int>>? _multiplyScalarKernelInt;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>? _divideKernelInt;
    private readonly Action<Index1D, ArrayView<int>, int, ArrayView<int>>? _divideScalarKernelInt;

    // Kernel cache for long operations (Phase B: US-GPU-005)
    private readonly Action<Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>? _addKernelLong;
    private readonly Action<Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>? _subtractKernelLong;
    private readonly Action<Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>? _multiplyKernelLong;
    private readonly Action<Index1D, ArrayView<long>, long, ArrayView<long>>? _multiplyScalarKernelLong;
    private readonly Action<Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>? _divideKernelLong;
    private readonly Action<Index1D, ArrayView<long>, long, ArrayView<long>>? _divideScalarKernelLong;

    // Kernel cache for matrix operations - float (Phase B: Epic 2)
    private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int>? _matrixMultiplyKernelFloat;
    private readonly Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>, int, int>? _matrixVectorMultiplyKernelFloat;
    private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>? _matrixTransposeKernelFloat;
    private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>? _matrixAddKernelFloat;
    private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, float, ArrayView2D<float, Stride2D.DenseX>>? _matrixMultiplyScalarKernelFloat;

    // Kernel cache for matrix operations - double (Phase B: Epic 2)
    private readonly Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, int>? _matrixMultiplyKernelDouble;
    private readonly Action<Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, ArrayView<double>, int, int>? _matrixVectorMultiplyKernelDouble;
    private readonly Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>>? _matrixTransposeKernelDouble;
    private readonly Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>>? _matrixAddKernelDouble;
    private readonly Action<Index2D, ArrayView2D<double, Stride2D.DenseX>, double, ArrayView2D<double, Stride2D.DenseX>>? _matrixMultiplyScalarKernelDouble;

    /// <inheritdoc/>
    public string Name => _accelerator != null
        ? $"GPU Engine ({_accelerator.Name})"
        : "GPU Engine (Not Available)";

    /// <inheritdoc/>
    public bool SupportsGpu => _accelerator != null;

    /// <summary>
    /// Initializes a new instance of the GpuEngine class with default adaptive thresholds.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The constructor attempts to initialize GPU acceleration. If no compatible GPU
    /// is found, the engine will still work but operations will fallback to CPU.
    /// </para>
    /// </remarks>
    public GpuEngine()
        : this(AdaptiveThresholds.Default)
    {
    }

    /// <summary>
    /// Initializes a new instance of the GpuEngine class with custom adaptive thresholds.
    /// </summary>
    /// <param name="thresholds">Custom thresholds for adaptive CPU/GPU routing.</param>
    /// <remarks>
    /// <para>
    /// Use this constructor to fine-tune performance for your specific hardware.
    /// See <see cref="AdaptiveThresholds"/> for preset configurations.
    /// </para>
    /// </remarks>
    public GpuEngine(AdaptiveThresholds thresholds)
    {
        _thresholds = thresholds ?? AdaptiveThresholds.Default;
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

                Console.WriteLine("[GpuEngine] Float kernels pre-compiled");

                // Pre-compile kernels for double operations (Phase B: US-GPU-005)
                _addKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);
                _subtractKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = a[index] - b[index]);
                _multiplyKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = a[index] * b[index]);
                _multiplyScalarKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<double>, double, ArrayView<double>>(
                    (index, vec, scalar, result) => result[index] = vec[index] * scalar);
                _divideKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = a[index] / b[index]);
                _divideScalarKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<double>, double, ArrayView<double>>(
                    (index, vec, scalar, result) => result[index] = vec[index] / scalar);
                _sqrtKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, vec, result) => result[index] = XMath.Sqrt(vec[index]));
                _powerKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<double>, double, ArrayView<double>>(
                    (index, vec, exp, result) => result[index] = XMath.Pow(vec[index], exp));
                Console.WriteLine("[GpuEngine] Double kernels pre-compiled");

                // Pre-compile kernels for int operations (Phase B: US-GPU-005)
                _addKernelInt = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);
                _subtractKernelInt = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                    (index, a, b, result) => result[index] = a[index] - b[index]);
                _multiplyKernelInt = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                    (index, a, b, result) => result[index] = a[index] * b[index]);
                _multiplyScalarKernelInt = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, int, ArrayView<int>>(
                    (index, vec, scalar, result) => result[index] = vec[index] * scalar);
                _divideKernelInt = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                    (index, a, b, result) => result[index] = a[index] / b[index]);
                _divideScalarKernelInt = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, int, ArrayView<int>>(
                    (index, vec, scalar, result) => result[index] = vec[index] / scalar);
                Console.WriteLine("[GpuEngine] Int kernels pre-compiled");

                // Pre-compile kernels for long operations (Phase B: US-GPU-005)
                _addKernelLong = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);
                _subtractKernelLong = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>(
                    (index, a, b, result) => result[index] = a[index] - b[index]);
                _multiplyKernelLong = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>(
                    (index, a, b, result) => result[index] = a[index] * b[index]);
                _multiplyScalarKernelLong = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<long>, long, ArrayView<long>>(
                    (index, vec, scalar, result) => result[index] = vec[index] * scalar);
                _divideKernelLong = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>(
                    (index, a, b, result) => result[index] = a[index] / b[index]);
                _divideScalarKernelLong = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<long>, long, ArrayView<long>>(
                    (index, vec, scalar, result) => result[index] = vec[index] / scalar);
                Console.WriteLine("[GpuEngine] Long kernels pre-compiled");

                // Pre-compile kernels for matrix operations - float (Phase B: Epic 2)
                _matrixMultiplyKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int>(
                    (index, a, b, result, k) =>
                    {
                        float sum = 0;
                        for (int i = 0; i < k; i++)
                            sum += a[index.X, i] * b[i, index.Y];
                        result[index] = sum;
                    });

                _matrixVectorMultiplyKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>, int, int>(
                    (index, matrix, vector, result, rows, cols) =>
                    {
                        float sum = 0;
                        for (int j = 0; j < cols; j++)
                            sum += matrix[index, j] * vector[j];
                        result[index] = sum;
                    });

                _matrixTransposeKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(
                    (index, input, output) => output[index.Y, index.X] = input[index]);

                _matrixAddKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);

                _matrixMultiplyScalarKernelFloat = _accelerator.LoadAutoGroupedStreamKernel<
                    Index2D, ArrayView2D<float, Stride2D.DenseX>, float, ArrayView2D<float, Stride2D.DenseX>>(
                    (index, matrix, scalar, result) => result[index] = matrix[index] * scalar);
                Console.WriteLine("[GpuEngine] Float matrix kernels pre-compiled");

                // Pre-compile kernels for matrix operations - double (Phase B: Epic 2)
                _matrixMultiplyKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, int>(
                    (index, a, b, result, k) =>
                    {
                        double sum = 0;
                        for (int i = 0; i < k; i++)
                            sum += a[index.X, i] * b[i, index.Y];
                        result[index] = sum;
                    });

                _matrixVectorMultiplyKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, ArrayView<double>, int, int>(
                    (index, matrix, vector, result, rows, cols) =>
                    {
                        double sum = 0;
                        for (int j = 0; j < cols; j++)
                            sum += matrix[index, j] * vector[j];
                        result[index] = sum;
                    });

                _matrixTransposeKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>>(
                    (index, input, output) => output[index.Y, index.X] = input[index]);

                _matrixAddKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);

                _matrixMultiplyScalarKernelDouble = _accelerator.LoadAutoGroupedStreamKernel<
                    Index2D, ArrayView2D<double, Stride2D.DenseX>, double, ArrayView2D<double, Stride2D.DenseX>>(
                    (index, matrix, scalar, result) => result[index] = matrix[index] * scalar);
                Console.WriteLine("[GpuEngine] Double matrix kernels pre-compiled");

                Console.WriteLine("[GpuEngine] All kernel pre-compilation complete");

                // Initialize memory pools (Phase B: US-GPU-002, US-GPU-005)
                _memoryPoolFloat = new GpuMemoryPool<float>(_accelerator);
                _memoryPoolDouble = new GpuMemoryPool<double>(_accelerator);
                _memoryPoolInt = new GpuMemoryPool<int>(_accelerator);
                _memoryPoolLong = new GpuMemoryPool<long>(_accelerator);
                Console.WriteLine("[GpuEngine] Memory pools initialized");
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
        // Adaptive execution: check size threshold (Phase B: US-GPU-004)
        if (a.Length < _thresholds.VectorAdd)
        {
            return _cpuFallback.Add(a, b); // CPU for small operations
        }

        // Check GPU health before attempting GPU operations (Phase B: US-GPU-006)
        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)AddGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)AddGpuDouble((Vector<double>)(object)a, (Vector<double>)(object)b);
            if (typeof(T) == typeof(int))
                return (Vector<T>)(object)AddGpuInt((Vector<int>)(object)a, (Vector<int>)(object)b);
            if (typeof(T) == typeof(long))
                return (Vector<T>)(object)AddGpuLong((Vector<long>)(object)a, (Vector<long>)(object)b);
        }

        // Fallback to CPU for unsupported types or unhealthy GPU
        return _cpuFallback.Add(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Subtract<T>(Vector<T> a, Vector<T> b)
    {
        if (a.Length < _thresholds.VectorSubtract)
            return _cpuFallback.Subtract(a, b);

        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)SubtractGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
        }

        return _cpuFallback.Subtract(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Multiply<T>(Vector<T> a, Vector<T> b)
    {
        if (a.Length < _thresholds.VectorMultiply)
            return _cpuFallback.Multiply(a, b);

        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)MultiplyGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
        }

        return _cpuFallback.Multiply(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Multiply<T>(Vector<T> vector, T scalar)
    {
        if (vector.Length < _thresholds.VectorMultiply)
            return _cpuFallback.Multiply(vector, scalar);

        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)MultiplyScalarGpu((Vector<float>)(object)vector, (float)(object)scalar!);
        }

        return _cpuFallback.Multiply(vector, scalar);
    }

    /// <inheritdoc/>
    public Vector<T> Divide<T>(Vector<T> a, Vector<T> b)
    {
        if (a.Length < _thresholds.VectorDivide)
            return _cpuFallback.Divide(a, b);

        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)DivideGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
        }

        return _cpuFallback.Divide(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Divide<T>(Vector<T> vector, T scalar)
    {
        if (vector.Length < _thresholds.VectorDivide)
            return _cpuFallback.Divide(vector, scalar);

        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)DivideScalarGpu((Vector<float>)(object)vector, (float)(object)scalar!);
        }

        return _cpuFallback.Divide(vector, scalar);
    }

    /// <inheritdoc/>
    public Vector<T> Sqrt<T>(Vector<T> vector)
    {
        if (vector.Length < _thresholds.VectorSqrt)
            return _cpuFallback.Sqrt(vector);

        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return (Vector<T>)(object)SqrtGpu((Vector<float>)(object)vector);
        }

        return _cpuFallback.Sqrt(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Power<T>(Vector<T> vector, T exponent)
    {
        if (vector.Length < _thresholds.VectorPower)
            return _cpuFallback.Power(vector, exponent);

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
            // Zero-copy: Use span instead of ToArray() (Phase B: US-GPU-003)
            gpuA.CopyFromCPU(a.AsSpan());
            gpuB.CopyFromCPU(b.AsSpan());

            // Use pre-compiled cached kernel (Phase B: US-GPU-001)
            _addKernelFloat!(a.Length, gpuA.View, gpuB.View, gpuResult.View);
            _accelerator!.Synchronize();

            // Zero-copy: Write directly to result's internal storage (Phase B: US-GPU-003)
            gpuResult.CopyToCPU(result.AsWritableSpan());

            return result;
        }
        catch (OutOfMemoryException ex)
        {
            // GPU memory exhausted - fallback to CPU (Phase B: US-GPU-006)
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.Add(a, b);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            // Critical GPU failure - mark unhealthy and fallback (Phase B: US-GPU-006)
            _gpuHealthy = false;
            Console.WriteLine($"[GpuEngine] Critical GPU failure: {ex.Message}. GPU disabled, all operations will use CPU.");
            return _cpuFallback.Add(a, b);
        }
        catch (Exception ex)
        {
            // GPU operation failed - fallback to CPU (Phase B: US-GPU-006)
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.Add(a, b);
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
            gpuA.CopyFromCPU(a.AsSpan());
            gpuB.CopyFromCPU(b.AsSpan());
            _subtractKernelFloat!(a.Length, gpuA.View, gpuB.View, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.AsWritableSpan());
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
            gpuA.CopyFromCPU(a.AsSpan());
            gpuB.CopyFromCPU(b.AsSpan());
            _multiplyKernelFloat!(a.Length, gpuA.View, gpuB.View, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.AsWritableSpan());
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
            gpuVector.CopyFromCPU(vector.AsSpan());
            _multiplyScalarKernelFloat!(vector.Length, gpuVector.View, scalar, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.AsWritableSpan());
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
            gpuA.CopyFromCPU(a.AsSpan());
            gpuB.CopyFromCPU(b.AsSpan());
            _divideKernelFloat!(a.Length, gpuA.View, gpuB.View, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.AsWritableSpan());
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
            gpuVector.CopyFromCPU(vector.AsSpan());
            _divideScalarKernelFloat!(vector.Length, gpuVector.View, scalar, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.AsWritableSpan());
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
            gpuVector.CopyFromCPU(vector.AsSpan());
            _sqrtKernelFloat!(vector.Length, gpuVector.View, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.AsWritableSpan());
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
            gpuVector.CopyFromCPU(vector.AsSpan());
            _powerKernelFloat!(vector.Length, gpuVector.View, exponent, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    #endregion

    #region GPU Kernels (Double, Int, Long Implementation - Phase B: US-GPU-005)

    // GPU operations for double type
    private Vector<double> AddGpuDouble(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<double>(a.Length);
        var gpuA = _memoryPoolDouble!.Rent(a.Length);
        var gpuB = _memoryPoolDouble.Rent(b.Length);
        var gpuResult = _memoryPoolDouble.Rent(a.Length);

        try
        {
            gpuA.CopyFromCPU(a.AsSpan());
            gpuB.CopyFromCPU(b.AsSpan());
            _addKernelDouble!(a.Length, gpuA.View, gpuB.View, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    // GPU operations for int type
    private Vector<int> AddGpuInt(Vector<int> a, Vector<int> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<int>(a.Length);
        var gpuA = _memoryPoolInt!.Rent(a.Length);
        var gpuB = _memoryPoolInt.Rent(b.Length);
        var gpuResult = _memoryPoolInt.Rent(a.Length);

        try
        {
            gpuA.CopyFromCPU(a.AsSpan());
            gpuB.CopyFromCPU(b.AsSpan());
            _addKernelInt!(a.Length, gpuA.View, gpuB.View, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolInt.Return(gpuA);
            _memoryPoolInt.Return(gpuB);
            _memoryPoolInt.Return(gpuResult);
        }
    }

    // GPU operations for long type
    private Vector<long> AddGpuLong(Vector<long> a, Vector<long> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<long>(a.Length);
        var gpuA = _memoryPoolLong!.Rent(a.Length);
        var gpuB = _memoryPoolLong.Rent(b.Length);
        var gpuResult = _memoryPoolLong.Rent(a.Length);

        try
        {
            gpuA.CopyFromCPU(a.AsSpan());
            gpuB.CopyFromCPU(b.AsSpan());
            _addKernelLong!(a.Length, gpuA.View, gpuB.View, gpuResult.View);
            _accelerator!.Synchronize();
            gpuResult.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolLong.Return(gpuA);
            _memoryPoolLong.Return(gpuB);
            _memoryPoolLong.Return(gpuResult);
        }
    }

    #endregion

    #region Matrix Operations (Phase B: Epic 2)

    /// <inheritdoc/>
    public Matrix<T> MatrixMultiply<T>(Matrix<T> a, Matrix<T> b)
    {
        // Adaptive execution: check matrix size threshold (Phase B: US-GPU-004)
        if (Math.Max(a.Rows, Math.Max(a.Columns, b.Columns)) < _thresholds.MatrixMultiply)
        {
            return _cpuFallback.MatrixMultiply(a, b);
        }

        // Check GPU health and type support (Phase B: US-GPU-006)
        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Matrix<T>)(object)MatrixMultiplyGpu((Matrix<float>)(object)a, (Matrix<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Matrix<T>)(object)MatrixMultiplyGpuDouble((Matrix<double>)(object)a, (Matrix<double>)(object)b);
        }

        // Fallback to CPU for unsupported types or unhealthy GPU
        return _cpuFallback.MatrixMultiply(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> MatrixVectorMultiply<T>(Matrix<T> matrix, Vector<T> vector)
    {
        // Adaptive execution
        if (Math.Max(matrix.Rows, matrix.Columns) < _thresholds.MatrixVectorMultiply)
        {
            return _cpuFallback.MatrixVectorMultiply(matrix, vector);
        }

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)MatrixVectorMultiplyGpu((Matrix<float>)(object)matrix, (Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)MatrixVectorMultiplyGpuDouble((Matrix<double>)(object)matrix, (Vector<double>)(object)vector);
        }

        return _cpuFallback.MatrixVectorMultiply(matrix, vector);
    }

    /// <inheritdoc/>
    public Matrix<T> MatrixTranspose<T>(Matrix<T> matrix)
    {
        // Transpose is memory-bound, benefit from GPU at smaller sizes
        if (Math.Max(matrix.Rows, matrix.Columns) < _thresholds.MatrixMultiply / 2)
        {
            return _cpuFallback.MatrixTranspose(matrix);
        }

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Matrix<T>)(object)MatrixTransposeGpu((Matrix<float>)(object)matrix);
            if (typeof(T) == typeof(double))
                return (Matrix<T>)(object)MatrixTransposeGpuDouble((Matrix<double>)(object)matrix);
        }

        return _cpuFallback.MatrixTranspose(matrix);
    }

    /// <inheritdoc/>
    public Matrix<T> MatrixAdd<T>(Matrix<T> a, Matrix<T> b)
    {
        // Element-wise operations benefit from GPU at similar thresholds to vector ops
        if (a.Rows * a.Columns < _thresholds.VectorAdd)
        {
            return _cpuFallback.MatrixAdd(a, b);
        }

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Matrix<T>)(object)MatrixAddGpu((Matrix<float>)(object)a, (Matrix<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Matrix<T>)(object)MatrixAddGpuDouble((Matrix<double>)(object)a, (Matrix<double>)(object)b);
        }

        return _cpuFallback.MatrixAdd(a, b);
    }

    /// <inheritdoc/>
    public Matrix<T> MatrixMultiplyScalar<T>(Matrix<T> matrix, T scalar)
    {
        if (matrix.Rows * matrix.Columns < _thresholds.VectorMultiply)
        {
            return _cpuFallback.MatrixMultiplyScalar(matrix, scalar);
        }

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Matrix<T>)(object)MatrixMultiplyScalarGpu((Matrix<float>)(object)matrix, (float)(object)scalar);
            if (typeof(T) == typeof(double))
                return (Matrix<T>)(object)MatrixMultiplyScalarGpuDouble((Matrix<double>)(object)matrix, (double)(object)scalar);
        }

        return _cpuFallback.MatrixMultiplyScalar(matrix, scalar);
    }

    // GPU implementations for float matrices

    private Matrix<float> MatrixMultiplyGpu(Matrix<float> a, Matrix<float> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException(
                $"Matrix dimensions incompatible for multiplication. " +
                $"First matrix is {a.Rows}x{a.Columns}, second is {b.Rows}x{b.Columns}.");
        }

        try
        {
            var result = new Matrix<float>(a.Rows, b.Columns);
            int m = a.Rows, k = a.Columns, n = b.Columns;

            // Allocate GPU buffers using memory pool (Phase B: US-GPU-002)
            var gpuA = _memoryPoolFloat!.Rent(m * k);
            var gpuB = _memoryPoolFloat!.Rent(k * n);
            var gpuResult = _memoryPoolFloat!.Rent(m * n);

            try
            {
                // Zero-copy transfer (Phase B: US-GPU-003)
                gpuA.CopyFromCPU(a.AsSpan());
                gpuB.CopyFromCPU(b.AsSpan());

                // Create 2D views
                var viewA = gpuA.View.As2DDenseX<Stride2D.DenseX>(new Index2D(m, k));
                var viewB = gpuB.View.As2DDenseX<Stride2D.DenseX>(new Index2D(k, n));
                var viewResult = gpuResult.View.As2DDenseX<Stride2D.DenseX>(new Index2D(m, n));

                // Execute pre-compiled kernel (Phase B: US-GPU-001, US-GPU-007)
                _matrixMultiplyKernelFloat!(new Index2D(m, n), viewA, viewB, viewResult, k);
                _accelerator!.Synchronize();

                // Zero-copy result transfer
                gpuResult.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuA);
                _memoryPoolFloat.Return(gpuB);
                _memoryPoolFloat.Return(gpuResult);
            }
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted for matrix multiply: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixMultiply(a, b);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            _gpuHealthy = false;
            Console.WriteLine($"[GpuEngine] Critical GPU failure in matrix multiply: {ex.Message}. GPU disabled.");
            return _cpuFallback.MatrixMultiply(a, b);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix multiply failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixMultiply(a, b);
        }
    }

    private Vector<float> MatrixVectorMultiplyGpu(Matrix<float> matrix, Vector<float> vector)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (matrix.Columns != vector.Length)
        {
            throw new ArgumentException(
                $"Matrix-vector dimensions incompatible. Matrix is {matrix.Rows}x{matrix.Columns}, vector has {vector.Length} elements.");
        }

        try
        {
            var result = new Vector<float>(matrix.Rows);
            int rows = matrix.Rows, cols = matrix.Columns;

            var gpuMatrix = _memoryPoolFloat!.Rent(rows * cols);
            var gpuVector = _memoryPoolFloat!.Rent(cols);
            var gpuResult = _memoryPoolFloat!.Rent(rows);

            try
            {
                gpuMatrix.CopyFromCPU(matrix.AsSpan());
                gpuVector.CopyFromCPU(vector.AsSpan());

                var viewMatrix = gpuMatrix.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));
                _matrixVectorMultiplyKernelFloat!(rows, viewMatrix, gpuVector.View, gpuResult.View, rows, cols);
                _accelerator!.Synchronize();

                gpuResult.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuMatrix);
                _memoryPoolFloat.Return(gpuVector);
                _memoryPoolFloat.Return(gpuResult);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix-vector multiply failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixVectorMultiply(matrix, vector);
        }
    }

    private Matrix<float> MatrixTransposeGpu(Matrix<float> matrix)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        try
        {
            var result = new Matrix<float>(matrix.Columns, matrix.Rows);
            int rows = matrix.Rows, cols = matrix.Columns;

            var gpuInput = _memoryPoolFloat!.Rent(rows * cols);
            var gpuOutput = _memoryPoolFloat!.Rent(rows * cols);

            try
            {
                gpuInput.CopyFromCPU(matrix.AsSpan());

                var viewInput = gpuInput.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));
                var viewOutput = gpuOutput.View.As2DDenseX<Stride2D.DenseX>(new Index2D(cols, rows));

                _matrixTransposeKernelFloat!(new Index2D(rows, cols), viewInput, viewOutput);
                _accelerator!.Synchronize();

                gpuOutput.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuInput);
                _memoryPoolFloat.Return(gpuOutput);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix transpose failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixTranspose(matrix);
        }
    }

    private Matrix<float> MatrixAddGpu(Matrix<float> a, Matrix<float> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rows != b.Rows || a.Columns != b.Columns)
        {
            throw new ArgumentException($"Matrix dimensions must match for addition.");
        }

        try
        {
            var result = new Matrix<float>(a.Rows, a.Columns);
            int rows = a.Rows, cols = a.Columns;

            var gpuA = _memoryPoolFloat!.Rent(rows * cols);
            var gpuB = _memoryPoolFloat!.Rent(rows * cols);
            var gpuResult = _memoryPoolFloat!.Rent(rows * cols);

            try
            {
                gpuA.CopyFromCPU(a.AsSpan());
                gpuB.CopyFromCPU(b.AsSpan());

                var viewA = gpuA.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));
                var viewB = gpuB.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));
                var viewResult = gpuResult.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));

                _matrixAddKernelFloat!(new Index2D(rows, cols), viewA, viewB, viewResult);
                _accelerator!.Synchronize();

                gpuResult.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuA);
                _memoryPoolFloat.Return(gpuB);
                _memoryPoolFloat.Return(gpuResult);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix add failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixAdd(a, b);
        }
    }

    private Matrix<float> MatrixMultiplyScalarGpu(Matrix<float> matrix, float scalar)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        try
        {
            var result = new Matrix<float>(matrix.Rows, matrix.Columns);
            int rows = matrix.Rows, cols = matrix.Columns;

            var gpuMatrix = _memoryPoolFloat!.Rent(rows * cols);
            var gpuResult = _memoryPoolFloat!.Rent(rows * cols);

            try
            {
                gpuMatrix.CopyFromCPU(matrix.AsSpan());

                var viewMatrix = gpuMatrix.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));
                var viewResult = gpuResult.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));

                _matrixMultiplyScalarKernelFloat!(new Index2D(rows, cols), viewMatrix, scalar, viewResult);
                _accelerator!.Synchronize();

                gpuResult.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuMatrix);
                _memoryPoolFloat.Return(gpuResult);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix scalar multiply failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixMultiplyScalar(matrix, scalar);
        }
    }

    // GPU implementations for double matrices

    private Matrix<double> MatrixMultiplyGpuDouble(Matrix<double> a, Matrix<double> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException(
                $"Matrix dimensions incompatible for multiplication. " +
                $"First matrix is {a.Rows}x{a.Columns}, second is {b.Rows}x{b.Columns}.");
        }

        try
        {
            var result = new Matrix<double>(a.Rows, b.Columns);
            int m = a.Rows, k = a.Columns, n = b.Columns;

            var gpuA = _memoryPoolDouble!.Rent(m * k);
            var gpuB = _memoryPoolDouble!.Rent(k * n);
            var gpuResult = _memoryPoolDouble!.Rent(m * n);

            try
            {
                gpuA.CopyFromCPU(a.AsSpan());
                gpuB.CopyFromCPU(b.AsSpan());

                var viewA = gpuA.View.As2DDenseX<Stride2D.DenseX>(new Index2D(m, k));
                var viewB = gpuB.View.As2DDenseX<Stride2D.DenseX>(new Index2D(k, n));
                var viewResult = gpuResult.View.As2DDenseX<Stride2D.DenseX>(new Index2D(m, n));

                _matrixMultiplyKernelDouble!(new Index2D(m, n), viewA, viewB, viewResult, k);
                _accelerator!.Synchronize();

                gpuResult.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuA);
                _memoryPoolDouble.Return(gpuB);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix multiply (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixMultiply(a, b);
        }
    }

    private Vector<double> MatrixVectorMultiplyGpuDouble(Matrix<double> matrix, Vector<double> vector)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (matrix.Columns != vector.Length)
        {
            throw new ArgumentException(
                $"Matrix-vector dimensions incompatible. Matrix is {matrix.Rows}x{matrix.Columns}, vector has {vector.Length} elements.");
        }

        try
        {
            var result = new Vector<double>(matrix.Rows);
            int rows = matrix.Rows, cols = matrix.Columns;

            var gpuMatrix = _memoryPoolDouble!.Rent(rows * cols);
            var gpuVector = _memoryPoolDouble!.Rent(cols);
            var gpuResult = _memoryPoolDouble!.Rent(rows);

            try
            {
                gpuMatrix.CopyFromCPU(matrix.AsSpan());
                gpuVector.CopyFromCPU(vector.AsSpan());

                var viewMatrix = gpuMatrix.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));
                _matrixVectorMultiplyKernelDouble!(rows, viewMatrix, gpuVector.View, gpuResult.View, rows, cols);
                _accelerator!.Synchronize();

                gpuResult.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuMatrix);
                _memoryPoolDouble.Return(gpuVector);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix-vector multiply (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixVectorMultiply(matrix, vector);
        }
    }

    private Matrix<double> MatrixTransposeGpuDouble(Matrix<double> matrix)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        try
        {
            var result = new Matrix<double>(matrix.Columns, matrix.Rows);
            int rows = matrix.Rows, cols = matrix.Columns;

            var gpuInput = _memoryPoolDouble!.Rent(rows * cols);
            var gpuOutput = _memoryPoolDouble!.Rent(rows * cols);

            try
            {
                gpuInput.CopyFromCPU(matrix.AsSpan());

                var viewInput = gpuInput.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));
                var viewOutput = gpuOutput.View.As2DDenseX<Stride2D.DenseX>(new Index2D(cols, rows));

                _matrixTransposeKernelDouble!(new Index2D(rows, cols), viewInput, viewOutput);
                _accelerator!.Synchronize();

                gpuOutput.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuInput);
                _memoryPoolDouble.Return(gpuOutput);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix transpose (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixTranspose(matrix);
        }
    }

    private Matrix<double> MatrixAddGpuDouble(Matrix<double> a, Matrix<double> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rows != b.Rows || a.Columns != b.Columns)
        {
            throw new ArgumentException($"Matrix dimensions must match for addition.");
        }

        try
        {
            var result = new Matrix<double>(a.Rows, a.Columns);
            int rows = a.Rows, cols = a.Columns;

            var gpuA = _memoryPoolDouble!.Rent(rows * cols);
            var gpuB = _memoryPoolDouble!.Rent(rows * cols);
            var gpuResult = _memoryPoolDouble!.Rent(rows * cols);

            try
            {
                gpuA.CopyFromCPU(a.AsSpan());
                gpuB.CopyFromCPU(b.AsSpan());

                var viewA = gpuA.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));
                var viewB = gpuB.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));
                var viewResult = gpuResult.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));

                _matrixAddKernelDouble!(new Index2D(rows, cols), viewA, viewB, viewResult);
                _accelerator!.Synchronize();

                gpuResult.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuA);
                _memoryPoolDouble.Return(gpuB);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix add (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixAdd(a, b);
        }
    }

    private Matrix<double> MatrixMultiplyScalarGpuDouble(Matrix<double> matrix, double scalar)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        try
        {
            var result = new Matrix<double>(matrix.Rows, matrix.Columns);
            int rows = matrix.Rows, cols = matrix.Columns;

            var gpuMatrix = _memoryPoolDouble!.Rent(rows * cols);
            var gpuResult = _memoryPoolDouble!.Rent(rows * cols);

            try
            {
                gpuMatrix.CopyFromCPU(matrix.AsSpan());

                var viewMatrix = gpuMatrix.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));
                var viewResult = gpuResult.View.As2DDenseX<Stride2D.DenseX>(new Index2D(rows, cols));

                _matrixMultiplyScalarKernelDouble!(new Index2D(rows, cols), viewMatrix, scalar, viewResult);
                _accelerator!.Synchronize();

                gpuResult.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuMatrix);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix scalar multiply (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixMultiplyScalar(matrix, scalar);
        }
    }

    #endregion

    /// <summary>
    /// Disposes GPU resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;

        // Dispose memory pools (Phase B: US-GPU-002, US-GPU-005)
        _memoryPoolFloat?.Dispose();
        _memoryPoolDouble?.Dispose();
        _memoryPoolInt?.Dispose();
        _memoryPoolLong?.Dispose();

        _accelerator?.Dispose();
        _context?.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
