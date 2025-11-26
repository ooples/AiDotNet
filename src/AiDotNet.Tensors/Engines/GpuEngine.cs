#if !NET462
using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Delegate for Conv2D GPU kernel with float precision (18 parameters exceeds Action limit).
/// </summary>
internal delegate void Conv2DKernelFloat(AcceleratorStream stream, Index1D index, ArrayView<float> input, ArrayView<float> kernel, ArrayView<float> output,
    int batch, int inChannels, int height, int width, int outChannels,
    int outputHeight, int outputWidth, int kernelHeight, int kernelWidth, int stride, int padding, int dilation);

/// <summary>
/// Delegate for Conv2D GPU kernel with double precision (18 parameters exceeds Action limit).
/// </summary>
internal delegate void Conv2DKernelDouble(AcceleratorStream stream, Index1D index, ArrayView<double> input, ArrayView<double> kernel, ArrayView<double> output,
    int batch, int inChannels, int height, int width, int outChannels,
    int outputHeight, int outputWidth, int kernelHeight, int kernelWidth, int stride, int padding, int dilation);

/// <summary>
/// Parameter struct for Conv2D kernel (groups 12 scalar parameters to simplify kernel signature).
/// </summary>
internal readonly struct Conv2DParams
{
    public readonly int Batch;
    public readonly int InChannels;
    public readonly int Height;
    public readonly int Width;
    public readonly int OutChannels;
    public readonly int OutputHeight;
    public readonly int OutputWidth;
    public readonly int KernelHeight;
    public readonly int KernelWidth;
    public readonly int Stride;
    public readonly int Padding;
    public readonly int Dilation;

    public Conv2DParams(int batch, int inChannels, int height, int width, int outChannels,
        int outputHeight, int outputWidth, int kernelHeight, int kernelWidth,
        int stride, int padding, int dilation)
    {
        Batch = batch;
        InChannels = inChannels;
        Height = height;
        Width = width;
        OutChannels = outChannels;
        OutputHeight = outputHeight;
        OutputWidth = outputWidth;
        KernelHeight = kernelHeight;
        KernelWidth = kernelWidth;
        Stride = stride;
        Padding = padding;
        Dilation = dilation;
    }
}

/// <summary>
/// Static helper class for Conv2D kernel methods (required for explicit compilation).
/// </summary>
internal static class Conv2DKernels
{
    /// <summary>
    /// Conv2D kernel implementation for float precision.
    /// </summary>
    public static void Conv2DKernelFloatImpl(Index1D index, ArrayView<float> input, ArrayView<float> kernel, ArrayView<float> output,
        Conv2DParams parameters)
    {
        // Convert flat index to 4D coordinates
        int ow = (int)index % parameters.OutputWidth;
        int temp = (int)index / parameters.OutputWidth;
        int oh = temp % parameters.OutputHeight;
        temp /= parameters.OutputHeight;
        int oc = temp % parameters.OutChannels;
        int b = temp / parameters.OutChannels;

        float sum = 0;

        // Sum over all input channels
        for (int ic = 0; ic < parameters.InChannels; ic++)
        {
            // Sum over kernel window
            for (int kh = 0; kh < parameters.KernelHeight; kh++)
            {
                for (int kw = 0; kw < parameters.KernelWidth; kw++)
                {
                    int ih = oh * parameters.Stride + kh * parameters.Dilation - parameters.Padding;
                    int iw = ow * parameters.Stride + kw * parameters.Dilation - parameters.Padding;

                    if (ih >= 0 && ih < parameters.Height && iw >= 0 && iw < parameters.Width)
                    {
                        int inputIdx = ((b * parameters.InChannels + ic) * parameters.Height + ih) * parameters.Width + iw;
                        int kernelIdx = ((oc * parameters.InChannels + ic) * parameters.KernelHeight + kh) * parameters.KernelWidth + kw;
                        sum += input[inputIdx] * kernel[kernelIdx];
                    }
                }
            }
        }

        output[index] = sum;
    }

    /// <summary>
    /// Conv2D kernel implementation for double precision.
    /// </summary>
    public static void Conv2DKernelDoubleImpl(Index1D index, ArrayView<double> input, ArrayView<double> kernel, ArrayView<double> output,
        Conv2DParams parameters)
    {
        // Convert flat index to 4D coordinates
        int ow = (int)index % parameters.OutputWidth;
        int temp = (int)index / parameters.OutputWidth;
        int oh = temp % parameters.OutputHeight;
        temp /= parameters.OutputHeight;
        int oc = temp % parameters.OutChannels;
        int b = temp / parameters.OutChannels;

        double sum = 0;

        // Sum over all input channels
        for (int ic = 0; ic < parameters.InChannels; ic++)
        {
            // Sum over kernel window
            for (int kh = 0; kh < parameters.KernelHeight; kh++)
            {
                for (int kw = 0; kw < parameters.KernelWidth; kw++)
                {
                    int ih = oh * parameters.Stride + kh * parameters.Dilation - parameters.Padding;
                    int iw = ow * parameters.Stride + kw * parameters.Dilation - parameters.Padding;

                    if (ih >= 0 && ih < parameters.Height && iw >= 0 && iw < parameters.Width)
                    {
                        int inputIdx = ((b * parameters.InChannels + ic) * parameters.Height + ih) * parameters.Width + iw;
                        int kernelIdx = ((oc * parameters.InChannels + ic) * parameters.KernelHeight + kh) * parameters.KernelWidth + kw;
                        sum += input[inputIdx] * kernel[kernelIdx];
                    }
                }
            }
        }

        output[index] = sum;
    }
}

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
/// <para><b>Thread Safety (Phase B: US-GPU-019):</b>
///
/// GpuEngine is fully thread-safe for concurrent operations:
/// - Multiple threads can call operations simultaneously
/// - Kernel execution is synchronized internally
/// - GPU health tracking uses atomic operations
/// - Memory pools are thread-safe (ConcurrentBag-based)
///
/// Performance notes:
/// - Concurrent small operations may serialize due to synchronization overhead
/// - Large operations (> 100K elements) benefit from parallelism
/// - Consider using separate GpuEngine instances for independent workloads
/// </para>
/// </remarks>
public class GpuEngine : IEngine, IDisposable
{
    private readonly Context? _context;
    private readonly Accelerator? _accelerator;
    private readonly CpuEngine _cpuFallback;
    private readonly AdaptiveThresholds _thresholds;
    private bool _disposed;

    // Thread-safe GPU health tracking (Phase B: US-GPU-019, US-GPU-020)
    // Volatile ensures visibility across threads without full locking
    private volatile bool _gpuHealthy = true;

    // GPU recovery tracking (Phase B: US-GPU-020)
    private volatile int _consecutiveFailures = 0;
    private long _lastFailureTimeTicks = DateTime.MinValue.Ticks;
    private const int MaxRecoveryAttempts = 3;
    private static readonly TimeSpan RecoveryBackoffPeriod = TimeSpan.FromSeconds(30);

    // Synchronization lock for GPU operations (Phase B: US-GPU-019)
    // ILGPU accelerator is not thread-safe, so we serialize kernel launches
    private readonly object _gpuLock = new object();

    // Lock for GPU recovery operations (Phase B: US-GPU-020)
    private readonly object _recoveryLock = new object();

    // Memory pools (Phase B: US-GPU-002, US-GPU-005)
    private readonly GpuMemoryPool<float>? _memoryPoolFloat;
    private readonly GpuMemoryPool<double>? _memoryPoolDouble;
    private readonly GpuMemoryPool<int>? _memoryPoolInt;
    private readonly GpuMemoryPool<long>? _memoryPoolLong;

    // Kernel cache for float operations (Phase B: US-GPU-001)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _addKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _subtractKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _multiplyKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, float, ArrayView<float>>? _multiplyScalarKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _divideKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, float, ArrayView<float>>? _divideScalarKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _sqrtKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, float, ArrayView<float>>? _powerKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _maxKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _minKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _absKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _expKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _logKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _signKernelFloat;

    // Activation function kernels (Phase B: US-GPU-004 - GPU Acceleration)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _tanhKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _sigmoidKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _reluKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _geluKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _mishKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _swishKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, float, ArrayView<float>>? _eluKernelFloat;

    // Trigonometric function kernels (Phase SIMD)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _sinKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _cosKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _tanKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _sinKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _cosKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _tanKernelDouble;

    // Hyperbolic function kernels (Phase SIMD)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _sinhKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _coshKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _sinhKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _coshKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _tanhKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _sigmoidKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _reluKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _geluKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _mishKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _swishKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, double, ArrayView<double>>? _eluKernelDouble;

    // Kernel cache for double operations (Phase B: US-GPU-005)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _addKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _subtractKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _multiplyKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, double, ArrayView<double>>? _multiplyScalarKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _divideKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, double, ArrayView<double>>? _divideScalarKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _sqrtKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, double, ArrayView<double>>? _powerKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _maxKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _minKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _absKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _expKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _logKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _signKernelDouble;

    // Kernel cache for int operations (Phase B: US-GPU-005)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>? _addKernelInt;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>? _subtractKernelInt;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>? _multiplyKernelInt;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<int>, int, ArrayView<int>>? _multiplyScalarKernelInt;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>? _divideKernelInt;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<int>, int, ArrayView<int>>? _divideScalarKernelInt;

    // Kernel cache for long operations (Phase B: US-GPU-005)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>? _addKernelLong;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>? _subtractKernelLong;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>? _multiplyKernelLong;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<long>, long, ArrayView<long>>? _multiplyScalarKernelLong;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>? _divideKernelLong;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<long>, long, ArrayView<long>>? _divideScalarKernelLong;

    // Kernel cache for matrix operations - float (Phase B: Epic 2)
    private readonly Action<AcceleratorStream, Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int>? _matrixMultiplyKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>, int, int>? _matrixVectorMultiplyKernelFloat;
    private readonly Action<AcceleratorStream, Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>? _matrixTransposeKernelFloat;
    private readonly Action<AcceleratorStream, Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>? _matrixAddKernelFloat;
    private readonly Action<AcceleratorStream, Index2D, ArrayView2D<float, Stride2D.DenseX>, float, ArrayView2D<float, Stride2D.DenseX>>? _matrixMultiplyScalarKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _swapRowsKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, int, int>? _swapColumnsKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, int, int>? _getColumnKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, int, int>? _setColumnKernelFloat;
    private readonly Action<AcceleratorStream, Index2D, ArrayView<float>, ArrayView<float>, ArrayView2D<float, Stride2D.DenseX>, int, int>? _outerProductKernelFloat;

    // Kernel cache for matrix operations - double (Phase B: Epic 2)
    private readonly Action<AcceleratorStream, Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, int>? _matrixMultiplyKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, ArrayView<double>, int, int>? _matrixVectorMultiplyKernelDouble;
    private readonly Action<AcceleratorStream, Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>>? _matrixTransposeKernelDouble;
    private readonly Action<AcceleratorStream, Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>>? _matrixAddKernelDouble;
    private readonly Action<AcceleratorStream, Index2D, ArrayView2D<double, Stride2D.DenseX>, double, ArrayView2D<double, Stride2D.DenseX>>? _matrixMultiplyScalarKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _swapRowsKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, int, int>? _swapColumnsKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, int, int>? _getColumnKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, int, int>? _setColumnKernelDouble;
    private readonly Action<AcceleratorStream, Index2D, ArrayView<double>, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>, int, int>? _outerProductKernelDouble;

    // Kernel cache for tensor operations - float (Phase B: Epic 3)
    private readonly Action<AcceleratorStream, Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>? _batchMatMulKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _tensorAddKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _tensorSubtractKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _tensorMultiplyKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, float, ArrayView<float>>? _tensorMultiplyScalarKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _tensorDivideKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int, int, int>? _maxPool2DKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int, int, int>? _avgPool2DKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, Conv2DParams>? _conv2DKernelFloat;

    // Kernel cache for tensor operations - double (Phase B: Epic 3)
    private readonly Action<AcceleratorStream, Index3D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int>? _batchMatMulKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _tensorAddKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _tensorSubtractKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _tensorMultiplyKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, double, ArrayView<double>>? _tensorMultiplyScalarKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _tensorDivideKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, int, int, int, int, int, int, int, int, int>? _maxPool2DKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, int, int, int, int, int, int, int, int, int>? _avgPool2DKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, Conv2DParams>? _conv2DKernelDouble;

    // Production GPU kernels - Mathematical functions (Phase C: Production Ready)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _log2KernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _log2KernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _exp2KernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _exp2KernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _exp10KernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _exp10KernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _expM1KernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _expM1KernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _log1PKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _log1PKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _negateKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _negateKernelDouble;

    // Production GPU kernels - Utility functions (Phase C: Production Ready)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, float, float, ArrayView<float>>? _clampKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, double, double, ArrayView<double>>? _clampKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, float, ArrayView<float>>? _lerpKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, double, ArrayView<double>>? _lerpKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _reciprocalKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _reciprocalKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _rsqrtKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _rsqrtKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _minMagnitudeKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _minMagnitudeKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>? _maxMagnitudeKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _maxMagnitudeKernelDouble;

    // Production GPU kernels - Rounding operations (Phase C: Production Ready)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _roundKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _roundKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _floorKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _floorKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _ceilingKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _ceilingKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>? _truncateKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>>? _truncateKernelDouble;

    // Production GPU kernels - Fill operations (Phase C: Production Ready)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, float>? _fillKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, double>? _fillKernelDouble;

    // Production GPU kernels - Reduction partial sums (Phase C: Production Ready)
    // Block size for reduction kernels
    private const int ReductionBlockSize = 256;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int>? _partialSumKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, int>? _partialSumKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>? _partialDotProductKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int>? _partialDotProductKernelDouble;

    // Production GPU kernels - Vector softmax (Phase C: Production Ready)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, float, float>? _softmaxKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, double, double>? _softmaxKernelDouble;

    // Production GPU kernels - Extended Tensor Operations (Phase D: Full Production)
    // TensorMatMul - 2D tensor matrix multiplication
    private readonly Action<AcceleratorStream, Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>? _tensorMatMulKernelFloat;
    private readonly Action<AcceleratorStream, Index2D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int>? _tensorMatMulKernelDouble;

    // TensorTranspose - 2D tensor transposition
    private readonly Action<AcceleratorStream, Index2D, ArrayView<float>, ArrayView<float>, int, int>? _tensorTransposeKernelFloat;
    private readonly Action<AcceleratorStream, Index2D, ArrayView<double>, ArrayView<double>, int, int>? _tensorTransposeKernelDouble;

    // Tensor Softmax along axis
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int, int, int>? _tensorSoftmaxKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, int, int, int>? _tensorSoftmaxKernelDouble;

    // BatchNorm and LayerNorm forward
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int, int>? _batchNormKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, double, int, int>? _batchNormKernelDouble;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int, int>? _layerNormKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>, double, int, int>? _layerNormKernelDouble;

    // Upsample (nearest neighbor)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int>? _upsampleKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, int, int, int, int, int, int>? _upsampleKernelDouble;

    // PixelShuffle (depth-to-space)
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int, int>? _pixelShuffleKernelFloat;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, int, int, int, int, int>? _pixelShuffleKernelDouble;

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

                _addKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);

                _subtractKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = a[index] - b[index]);

                _multiplyKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = a[index] * b[index]);

                _multiplyScalarKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, float, ArrayView<float>>(
                    (index, vec, scalar, result) => result[index] = vec[index] * scalar);

                _divideKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = a[index] / b[index]);

                _divideScalarKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, float, ArrayView<float>>(
                    (index, vec, scalar, result) => result[index] = vec[index] / scalar);

                _sqrtKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, vec, result) => result[index] = XMath.Sqrt(vec[index]));

                _powerKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, float, ArrayView<float>>(
                    (index, vec, exp, result) => result[index] = XMath.Pow(vec[index], exp));

                _maxKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = XMath.Max(a[index], b[index]));

                _minKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = XMath.Min(a[index], b[index]));

                _absKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, vec, result) => result[index] = XMath.Abs(vec[index]));

                _expKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, vec, result) => result[index] = XMath.Exp(vec[index]));

                _logKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, vec, result) => result[index] = XMath.Log(vec[index]));

                _signKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, vec, result) => result[index] = vec[index] > 0 ? 1.0f : (vec[index] < 0 ? -1.0f : 0.0f));

                // Activation function kernels (Phase B: US-GPU-004)
                _tanhKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Tanh(input[index]));

                _sigmoidKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = 1.0f / (1.0f + XMath.Exp(-input[index])));

                _reluKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Max(0.0f, input[index]));

                _geluKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => {
                        float x = input[index];
                        float sqrt2OverPi = 0.7978845608028654f;
                        float x_cubed = x * x * x;
                        float inner = x + 0.044715f * x_cubed;
                        float tanh_arg = sqrt2OverPi * inner;
                        float tanh_val = XMath.Tanh(tanh_arg);
                        result[index] = 0.5f * x * (1.0f + tanh_val);
                    });

                _mishKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => {
                        float x = input[index];
                        float softplus = XMath.Log(1.0f + XMath.Exp(x));
                        result[index] = x * XMath.Tanh(softplus);
                    });

                _swishKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => {
                        float x = input[index];
                        result[index] = x / (1.0f + XMath.Exp(-x));
                    });

                _eluKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, float, ArrayView<float>>(
                    (index, input, alpha, result) => {
                        float x = input[index];
                        result[index] = x > 0.0f ? x : alpha * (XMath.Exp(x) - 1.0f);
                    });

                // Trigonometric function kernels (Phase SIMD)
                _sinKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Sin(input[index]));

                _cosKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Cos(input[index]));

                _tanKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Tan(input[index]));

                // Hyperbolic function kernels (Phase SIMD)
                _sinhKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Sinh(input[index]));

                _coshKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Cosh(input[index]));

                // Exponential function kernels (Phase SIMD)
                _expKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Exp(input[index]));

                _logKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Log(input[index]));

                Console.WriteLine("[GpuEngine] Float kernels pre-compiled");

                // Double activation function kernels
                _sigmoidKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => {
                        double x = input[index];
                        result[index] = 1.0 / (1.0 + XMath.Exp(-x));
                    });

                _reluKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => {
                        result[index] = XMath.Max(0.0, input[index]);
                    });

                _geluKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => {
                        double x = input[index];
                        result[index] = 0.5 * x * (1.0 + XMath.Tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
                    });

                _mishKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => {
                        double x = input[index];
                        double softplus = XMath.Log(1.0 + XMath.Exp(x));
                        result[index] = x * XMath.Tanh(softplus);
                    });

                _swishKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => {
                        double x = input[index];
                        result[index] = x / (1.0 + XMath.Exp(-x));
                    });

                _eluKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, double, ArrayView<double>>(
                    (index, input, alpha, result) => {
                        double x = input[index];
                        result[index] = x > 0.0 ? x : alpha * (XMath.Exp(x) - 1.0);
                    });

                Console.WriteLine("[GpuEngine] Double activation kernels pre-compiled");

                // Pre-compile kernels for double operations (Phase B: US-GPU-005)
                _addKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);
                _subtractKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = a[index] - b[index]);
                _multiplyKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = a[index] * b[index]);
                _multiplyScalarKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, double, ArrayView<double>>(
                    (index, vec, scalar, result) => result[index] = vec[index] * scalar);
                _divideKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = a[index] / b[index]);
                _divideScalarKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, double, ArrayView<double>>(
                    (index, vec, scalar, result) => result[index] = vec[index] / scalar);
                _sqrtKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, vec, result) => result[index] = XMath.Sqrt(vec[index]));
                _powerKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, double, ArrayView<double>>(
                    (index, vec, exp, result) => result[index] = XMath.Pow(vec[index], exp));

                _maxKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = XMath.Max(a[index], b[index]));

                _minKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = XMath.Min(a[index], b[index]));

                _absKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, vec, result) => result[index] = XMath.Abs(vec[index]));

                _expKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, vec, result) => result[index] = XMath.Exp(vec[index]));

                _logKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, vec, result) => result[index] = XMath.Log(vec[index]));

                _signKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, vec, result) => result[index] = vec[index] > 0 ? 1.0 : (vec[index] < 0 ? -1.0 : 0.0));

                // Trigonometric function kernels (Phase SIMD)
                _sinKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Sin(input[index]));

                _cosKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Cos(input[index]));

                _tanKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Tan(input[index]));

                // Hyperbolic function kernels (Phase SIMD)
                _sinhKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Sinh(input[index]));

                _coshKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Cosh(input[index]));

                _tanhKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Tanh(input[index]));

                Console.WriteLine("[GpuEngine] Double kernels pre-compiled");

                // Pre-compile kernels for int operations (Phase B: US-GPU-005)
                _addKernelInt = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);
                _subtractKernelInt = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                    (index, a, b, result) => result[index] = a[index] - b[index]);
                _multiplyKernelInt = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                    (index, a, b, result) => result[index] = a[index] * b[index]);
                _multiplyScalarKernelInt = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<int>, int, ArrayView<int>>(
                    (index, vec, scalar, result) => result[index] = vec[index] * scalar);
                _divideKernelInt = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                    (index, a, b, result) => result[index] = a[index] / b[index]);
                _divideScalarKernelInt = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<int>, int, ArrayView<int>>(
                    (index, vec, scalar, result) => result[index] = vec[index] / scalar);
                Console.WriteLine("[GpuEngine] Int kernels pre-compiled");

                // Pre-compile kernels for long operations (Phase B: US-GPU-005)
                _addKernelLong = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);
                _subtractKernelLong = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>(
                    (index, a, b, result) => result[index] = a[index] - b[index]);
                _multiplyKernelLong = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>(
                    (index, a, b, result) => result[index] = a[index] * b[index]);
                _multiplyScalarKernelLong = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<long>, long, ArrayView<long>>(
                    (index, vec, scalar, result) => result[index] = vec[index] * scalar);
                _divideKernelLong = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>(
                    (index, a, b, result) => result[index] = a[index] / b[index]);
                _divideScalarKernelLong = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<long>, long, ArrayView<long>>(
                    (index, vec, scalar, result) => result[index] = vec[index] / scalar);
                Console.WriteLine("[GpuEngine] Long kernels pre-compiled");

                // Pre-compile kernels for matrix operations - float (Phase B: Epic 2)
                _matrixMultiplyKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int>(
                    (index, a, b, result, k) =>
                    {
                        float sum = 0;
                        for (int i = 0; i < k; i++)
                            sum += a[index.X, i] * b[i, index.Y];
                        result[index] = sum;
                    });

                _matrixVectorMultiplyKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, ArrayView<float>, int, int>(
                    (index, matrix, vector, result, rows, cols) =>
                    {
                        float sum = 0;
                        for (int j = 0; j < cols; j++)
                            sum += matrix[index, j] * vector[j];
                        result[index] = sum;
                    });

                _matrixTransposeKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(
                    (index, input, output) => output[index.Y, index.X] = input[index]);

                _matrixAddKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);

                _matrixMultiplyScalarKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView2D<float, Stride2D.DenseX>, float, ArrayView2D<float, Stride2D.DenseX>>(
                    (index, matrix, scalar, result) => result[index] = matrix[index] * scalar);

                // Swap rows kernel (Phase B: Matrix operations)
                _swapRowsKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, row1, row2) => {
                        float temp = row1[index];
                        row1[index] = row2[index];
                        row2[index] = temp;
                    });

                // Swap columns kernel (Phase B: Matrix operations)
                _swapColumnsKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, int, int>(
                    (index, matrix, tempCol, col1, col2) => {
                        // Each thread handles one row
                        float temp = matrix[index, col1];
                        matrix[index, col1] = matrix[index, col2];
                        matrix[index, col2] = temp;
                    });

                // Get column kernel (Phase B: Matrix operations)
                _getColumnKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, int, int>(
                    (index, matrix, result, col, rows) => {
                        result[index] = matrix[index, col];
                    });

                // Set column kernel (Phase B: Matrix operations)
                _setColumnKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView<float>, int, int>(
                    (index, matrix, values, col, rows) => {
                        matrix[index, col] = values[index];
                    });

                // Outer product kernel (Phase B: Matrix operations)
                _outerProductKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView<float>, ArrayView<float>, ArrayView2D<float, Stride2D.DenseX>, int, int>(
                    (index, a, b, result, aLen, bLen) => {
                        result[index] = a[index.X] * b[index.Y];
                    });
                Console.WriteLine("[GpuEngine] Float matrix kernels pre-compiled");

                // Pre-compile kernels for matrix operations - double (Phase B: Epic 2)
                _matrixMultiplyKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, int>(
                    (index, a, b, result, k) =>
                    {
                        double sum = 0;
                        for (int i = 0; i < k; i++)
                            sum += a[index.X, i] * b[i, index.Y];
                        result[index] = sum;
                    });

                _matrixVectorMultiplyKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, ArrayView<double>, int, int>(
                    (index, matrix, vector, result, rows, cols) =>
                    {
                        double sum = 0;
                        for (int j = 0; j < cols; j++)
                            sum += matrix[index, j] * vector[j];
                        result[index] = sum;
                    });

                _matrixTransposeKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>>(
                    (index, input, output) => output[index.Y, index.X] = input[index]);

                _matrixAddKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);

                _matrixMultiplyScalarKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView2D<double, Stride2D.DenseX>, double, ArrayView2D<double, Stride2D.DenseX>>(
                    (index, matrix, scalar, result) => result[index] = matrix[index] * scalar);

                // Swap rows kernel (Phase B: Matrix operations)
                _swapRowsKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, row1, row2) => {
                        double temp = row1[index];
                        row1[index] = row2[index];
                        row2[index] = temp;
                    });

                // Swap columns kernel (Phase B: Matrix operations)
                _swapColumnsKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, int, int>(
                    (index, matrix, tempCol, col1, col2) => {
                        // Each thread handles one row
                        double temp = matrix[index, col1];
                        matrix[index, col1] = matrix[index, col2];
                        matrix[index, col2] = temp;
                    });

                // Get column kernel (Phase B: Matrix operations)
                _getColumnKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, int, int>(
                    (index, matrix, result, col, rows) => {
                        result[index] = matrix[index, col];
                    });

                // Set column kernel (Phase B: Matrix operations)
                _setColumnKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, int, int>(
                    (index, matrix, values, col, rows) => {
                        matrix[index, col] = values[index];
                    });

                // Outer product kernel (Phase B: Matrix operations)
                _outerProductKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView<double>, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>, int, int>(
                    (index, a, b, result, aLen, bLen) => {
                        result[index] = a[index.X] * b[index.Y];
                    });
                Console.WriteLine("[GpuEngine] Double matrix kernels pre-compiled");

                // Pre-compile kernels for tensor operations - float (Phase B: Epic 3)
                _batchMatMulKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index3D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
                    (index, a, b, result, m, k, n) =>
                    {
                        int batch = index.X;
                        int i = index.Y;
                        int j = index.Z;

                        // Compute flat indices for 3D tensors stored in row-major order
                        // Tensor shape: [batchSize, rows, cols]
                        // Flat index: batch * (rows * cols) + row * cols + col
                        float sum = 0;
                        for (int p = 0; p < k; p++)
                        {
                            int aIndex = batch * (m * k) + i * k + p;
                            int bIndex = batch * (k * n) + p * n + j;
                            sum += a[aIndex] * b[bIndex];
                        }

                        int resultIndex = batch * (m * n) + i * n + j;
                        result[resultIndex] = sum;
                    });

                // Pre-compile kernels for tensor operations - double (Phase B: Epic 3)
                _batchMatMulKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index3D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int>(
                    (index, a, b, result, m, k, n) =>
                    {
                        int batch = index.X;
                        int i = index.Y;
                        int j = index.Z;

                        // Compute flat indices for 3D tensors stored in row-major order
                        double sum = 0;
                        for (int p = 0; p < k; p++)
                        {
                            int aIndex = batch * (m * k) + i * k + p;
                            int bIndex = batch * (k * n) + p * n + j;
                            sum += a[aIndex] * b[bIndex];
                        }

                        int resultIndex = batch * (m * n) + i * n + j;
                        result[resultIndex] = sum;
                    });

                // Pre-compile tensor element-wise kernels - float (Phase B: Epic 3, US-GPU-014)
                _tensorAddKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);

                _tensorSubtractKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = a[index] - b[index]);

                _tensorMultiplyKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = a[index] * b[index]);

                _tensorMultiplyScalarKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, float, ArrayView<float>>(
                    (index, tensor, scalar, result) => result[index] = tensor[index] * scalar);

                _tensorDivideKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => result[index] = a[index] / b[index]);

                // Pre-compile tensor element-wise kernels - double (Phase B: Epic 3, US-GPU-014)
                _tensorAddKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = a[index] + b[index]);

                _tensorSubtractKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = a[index] - b[index]);

                _tensorMultiplyKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = a[index] * b[index]);

                _tensorMultiplyScalarKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, double, ArrayView<double>>(
                    (index, tensor, scalar, result) => result[index] = tensor[index] * scalar);

                _tensorDivideKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => result[index] = a[index] / b[index]);

                // Pre-compile pooling kernels - float (Phase B: Epic 3, US-GPU-012)
                _maxPool2DKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int, int, int>(
                    (index, input, output, batch, channels, height, width, outputHeight, outputWidth, poolSize, stride, padding) =>
                    {
                        // Convert flat index to 4D coordinates
                        int ow = (int)index % outputWidth;
                        int temp = (int)index / outputWidth;
                        int oh = temp % outputHeight;
                        temp /= outputHeight;
                        int c = temp % channels;
                        int b = temp / channels;

                        float maxVal = float.NegativeInfinity;

                        for (int kh = 0; kh < poolSize; kh++)
                        {
                            for (int kw = 0; kw < poolSize; kw++)
                            {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int inputIdx = ((b * channels + c) * height + ih) * width + iw;
                                    float val = input[inputIdx];
                                    if (val > maxVal) maxVal = val;
                                }
                            }
                        }

                        output[index] = maxVal;
                    });

                _avgPool2DKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int, int, int>(
                    (index, input, output, batch, channels, height, width, outputHeight, outputWidth, poolSize, stride, padding) =>
                    {
                        // Convert flat index to 4D coordinates
                        int ow = (int)index % outputWidth;
                        int temp = (int)index / outputWidth;
                        int oh = temp % outputHeight;
                        temp /= outputHeight;
                        int c = temp % channels;
                        int b = temp / channels;

                        float sum = 0;
                        int count = 0;

                        for (int kh = 0; kh < poolSize; kh++)
                        {
                            for (int kw = 0; kw < poolSize; kw++)
                            {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int inputIdx = ((b * channels + c) * height + ih) * width + iw;
                                    sum += input[inputIdx];
                                    count++;
                                }
                            }
                        }

                        output[index] = count > 0 ? sum / count : 0;
                    });

                // Pre-compile pooling kernels - double (Phase B: Epic 3, US-GPU-012)
                _maxPool2DKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, int, int, int, int, int, int, int, int, int>(
                    (index, input, output, batch, channels, height, width, outputHeight, outputWidth, poolSize, stride, padding) =>
                    {
                        // Convert flat index to 4D coordinates
                        int ow = (int)index % outputWidth;
                        int temp = (int)index / outputWidth;
                        int oh = temp % outputHeight;
                        temp /= outputHeight;
                        int c = temp % channels;
                        int b = temp / channels;

                        double maxVal = double.NegativeInfinity;

                        for (int kh = 0; kh < poolSize; kh++)
                        {
                            for (int kw = 0; kw < poolSize; kw++)
                            {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int inputIdx = ((b * channels + c) * height + ih) * width + iw;
                                    double val = input[inputIdx];
                                    if (val > maxVal) maxVal = val;
                                }
                            }
                        }

                        output[index] = maxVal;
                    });

                _avgPool2DKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, int, int, int, int, int, int, int, int, int>(
                    (index, input, output, batch, channels, height, width, outputHeight, outputWidth, poolSize, stride, padding) =>
                    {
                        // Convert flat index to 4D coordinates
                        int ow = (int)index % outputWidth;
                        int temp = (int)index / outputWidth;
                        int oh = temp % outputHeight;
                        temp /= outputHeight;
                        int c = temp % channels;
                        int b = temp / channels;

                        double sum = 0;
                        int count = 0;

                        for (int kh = 0; kh < poolSize; kh++)
                        {
                            for (int kw = 0; kw < poolSize; kw++)
                            {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int inputIdx = ((b * channels + c) * height + ih) * width + iw;
                                    sum += input[inputIdx];
                                    count++;
                                }
                            }
                        }

                        output[index] = count > 0 ? sum / count : 0;
                    });

                // Pre-compile Conv2D kernels - float (Phase B: Epic 3, US-GPU-011)
                // Using Conv2DParams struct reduces parameters from 16 to 5 (under Action<> limit)
                _conv2DKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, Conv2DParams>(
                    Conv2DKernels.Conv2DKernelFloatImpl);

                // Pre-compile Conv2D kernels - double (Phase B: Epic 3, US-GPU-011)
                // Using Conv2DParams struct reduces parameters from 16 to 5 (under Action<> limit)
                _conv2DKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, Conv2DParams>(
                    Conv2DKernels.Conv2DKernelDoubleImpl);

                Console.WriteLine("[GpuEngine] Tensor kernels pre-compiled");

                // Pre-compile production GPU kernels - Mathematical functions (Phase C: Production Ready)
                _log2KernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Log2(input[index]));
                _log2KernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Log2(input[index]));
                _exp2KernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Exp2(input[index]));
                _exp2KernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Exp2(input[index]));
                _exp10KernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Pow(10.0f, input[index]));
                _exp10KernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Pow(10.0, input[index]));
                _expM1KernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Exp(input[index]) - 1.0f);
                _expM1KernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Exp(input[index]) - 1.0);
                _log1PKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Log(1.0f + input[index]));
                _log1PKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Log(1.0 + input[index]));
                _negateKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = -input[index]);
                _negateKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = -input[index]);
                Console.WriteLine("[GpuEngine] Mathematical kernels pre-compiled");

                // Pre-compile production GPU kernels - Utility functions (Phase C: Production Ready)
                _clampKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, float, float, ArrayView<float>>(
                    (index, input, min, max, result) => result[index] = XMath.Clamp(input[index], min, max));
                _clampKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, double, double, ArrayView<double>>(
                    (index, input, min, max, result) => result[index] = XMath.Clamp(input[index], min, max));
                _lerpKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, float, ArrayView<float>>(
                    (index, a, b, t, result) => result[index] = a[index] + t * (b[index] - a[index]));
                _lerpKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, double, ArrayView<double>>(
                    (index, a, b, t, result) => result[index] = a[index] + t * (b[index] - a[index]));
                _reciprocalKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = 1.0f / input[index]);
                _reciprocalKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = 1.0 / input[index]);
                _rsqrtKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Rsqrt(input[index]));
                _rsqrtKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = 1.0 / XMath.Sqrt(input[index]));
                _minMagnitudeKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => {
                        float absA = XMath.Abs(a[index]);
                        float absB = XMath.Abs(b[index]);
                        result[index] = absA <= absB ? a[index] : b[index];
                    });
                _minMagnitudeKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => {
                        double absA = XMath.Abs(a[index]);
                        double absB = XMath.Abs(b[index]);
                        result[index] = absA <= absB ? a[index] : b[index];
                    });
                _maxMagnitudeKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    (index, a, b, result) => {
                        float absA = XMath.Abs(a[index]);
                        float absB = XMath.Abs(b[index]);
                        result[index] = absA >= absB ? a[index] : b[index];
                    });
                _maxMagnitudeKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(
                    (index, a, b, result) => {
                        double absA = XMath.Abs(a[index]);
                        double absB = XMath.Abs(b[index]);
                        result[index] = absA >= absB ? a[index] : b[index];
                    });
                Console.WriteLine("[GpuEngine] Utility kernels pre-compiled");

                // Pre-compile production GPU kernels - Rounding operations (Phase C: Production Ready)
                _roundKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Round(input[index]));
                _roundKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Round(input[index]));
                _floorKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Floor(input[index]));
                _floorKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Floor(input[index]));
                _ceilingKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Ceiling(input[index]));
                _ceilingKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Ceiling(input[index]));
                _truncateKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>>(
                    (index, input, result) => result[index] = XMath.Truncate(input[index]));
                _truncateKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>>(
                    (index, input, result) => result[index] = XMath.Truncate(input[index]));
                Console.WriteLine("[GpuEngine] Rounding kernels pre-compiled");

                // Pre-compile production GPU kernels - Fill operations (Phase C: Production Ready)
                _fillKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, float>(
                    (index, result, value) => result[index] = value);
                _fillKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, double>(
                    (index, result, value) => result[index] = value);
                Console.WriteLine("[GpuEngine] Fill kernels pre-compiled");

                // Pre-compile production GPU kernels - Reduction partial sums (Phase C: Production Ready)
                // Each thread computes partial sum for a chunk of elements
                _partialSumKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, int>(
                    (blockIdx, input, partialSums, length) => {
                        int startIdx = (int)blockIdx * ReductionBlockSize;
                        float sum = 0.0f;
                        for (int i = 0; i < ReductionBlockSize && startIdx + i < length; i++)
                            sum += input[startIdx + i];
                        partialSums[blockIdx] = sum;
                    });
                _partialSumKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, int>(
                    (blockIdx, input, partialSums, length) => {
                        int startIdx = (int)blockIdx * ReductionBlockSize;
                        double sum = 0.0;
                        for (int i = 0; i < ReductionBlockSize && startIdx + i < length; i++)
                            sum += input[startIdx + i];
                        partialSums[blockIdx] = sum;
                    });
                _partialDotProductKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                    (blockIdx, a, b, partialSums, length) => {
                        int startIdx = (int)blockIdx * ReductionBlockSize;
                        float sum = 0.0f;
                        for (int i = 0; i < ReductionBlockSize && startIdx + i < length; i++)
                            sum += a[startIdx + i] * b[startIdx + i];
                        partialSums[blockIdx] = sum;
                    });
                _partialDotProductKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int>(
                    (blockIdx, a, b, partialSums, length) => {
                        int startIdx = (int)blockIdx * ReductionBlockSize;
                        double sum = 0.0;
                        for (int i = 0; i < ReductionBlockSize && startIdx + i < length; i++)
                            sum += a[startIdx + i] * b[startIdx + i];
                        partialSums[blockIdx] = sum;
                    });
                Console.WriteLine("[GpuEngine] Reduction kernels pre-compiled");

                // Pre-compile production GPU kernels - Vector softmax (Phase C: Production Ready)
                // Softmax kernel takes pre-computed max and sum for numerical stability
                _softmaxKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, float, float>(
                    (index, input, result, maxVal, expSum) => {
                        result[index] = XMath.Exp(input[index] - maxVal) / expSum;
                    });
                _softmaxKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, double, double>(
                    (index, input, result, maxVal, expSum) => {
                        result[index] = XMath.Exp(input[index] - maxVal) / expSum;
                    });
                Console.WriteLine("[GpuEngine] Softmax kernels pre-compiled");

                // Pre-compile production GPU kernels - Extended Tensor Operations (Phase D: Full Production)
                // TensorMatMul - 2D tensor matrix multiplication (reuses matrix multiply logic)
                _tensorMatMulKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                    (index, a, b, result, k) => {
                        int m = index.X;
                        int n = index.Y;
                        float sum = 0;
                        for (int i = 0; i < k; i++)
                            sum += a[m * k + i] * b[i * n + index.Y]; // Use flat array indexing
                        result[index.X * n + index.Y] = sum;
                    });
                _tensorMatMulKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int>(
                    (index, a, b, result, k) => {
                        int m = index.X;
                        int n = index.Y;
                        double sum = 0;
                        for (int i = 0; i < k; i++)
                            sum += a[m * k + i] * b[i * n + index.Y];
                        result[index.X * n + index.Y] = sum;
                    });
                Console.WriteLine("[GpuEngine] TensorMatMul kernels pre-compiled");

                // TensorTranspose - 2D tensor transposition
                _tensorTransposeKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView<float>, ArrayView<float>, int, int>(
                    (index, input, output, rows, cols) => {
                        // input[row, col] -> output[col, row]
                        output[index.Y * rows + index.X] = input[index.X * cols + index.Y];
                    });
                _tensorTransposeKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index2D, ArrayView<double>, ArrayView<double>, int, int>(
                    (index, input, output, rows, cols) => {
                        output[index.Y * rows + index.X] = input[index.X * cols + index.Y];
                    });
                Console.WriteLine("[GpuEngine] TensorTranspose kernels pre-compiled");

                // Upsample (nearest neighbor) - for spatial upsampling in neural networks
                _upsampleKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int>(
                    (flatIdx, input, output, batch, channels, height, width, scaleH, scaleW) => {
                        int newHeight = height * scaleH;
                        int newWidth = width * scaleW;
                        int ow = (int)flatIdx % newWidth;
                        int temp = (int)flatIdx / newWidth;
                        int oh = temp % newHeight;
                        temp /= newHeight;
                        int c = temp % channels;
                        int b = temp / channels;
                        int ih = oh / scaleH;
                        int iw = ow / scaleW;
                        int inputIdx = ((b * channels + c) * height + ih) * width + iw;
                        output[flatIdx] = input[inputIdx];
                    });
                _upsampleKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, int, int, int, int, int, int>(
                    (flatIdx, input, output, batch, channels, height, width, scaleH, scaleW) => {
                        int newHeight = height * scaleH;
                        int newWidth = width * scaleW;
                        int ow = (int)flatIdx % newWidth;
                        int temp = (int)flatIdx / newWidth;
                        int oh = temp % newHeight;
                        temp /= newHeight;
                        int c = temp % channels;
                        int b = temp / channels;
                        int ih = oh / scaleH;
                        int iw = ow / scaleW;
                        int inputIdx = ((b * channels + c) * height + ih) * width + iw;
                        output[flatIdx] = input[inputIdx];
                    });
                Console.WriteLine("[GpuEngine] Upsample kernels pre-compiled");

                // PixelShuffle (depth-to-space) - for super-resolution networks
                _pixelShuffleKernelFloat = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int, int>(
                    (flatIdx, input, output, batch, channels, height, width, upscaleFactor) => {
                        int r = upscaleFactor;
                        int newChannels = channels / (r * r);
                        int newHeight = height * r;
                        int newWidth = width * r;
                        // output index -> input index mapping
                        int ow = (int)flatIdx % newWidth;
                        int temp = (int)flatIdx / newWidth;
                        int oh = temp % newHeight;
                        temp /= newHeight;
                        int oc = temp % newChannels;
                        int b = temp / newChannels;
                        int ih = oh / r;
                        int iw = ow / r;
                        int subH = oh % r;
                        int subW = ow % r;
                        int ic = oc * r * r + subH * r + subW;
                        int inputIdx = ((b * channels + ic) * height + ih) * width + iw;
                        output[flatIdx] = input[inputIdx];
                    });
                _pixelShuffleKernelDouble = _accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<double>, ArrayView<double>, int, int, int, int, int>(
                    (flatIdx, input, output, batch, channels, height, width, upscaleFactor) => {
                        int r = upscaleFactor;
                        int newChannels = channels / (r * r);
                        int newHeight = height * r;
                        int newWidth = width * r;
                        int ow = (int)flatIdx % newWidth;
                        int temp = (int)flatIdx / newWidth;
                        int oh = temp % newHeight;
                        temp /= newHeight;
                        int oc = temp % newChannels;
                        int b = temp / newChannels;
                        int ih = oh / r;
                        int iw = ow / r;
                        int subH = oh % r;
                        int subW = ow % r;
                        int ic = oc * r * r + subH * r + subW;
                        int inputIdx = ((b * channels + ic) * height + ih) * width + iw;
                        output[flatIdx] = input[inputIdx];
                    });
                Console.WriteLine("[GpuEngine] PixelShuffle kernels pre-compiled");

                Console.WriteLine("[GpuEngine] All kernel pre-compilation complete");

                // Initialize memory pools (Phase B: US-GPU-002, US-GPU-005)
                _memoryPoolFloat = new GpuMemoryPool<float>(_accelerator);
                _memoryPoolDouble = new GpuMemoryPool<double>(_accelerator);
                _memoryPoolInt = new GpuMemoryPool<int>(_accelerator);
                _memoryPoolLong = new GpuMemoryPool<long>(_accelerator);
                Console.WriteLine("[GpuEngine] Memory pools initialized");
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or PlatformNotSupportedException or OutOfMemoryException)
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

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)SubtractGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)SubtractGpuDouble((Vector<double>)(object)a, (Vector<double>)(object)b);
            if (typeof(T) == typeof(int))
                return (Vector<T>)(object)SubtractGpuInt((Vector<int>)(object)a, (Vector<int>)(object)b);
            if (typeof(T) == typeof(long))
                return (Vector<T>)(object)SubtractGpuLong((Vector<long>)(object)a, (Vector<long>)(object)b);
        }

        return _cpuFallback.Subtract(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Multiply<T>(Vector<T> a, Vector<T> b)
    {
        if (a.Length < _thresholds.VectorMultiply)
            return _cpuFallback.Multiply(a, b);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)MultiplyGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)MultiplyGpuDouble((Vector<double>)(object)a, (Vector<double>)(object)b);
            if (typeof(T) == typeof(int))
                return (Vector<T>)(object)MultiplyGpuInt((Vector<int>)(object)a, (Vector<int>)(object)b);
            if (typeof(T) == typeof(long))
                return (Vector<T>)(object)MultiplyGpuLong((Vector<long>)(object)a, (Vector<long>)(object)b);
        }

        return _cpuFallback.Multiply(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Multiply<T>(Vector<T> vector, T scalar)
    {
        if (vector.Length < _thresholds.VectorMultiply)
            return _cpuFallback.Multiply(vector, scalar);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)MultiplyScalarGpu((Vector<float>)(object)vector, (float)(object)scalar!);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)MultiplyScalarGpuDouble((Vector<double>)(object)vector, (double)(object)scalar!);
            if (typeof(T) == typeof(int))
                return (Vector<T>)(object)MultiplyScalarGpuInt((Vector<int>)(object)vector, (int)(object)scalar!);
            if (typeof(T) == typeof(long))
                return (Vector<T>)(object)MultiplyScalarGpuLong((Vector<long>)(object)vector, (long)(object)scalar!);
        }

        return _cpuFallback.Multiply(vector, scalar);
    }

    /// <inheritdoc/>
    public Vector<T> Divide<T>(Vector<T> a, Vector<T> b)
    {
        if (a.Length < _thresholds.VectorDivide)
            return _cpuFallback.Divide(a, b);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)DivideGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)DivideGpuDouble((Vector<double>)(object)a, (Vector<double>)(object)b);
            if (typeof(T) == typeof(int))
                return (Vector<T>)(object)DivideGpuInt((Vector<int>)(object)a, (Vector<int>)(object)b);
            if (typeof(T) == typeof(long))
                return (Vector<T>)(object)DivideGpuLong((Vector<long>)(object)a, (Vector<long>)(object)b);
        }

        return _cpuFallback.Divide(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Divide<T>(Vector<T> vector, T scalar)
    {
        if (vector.Length < _thresholds.VectorDivide)
            return _cpuFallback.Divide(vector, scalar);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)DivideScalarGpu((Vector<float>)(object)vector, (float)(object)scalar!);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)DivideScalarGpuDouble((Vector<double>)(object)vector, (double)(object)scalar!);
            if (typeof(T) == typeof(int))
                return (Vector<T>)(object)DivideScalarGpuInt((Vector<int>)(object)vector, (int)(object)scalar!);
            if (typeof(T) == typeof(long))
                return (Vector<T>)(object)DivideScalarGpuLong((Vector<long>)(object)vector, (long)(object)scalar!);
        }

        return _cpuFallback.Divide(vector, scalar);
    }

    /// <inheritdoc/>
    public Vector<T> Sqrt<T>(Vector<T> vector)
    {
        if (vector.Length < _thresholds.VectorSqrt)
            return _cpuFallback.Sqrt(vector);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)SqrtGpu((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)SqrtGpuDouble((Vector<double>)(object)vector);
        }

        return _cpuFallback.Sqrt(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Power<T>(Vector<T> vector, T exponent)
    {
        if (vector.Length < _thresholds.VectorPower)
            return _cpuFallback.Power(vector, exponent);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)PowerGpu((Vector<float>)(object)vector, (float)(object)exponent!);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)PowerGpuDouble((Vector<double>)(object)vector, (double)(object)exponent!);
        }

        return _cpuFallback.Power(vector, exponent);
    }

    /// <inheritdoc/>
    public Vector<T> Max<T>(Vector<T> a, Vector<T> b)
    {
        if (a.Length < _thresholds.VectorAdd) // Reuse VectorAdd threshold
            return _cpuFallback.Max(a, b);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)MaxGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)MaxGpuDouble((Vector<double>)(object)a, (Vector<double>)(object)b);
        }

        return _cpuFallback.Max(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Min<T>(Vector<T> a, Vector<T> b)
    {
        if (a.Length < _thresholds.VectorAdd) // Reuse VectorAdd threshold
            return _cpuFallback.Min(a, b);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)MinGpu((Vector<float>)(object)a, (Vector<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)MinGpuDouble((Vector<double>)(object)a, (Vector<double>)(object)b);
        }

        return _cpuFallback.Min(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Abs<T>(Vector<T> vector)
    {
        if (vector.Length < _thresholds.VectorSqrt) // Reuse VectorSqrt threshold
            return _cpuFallback.Abs(vector);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)AbsGpu((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)AbsGpuDouble((Vector<double>)(object)vector);
        }

        return _cpuFallback.Abs(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Exp<T>(Vector<T> vector)
    {
        if (vector.Length < _thresholds.VectorSqrt) // Reuse VectorSqrt threshold
            return _cpuFallback.Exp(vector);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)ExpGpu((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)ExpGpuDouble((Vector<double>)(object)vector);
        }

        return _cpuFallback.Exp(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Log<T>(Vector<T> vector)
    {
        if (vector.Length < _thresholds.VectorSqrt) // Reuse VectorSqrt threshold
            return _cpuFallback.Log(vector);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)LogGpu((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)LogGpuDouble((Vector<double>)(object)vector);
        }

        return _cpuFallback.Log(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Sign<T>(Vector<T> vector)
    {
        if (vector.Length < _thresholds.VectorSqrt) // Reuse VectorSqrt threshold
            return _cpuFallback.Sign(vector);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)SignGpu((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)SignGpuDouble((Vector<double>)(object)vector);
        }

        return _cpuFallback.Sign(vector);
    }

    #region Reduction Operations

    /// <inheritdoc/>
    public T Sum<T>(Vector<T> vector)
    {
        // GPU reduction for large vectors
        if (vector.Length >= _thresholds.VectorAdd && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (T)(object)SumGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (T)(object)SumGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Sum(vector);
    }

    /// <inheritdoc/>
    public T DotProduct<T>(Vector<T> a, Vector<T> b)
    {
        // GPU dot product for large vectors
        if (a.Length >= _thresholds.VectorAdd && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (T)(object)DotProductGpuFloat((Vector<float>)(object)a, (Vector<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (T)(object)DotProductGpuDouble((Vector<double>)(object)a, (Vector<double>)(object)b);
        }
        return _cpuFallback.DotProduct(a, b);
    }

    /// <inheritdoc/>
    public T Mean<T>(Vector<T> vector)
    {
        // GPU mean = sum / length
        if (vector.Length >= _thresholds.VectorAdd && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
            {
                float sum = SumGpuFloat((Vector<float>)(object)vector);
                return (T)(object)(sum / vector.Length);
            }
            if (typeof(T) == typeof(double))
            {
                double sum = SumGpuDouble((Vector<double>)(object)vector);
                return (T)(object)(sum / vector.Length);
            }
        }
        return _cpuFallback.Mean(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Softmax<T>(Vector<T> vector)
    {
        // GPU softmax with numerical stability
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)SoftmaxGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)SoftmaxGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Softmax(vector);
    }

    /// <inheritdoc/>
    public T CosineSimilarity<T>(Vector<T> a, Vector<T> b)
    {
        // GPU cosine similarity using dot product and norms
        if (a.Length >= _thresholds.VectorAdd && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
            {
                float dot = DotProductGpuFloat((Vector<float>)(object)a, (Vector<float>)(object)b);
                float normA = NormGpuFloat((Vector<float>)(object)a);
                float normB = NormGpuFloat((Vector<float>)(object)b);
                if (normA == 0 || normB == 0) return (T)(object)0.0f;
                return (T)(object)(dot / (normA * normB));
            }
            if (typeof(T) == typeof(double))
            {
                double dot = DotProductGpuDouble((Vector<double>)(object)a, (Vector<double>)(object)b);
                double normA = NormGpuDouble((Vector<double>)(object)a);
                double normB = NormGpuDouble((Vector<double>)(object)b);
                if (normA == 0 || normB == 0) return (T)(object)0.0;
                return (T)(object)(dot / (normA * normB));
            }
        }
        return _cpuFallback.CosineSimilarity(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Log2<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)Log2GpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)Log2GpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Log2(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Exp2<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)Exp2GpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)Exp2GpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Exp2(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Exp10<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)Exp10GpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)Exp10GpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Exp10(vector);
    }

    /// <inheritdoc/>
    public Vector<T> ExpM1<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)ExpM1GpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)ExpM1GpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.ExpM1(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Log1P<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)Log1PGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)Log1PGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Log1P(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Negate<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)NegateGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)NegateGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Negate(vector);
    }

    /// <inheritdoc/>
    public T Product<T>(Vector<T> vector)
    {
        // Product reduction is complex on GPU due to numerical instability
        // Using log-sum-exp: prod = exp(sum(log(x)))
        // For now, use CPU for correctness
        return _cpuFallback.Product(vector);
    }

    /// <inheritdoc/>
    public T StdDev<T>(Vector<T> vector)
    {
        // GPU standard deviation using mean and variance
        if (vector.Length >= _thresholds.VectorAdd && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
            {
                float mean = (float)(object)Mean(vector)!;
                float variance = StdDevGpuFloat((Vector<float>)(object)vector, mean);
                return (T)(object)variance;
            }
            if (typeof(T) == typeof(double))
            {
                double mean = (double)(object)Mean(vector)!;
                double variance = StdDevGpuDouble((Vector<double>)(object)vector, mean);
                return (T)(object)variance;
            }
        }
        return _cpuFallback.StdDev(vector);
    }

    /// <inheritdoc/>
    public T Norm<T>(Vector<T> vector)
    {
        // GPU L2 norm: sqrt(sum(x^2))
        if (vector.Length >= _thresholds.VectorAdd && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (T)(object)NormGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (T)(object)NormGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Norm(vector);
    }

    /// <inheritdoc/>
    public T Distance<T>(Vector<T> a, Vector<T> b)
    {
        // GPU Euclidean distance: sqrt(sum((a-b)^2))
        if (a.Length >= _thresholds.VectorAdd && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (T)(object)DistanceGpuFloat((Vector<float>)(object)a, (Vector<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (T)(object)DistanceGpuDouble((Vector<double>)(object)a, (Vector<double>)(object)b);
        }
        return _cpuFallback.Distance(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> MinMagnitude<T>(Vector<T> a, Vector<T> b)
    {
        if (a.Length >= _thresholds.VectorAdd && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)MinMagnitudeGpuFloat((Vector<float>)(object)a, (Vector<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)MinMagnitudeGpuDouble((Vector<double>)(object)a, (Vector<double>)(object)b);
        }
        return _cpuFallback.MinMagnitude(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> MaxMagnitude<T>(Vector<T> a, Vector<T> b)
    {
        if (a.Length >= _thresholds.VectorAdd && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)MaxMagnitudeGpuFloat((Vector<float>)(object)a, (Vector<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)MaxMagnitudeGpuDouble((Vector<double>)(object)a, (Vector<double>)(object)b);
        }
        return _cpuFallback.MaxMagnitude(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Clamp<T>(Vector<T> vector, T min, T max)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)ClampGpuFloat((Vector<float>)(object)vector, (float)(object)min!, (float)(object)max!);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)ClampGpuDouble((Vector<double>)(object)vector, (double)(object)min!, (double)(object)max!);
        }
        return _cpuFallback.Clamp(vector, min, max);
    }

    /// <inheritdoc/>
    public Vector<T> Lerp<T>(Vector<T> a, Vector<T> b, T t)
    {
        if (a.Length >= _thresholds.VectorAdd && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)LerpGpuFloat((Vector<float>)(object)a, (Vector<float>)(object)b, (float)(object)t!);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)LerpGpuDouble((Vector<double>)(object)a, (Vector<double>)(object)b, (double)(object)t!);
        }
        return _cpuFallback.Lerp(a, b, t);
    }

    /// <inheritdoc/>
    public Vector<T> Reciprocal<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)ReciprocalGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)ReciprocalGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Reciprocal(vector);
    }

    /// <inheritdoc/>
    public Vector<T> ReciprocalSqrt<T>(Vector<T> vector)
    {
        // Hardware rsqrt is critical for normalization layers
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)ReciprocalSqrtGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)ReciprocalSqrtGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.ReciprocalSqrt(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Sin<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)SinGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)SinGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Sin(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Cos<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)CosGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)CosGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Cos(vector);
    }

    /// <inheritdoc/>
    public void SinCos<T>(Vector<T> vector, out Vector<T> sinResult, out Vector<T> cosResult)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
            {
                sinResult = (Vector<T>)(object)SinGpuFloat((Vector<float>)(object)vector);
                cosResult = (Vector<T>)(object)CosGpuFloat((Vector<float>)(object)vector);
                return;
            }
            if (typeof(T) == typeof(double))
            {
                sinResult = (Vector<T>)(object)SinGpuDouble((Vector<double>)(object)vector);
                cosResult = (Vector<T>)(object)CosGpuDouble((Vector<double>)(object)vector);
                return;
            }
        }
        _cpuFallback.SinCos(vector, out sinResult, out cosResult);
    }

    /// <inheritdoc/>
    public Vector<T> Sinh<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)SinhGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)SinhGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Sinh(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Cosh<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)CoshGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)CoshGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Cosh(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Asinh<T>(Vector<T> vector)
    {
        // Asinh not available in XMath, use CPU
        return _cpuFallback.Asinh(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Acosh<T>(Vector<T> vector)
    {
        // Acosh not available in XMath, use CPU
        return _cpuFallback.Acosh(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Atanh<T>(Vector<T> vector)
    {
        // Atanh not available in XMath, use CPU
        return _cpuFallback.Atanh(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Round<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)RoundGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)RoundGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Round(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Floor<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)FloorGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)FloorGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Floor(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Ceiling<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)CeilingGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)CeilingGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Ceiling(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Truncate<T>(Vector<T> vector)
    {
        if (vector.Length >= _thresholds.VectorSqrt && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)TruncateGpuFloat((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)TruncateGpuDouble((Vector<double>)(object)vector);
        }
        return _cpuFallback.Truncate(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Fill<T>(int length, T value)
    {
        if (length >= _thresholds.VectorAdd && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)FillGpuFloat(length, (float)(object)value!);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)FillGpuDouble(length, (double)(object)value!);
        }
        return _cpuFallback.Fill(length, value);
    }

    /// <inheritdoc/>
    public Vector<T> FillZero<T>(int length)
    {
        if (length >= _thresholds.VectorAdd && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)FillGpuFloat(length, 0.0f);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)FillGpuDouble(length, 0.0);
        }
        return _cpuFallback.FillZero<T>(length);
    }

    /// <inheritdoc/>
    public Vector<T> GenerateDropoutMask<T>(int length, T dropoutRate, T scale, int? seed = null)
    {
        // GPU random number generation requires cuRAND integration
        // CPU fallback maintains reproducibility with seed
        return _cpuFallback.GenerateDropoutMask(length, dropoutRate, scale, seed);
    }

    /// <inheritdoc/>
    public void CopyVectorToTensor<T>(Vector<T> source, Tensor<T> destination)
    {
        // Direct memory copy handled by CPU for cross-type flexibility
        _cpuFallback.CopyVectorToTensor(source, destination);
    }

    /// <inheritdoc/>
    public Vector<T> GenerateGaussianNoise<T>(int length, T mean, T standardDeviation, int? seed = null)
    {
        // GPU random number generation requires cuRAND integration
        // CPU fallback maintains reproducibility with seed
        return _cpuFallback.GenerateGaussianNoise(length, mean, standardDeviation, seed);
    }

    #endregion

    #region Activation Functions

    /// <inheritdoc/>
    public Vector<T> Tanh<T>(Vector<T> vector)
    {
        if (vector.Length < _thresholds.VectorSqrt)
            return _cpuFallback.Tanh(vector);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)TanhGpu((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)TanhGpuDouble((Vector<double>)(object)vector);
        }

        return _cpuFallback.Tanh(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Sigmoid<T>(Vector<T> vector)
    {
        if (vector.Length < _thresholds.VectorSqrt)
            return _cpuFallback.Sigmoid(vector);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)SigmoidGpu((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)SigmoidGpuDouble((Vector<double>)(object)vector);
        }

        return _cpuFallback.Sigmoid(vector);
    }

    /// <inheritdoc/>
    public Vector<T> ReLU<T>(Vector<T> vector)
    {
        if (vector.Length < _thresholds.VectorSqrt)
            return _cpuFallback.ReLU(vector);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)ReLUGpu((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)ReLUGpuDouble((Vector<double>)(object)vector);
        }

        return _cpuFallback.ReLU(vector);
    }

    /// <inheritdoc/>
    public Tensor<T> Tanh<T>(Tensor<T> tensor)
    {
        if (tensor.Length < _thresholds.MatrixMultiply)
            return _cpuFallback.Tanh(tensor);

        if (SupportsGpu && _gpuHealthy)
        {
            var flatVector = tensor.ToVector();
            if (typeof(T) == typeof(float))
            {
                var result = TanhGpu((Vector<float>)(object)flatVector);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
            if (typeof(T) == typeof(double))
            {
                var result = TanhGpuDouble((Vector<double>)(object)flatVector);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
        }

        return _cpuFallback.Tanh(tensor);
    }

    /// <inheritdoc/>
    public Tensor<T> Sigmoid<T>(Tensor<T> tensor)
    {
        if (tensor.Length < _thresholds.MatrixMultiply)
            return _cpuFallback.Sigmoid(tensor);

        if (SupportsGpu && _gpuHealthy)
        {
            var flatVector = tensor.ToVector();
            if (typeof(T) == typeof(float))
            {
                var result = SigmoidGpu((Vector<float>)(object)flatVector);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
            if (typeof(T) == typeof(double))
            {
                var result = SigmoidGpuDouble((Vector<double>)(object)flatVector);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
        }

        return _cpuFallback.Sigmoid(tensor);
    }

    /// <inheritdoc/>
    public Tensor<T> ReLU<T>(Tensor<T> tensor)
    {
        if (tensor.Length < _thresholds.MatrixMultiply)
            return _cpuFallback.ReLU(tensor);

        if (SupportsGpu && _gpuHealthy)
        {
            var flatVector = tensor.ToVector();
            if (typeof(T) == typeof(float))
            {
                var result = ReLUGpu((Vector<float>)(object)flatVector);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
            if (typeof(T) == typeof(double))
            {
                var result = ReLUGpuDouble((Vector<double>)(object)flatVector);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
        }

        return _cpuFallback.ReLU(tensor);
    }

    /// <inheritdoc/>
    public Vector<T> GELU<T>(Vector<T> vector)
    {
        if (vector.Length < _thresholds.VectorSqrt)
            return _cpuFallback.GELU(vector);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)GELUGpu((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)GELUGpuDouble((Vector<double>)(object)vector);
        }

        return _cpuFallback.GELU(vector);
    }

    /// <inheritdoc/>
    public Tensor<T> GELU<T>(Tensor<T> tensor)
    {
        if (tensor.Length < _thresholds.MatrixMultiply)
            return _cpuFallback.GELU(tensor);

        if (SupportsGpu && _gpuHealthy)
        {
            var flatVector = tensor.ToVector();
            if (typeof(T) == typeof(float))
            {
                var result = GELUGpu((Vector<float>)(object)flatVector);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
            if (typeof(T) == typeof(double))
            {
                var result = GELUGpuDouble((Vector<double>)(object)flatVector);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
        }

        return _cpuFallback.GELU(tensor);
    }

    /// <inheritdoc/>
    public Vector<T> Mish<T>(Vector<T> vector)
    {
        if (vector.Length < _thresholds.VectorSqrt)
            return _cpuFallback.Mish(vector);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)MishGpu((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)MishGpuDouble((Vector<double>)(object)vector);
        }

        return _cpuFallback.Mish(vector);
    }

    /// <inheritdoc/>
    public Tensor<T> Mish<T>(Tensor<T> tensor)
    {
        if (tensor.Length < _thresholds.MatrixMultiply)
            return _cpuFallback.Mish(tensor);

        if (SupportsGpu && _gpuHealthy)
        {
            var flatVector = tensor.ToVector();
            if (typeof(T) == typeof(float))
            {
                var result = MishGpu((Vector<float>)(object)flatVector);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
            if (typeof(T) == typeof(double))
            {
                var result = MishGpuDouble((Vector<double>)(object)flatVector);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
        }

        return _cpuFallback.Mish(tensor);
    }

    /// <inheritdoc/>
    public Vector<T> Swish<T>(Vector<T> vector)
    {
        if (vector.Length < _thresholds.VectorSqrt)
            return _cpuFallback.Swish(vector);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)SwishGpu((Vector<float>)(object)vector);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)SwishGpuDouble((Vector<double>)(object)vector);
        }

        return _cpuFallback.Swish(vector);
    }

    /// <inheritdoc/>
    public Tensor<T> Swish<T>(Tensor<T> tensor)
    {
        if (tensor.Length < _thresholds.MatrixMultiply)
            return _cpuFallback.Swish(tensor);

        if (SupportsGpu && _gpuHealthy)
        {
            var flatVector = tensor.ToVector();
            if (typeof(T) == typeof(float))
            {
                var result = SwishGpu((Vector<float>)(object)flatVector);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
            if (typeof(T) == typeof(double))
            {
                var result = SwishGpuDouble((Vector<double>)(object)flatVector);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
        }

        return _cpuFallback.Swish(tensor);
    }

    /// <inheritdoc/>
    public Vector<T> ELU<T>(Vector<T> vector, double alpha = 1.0)
    {
        if (vector.Length < _thresholds.VectorSqrt)
            return _cpuFallback.ELU(vector, alpha);

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Vector<T>)(object)ELUGpu((Vector<float>)(object)vector, (float)alpha);
            if (typeof(T) == typeof(double))
                return (Vector<T>)(object)ELUGpuDouble((Vector<double>)(object)vector, alpha);
        }

        return _cpuFallback.ELU(vector, alpha);
    }

    /// <inheritdoc/>
    public Tensor<T> ELU<T>(Tensor<T> tensor, double alpha = 1.0)
    {
        if (tensor.Length < _thresholds.MatrixMultiply)
            return _cpuFallback.ELU(tensor, alpha);

        if (SupportsGpu && _gpuHealthy)
        {
            var flatVector = tensor.ToVector();
            if (typeof(T) == typeof(float))
            {
                var result = ELUGpu((Vector<float>)(object)flatVector, (float)alpha);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
            if (typeof(T) == typeof(double))
            {
                var result = ELUGpuDouble((Vector<double>)(object)flatVector, alpha);
                return new Tensor<T>(tensor.Shape, (Vector<T>)(object)result);
            }
        }

        return _cpuFallback.ELU(tensor, alpha);
    }

    #endregion

    #region GPU Kernels (Float Implementation)

    // Note: These are simple, unoptimized kernels for the prototype.
    // Production implementation would use optimized ILGPU.Algorithms or custom kernels.

    private Vector<float> AddGpu(Vector<float> a, Vector<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<float>(a.Length);

        // Rent GPU memory from pool (Phase B: US-GPU-002)
        var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            // Zero-copy: Use span instead of ToArray() (Phase B: US-GPU-003)
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                // Use pre-compiled cached kernel (Phase B: US-GPU-001)
                (_addKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            // Zero-copy: Write directly to result's internal storage (Phase B: US-GPU-003)
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());

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
            // Critical GPU failure - record and potentially recover (Phase B: US-GPU-006, US-GPU-020)
            RecordGpuFailure(ex);
            return _cpuFallback.Add(a, b);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
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
        var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            (_subtractKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_subtractKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
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
        var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            (_multiplyKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_multiplyKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
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
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());
            (_multiplyScalarKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, scalar, gpuResult.View);
            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_multiplyScalarKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, scalar, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
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
        var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            (_divideKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_divideKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
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
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());
            (_divideScalarKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, scalar, gpuResult.View);
            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_divideScalarKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, scalar, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
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
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());
            (_sqrtKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_sqrtKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
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
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());
            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_powerKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, exponent, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> MaxGpu(Vector<float> a, Vector<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<float>(a.Length);
        var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_maxKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Max(a, b);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuA);
            _memoryPoolFloat.Return(gpuB);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> MinGpu(Vector<float> a, Vector<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<float>(a.Length);
        var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_minKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Min(a, b);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuA);
            _memoryPoolFloat.Return(gpuB);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> AbsGpu(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_absKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Abs(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> ExpGpu(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_expKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Exp(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> LogGpu(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_logKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Log(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> SignGpu(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_signKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Sign(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    // Activation function GPU implementations (Phase B: US-GPU-004)
    private Vector<float> TanhGpu(Vector<float> input)
    {
        var result = new Vector<float>(input.Length);
        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            // Zero-copy: Use span instead of ToArray()
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());

            // Thread-safe kernel execution
            lock (_gpuLock)
            {
                (_tanhKernelFloat ?? throw new InvalidOperationException("Tanh kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            // Zero-copy: Write directly to result
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());

            return result;
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.Tanh(input);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Tanh(input);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.Tanh(input);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> SigmoidGpu(Vector<float> input)
    {
        var result = new Vector<float>(input.Length);
        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());

            lock (_gpuLock)
            {
                (_sigmoidKernelFloat ?? throw new InvalidOperationException("Sigmoid kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());

            return result;
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.Sigmoid(input);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Sigmoid(input);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.Sigmoid(input);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> ReLUGpu(Vector<float> input)
    {
        var result = new Vector<float>(input.Length);
        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());

            lock (_gpuLock)
            {
                (_reluKernelFloat ?? throw new InvalidOperationException("ReLU kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());

            return result;
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.ReLU(input);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.ReLU(input);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.ReLU(input);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> GELUGpu(Vector<float> input)
    {
        var result = new Vector<float>(input.Length);
        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());

            lock (_gpuLock)
            {
                (_geluKernelFloat ?? throw new InvalidOperationException("GELU kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());

            return result;
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.GELU(input);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.GELU(input);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.GELU(input);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> MishGpu(Vector<float> input)
    {
        var result = new Vector<float>(input.Length);
        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());

            lock (_gpuLock)
            {
                (_mishKernelFloat ?? throw new InvalidOperationException("Mish kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());

            return result;
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.Mish(input);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Mish(input);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.Mish(input);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> SwishGpu(Vector<float> input)
    {
        var result = new Vector<float>(input.Length);
        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());

            lock (_gpuLock)
            {
                (_swishKernelFloat ?? throw new InvalidOperationException("Swish kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());

            return result;
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.Swish(input);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Swish(input);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.Swish(input);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> ELUGpu(Vector<float> input, float alpha)
    {
        var result = new Vector<float>(input.Length);
        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());

            lock (_gpuLock)
            {
                (_eluKernelFloat ?? throw new InvalidOperationException("ELU kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    alpha,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());

            return result;
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.ELU(input, alpha);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.ELU(input, alpha);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.ELU(input, alpha);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    // Double activation function GPU implementations
    private Vector<double> TanhGpuDouble(Vector<double> input)
    {
        var result = new Vector<double>(input.Length);
        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());
            lock (_gpuLock)
            {
                (_tanhKernelDouble ?? throw new InvalidOperationException("Tanh kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length, gpuInput.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> SigmoidGpuDouble(Vector<double> input)
    {
        var result = new Vector<double>(input.Length);
        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());
            lock (_gpuLock)
            {
                (_sigmoidKernelDouble ?? throw new InvalidOperationException("Sigmoid kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length, gpuInput.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> ReLUGpuDouble(Vector<double> input)
    {
        var result = new Vector<double>(input.Length);
        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());
            lock (_gpuLock)
            {
                (_reluKernelDouble ?? throw new InvalidOperationException("ReLU kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length, gpuInput.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> GELUGpuDouble(Vector<double> input)
    {
        var result = new Vector<double>(input.Length);
        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());
            lock (_gpuLock)
            {
                (_geluKernelDouble ?? throw new InvalidOperationException("GELU kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length, gpuInput.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> MishGpuDouble(Vector<double> input)
    {
        var result = new Vector<double>(input.Length);
        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());
            lock (_gpuLock)
            {
                (_mishKernelDouble ?? throw new InvalidOperationException("Mish kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length, gpuInput.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> SwishGpuDouble(Vector<double> input)
    {
        var result = new Vector<double>(input.Length);
        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());
            lock (_gpuLock)
            {
                (_swishKernelDouble ?? throw new InvalidOperationException("Swish kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length, gpuInput.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> ELUGpuDouble(Vector<double> input, double alpha)
    {
        var result = new Vector<double>(input.Length);
        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());
            lock (_gpuLock)
            {
                (_eluKernelDouble ?? throw new InvalidOperationException("ELU kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length, gpuInput.View, alpha, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private void SinGpuFloat(ReadOnlySpan<float> input, Span<float> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_sinKernelFloat ?? throw new InvalidOperationException("Sin kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorFloat>(input, destination);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private void CosGpuFloat(ReadOnlySpan<float> input, Span<float> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_cosKernelFloat ?? throw new InvalidOperationException("Cos kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorFloat>(input, destination);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private void SinGpuDouble(ReadOnlySpan<double> input, Span<double> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_sinKernelDouble ?? throw new InvalidOperationException("Sin kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorDouble>(input, destination);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private void CosGpuDouble(ReadOnlySpan<double> input, Span<double> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_cosKernelDouble ?? throw new InvalidOperationException("Cos kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorDouble>(input, destination);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private void TanGpuFloat(ReadOnlySpan<float> input, Span<float> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_tanKernelFloat ?? throw new InvalidOperationException("Tan kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorFloat>(input, destination);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private void TanGpuDouble(ReadOnlySpan<double> input, Span<double> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_tanKernelDouble ?? throw new InvalidOperationException("Tan kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorDouble>(input, destination);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private void ExpGpuFloat(ReadOnlySpan<float> input, Span<float> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_expKernelFloat ?? throw new InvalidOperationException("Exp kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorFloat>(input, destination);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private void LogGpuFloat(ReadOnlySpan<float> input, Span<float> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_logKernelFloat ?? throw new InvalidOperationException("Log kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorFloat>(input, destination);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private void ExpGpuDouble(ReadOnlySpan<double> input, Span<double> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_expKernelDouble ?? throw new InvalidOperationException("Exp kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorDouble>(input, destination);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private void LogGpuDouble(ReadOnlySpan<double> input, Span<double> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_logKernelDouble ?? throw new InvalidOperationException("Log kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorDouble>(input, destination);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private void SqrtGpuFloat(ReadOnlySpan<float> input, Span<float> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_sqrtKernelFloat ?? throw new InvalidOperationException("Sqrt kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorFloat>(input, destination);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private void SqrtGpuDouble(ReadOnlySpan<double> input, Span<double> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_sqrtKernelDouble ?? throw new InvalidOperationException("Sqrt kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorDouble>(input, destination);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private void AbsGpuFloat(ReadOnlySpan<float> input, Span<float> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_absKernelFloat ?? throw new InvalidOperationException("Abs kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorFloat>(input, destination);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private void AbsGpuDouble(ReadOnlySpan<double> input, Span<double> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_absKernelDouble ?? throw new InvalidOperationException("Abs kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorDouble>(input, destination);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private void SinhGpuFloat(ReadOnlySpan<float> input, Span<float> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_sinhKernelFloat ?? throw new InvalidOperationException("Sinh kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorFloat>(input, destination);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private void SinhGpuDouble(ReadOnlySpan<double> input, Span<double> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_sinhKernelDouble ?? throw new InvalidOperationException("Sinh kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorDouble>(input, destination);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private void CoshGpuFloat(ReadOnlySpan<float> input, Span<float> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_coshKernelFloat ?? throw new InvalidOperationException("Cosh kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorFloat>(input, destination);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private void CoshGpuDouble(ReadOnlySpan<double> input, Span<double> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_coshKernelDouble ?? throw new InvalidOperationException("Cosh kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorDouble>(input, destination);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private void TanhGpuFloat(ReadOnlySpan<float> input, Span<float> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_tanhKernelFloat ?? throw new InvalidOperationException("Tanh kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorFloat>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorFloat>(input, destination);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private void TanhGpuDouble(ReadOnlySpan<double> input, Span<double> destination)
    {
        if (input.Length != destination.Length)
            throw new ArgumentException("Input and destination lengths must match");

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input);

            lock (_gpuLock)
            {
                (_tanhKernelDouble ?? throw new InvalidOperationException("Tanh kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    input.Length,
                    gpuInput.View,
                    gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(destination);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorDouble>(input, destination);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU operation failed: {ex.Message}. Falling back to CPU.");
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorDouble>(input, destination);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    // Float GPU helper methods for Phase C production operations
    private Vector<float> Log2GpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_log2KernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Log2(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> Exp2GpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_exp2KernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Exp2(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> Exp10GpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_exp10KernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Exp10(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> ExpM1GpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_expM1KernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.ExpM1(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> Log1PGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_log1PKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Log1P(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> NegateGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_negateKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Negate(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> ClampGpuFloat(Vector<float> vector, float min, float max)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_clampKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, min, max, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Clamp(vector, min, max);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> LerpGpuFloat(Vector<float> a, Vector<float> b, float t)
    {
        var result = new Vector<float>(a.Length);
        var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            lock (_gpuLock)
            {
                (_lerpKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, t, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Lerp(a, b, t);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuA);
            _memoryPoolFloat.Return(gpuB);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> ReciprocalGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_reciprocalKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Reciprocal(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> ReciprocalSqrtGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_rsqrtKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.ReciprocalSqrt(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> MinMagnitudeGpuFloat(Vector<float> a, Vector<float> b)
    {
        var result = new Vector<float>(a.Length);
        var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            lock (_gpuLock)
            {
                (_minMagnitudeKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.MinMagnitude(a, b);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuA);
            _memoryPoolFloat.Return(gpuB);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> MaxMagnitudeGpuFloat(Vector<float> a, Vector<float> b)
    {
        var result = new Vector<float>(a.Length);
        var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            lock (_gpuLock)
            {
                (_maxMagnitudeKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.MaxMagnitude(a, b);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuA);
            _memoryPoolFloat.Return(gpuB);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> RoundGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_roundKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Round(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> FloorGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_floorKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Floor(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> CeilingGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_ceilingKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Ceiling(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> TruncateGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_truncateKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Truncate(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> FillGpuFloat(int length, float value)
    {
        var result = new Vector<float>(length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(length);

        try
        {
            lock (_gpuLock)
            {
                (_fillKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, length, gpuResult.View, value);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Fill(length, value);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private float NormGpuFloat(Vector<float> vector)
    {
        // Use partial sums for L2 norm: sqrt(sum(x^2))
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var numBlocks = (vector.Length + ReductionBlockSize - 1) / ReductionBlockSize;
        var gpuPartialSums = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(numBlocks);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                // Compute partial dot products (x dot x = sum of squares)
                (_partialDotProductKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, numBlocks, gpuVector.View, gpuVector.View, gpuPartialSums.View, vector.Length);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            // Sum partial results on CPU
            var partialSums = new float[numBlocks];
            gpuPartialSums.View.BaseView.CopyToCPU(partialSums);
            float sumOfSquares = 0;
            for (int i = 0; i < numBlocks; i++)
                sumOfSquares += partialSums[i];
            return (float)Math.Sqrt(sumOfSquares);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Norm(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuPartialSums);
        }
    }

    private float StdDevGpuFloat(Vector<float> vector, float mean)
    {
        // Compute variance: sum((x - mean)^2) / n, then sqrt
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuTemp = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var numBlocks = (vector.Length + ReductionBlockSize - 1) / ReductionBlockSize;
        var gpuPartialSums = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(numBlocks);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                // Compute (x - mean)^2 in temp buffer using available kernels
                // This is a simplified approach - for production, a dedicated variance kernel would be more efficient
                (_partialSumKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, numBlocks, gpuVector.View, gpuPartialSums.View, vector.Length);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            // Compute variance on CPU with the mean
            var data = vector.AsSpan();
            float sumSquaredDiff = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                float diff = data[i] - mean;
                sumSquaredDiff += diff * diff;
            }
            return (float)Math.Sqrt(sumSquaredDiff / vector.Length);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.StdDev(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuTemp);
            _memoryPoolFloat.Return(gpuPartialSums);
        }
    }

    private float DistanceGpuFloat(Vector<float> a, Vector<float> b)
    {
        // Euclidean distance: sqrt(sum((a-b)^2))
        var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuDiff = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var numBlocks = (a.Length + ReductionBlockSize - 1) / ReductionBlockSize;
        var gpuPartialSums = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(numBlocks);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            lock (_gpuLock)
            {
                // Compute a - b
                (_subtractKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuDiff.View);
                // Compute sum of (a-b)^2
                (_partialDotProductKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, numBlocks, gpuDiff.View, gpuDiff.View, gpuPartialSums.View, a.Length);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            // Sum partial results on CPU
            var partialSums = new float[numBlocks];
            gpuPartialSums.View.BaseView.CopyToCPU(partialSums);
            float sumOfSquares = 0;
            for (int i = 0; i < numBlocks; i++)
                sumOfSquares += partialSums[i];
            return (float)Math.Sqrt(sumOfSquares);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Distance(a, b);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuA);
            _memoryPoolFloat.Return(gpuB);
            _memoryPoolFloat.Return(gpuDiff);
            _memoryPoolFloat.Return(gpuPartialSums);
        }
    }

    private Vector<float> SinGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_sinKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Sin(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> CosGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_cosKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Cos(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> SinhGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_sinhKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Sinh(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Vector<float> CoshGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_coshKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Cosh(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private float SumGpuFloat(Vector<float> vector)
    {
        // Use partial sums reduction
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var numBlocks = (vector.Length + ReductionBlockSize - 1) / ReductionBlockSize;
        var gpuPartialSums = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(numBlocks);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_partialSumKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, numBlocks, gpuVector.View, gpuPartialSums.View, vector.Length);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            // Sum partial results on CPU
            var partialSums = new float[numBlocks];
            gpuPartialSums.View.BaseView.CopyToCPU(partialSums);
            float sum = 0;
            for (int i = 0; i < numBlocks; i++)
                sum += partialSums[i];
            return sum;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Sum(vector);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuVector);
            _memoryPoolFloat.Return(gpuPartialSums);
        }
    }

    private float DotProductGpuFloat(Vector<float> a, Vector<float> b)
    {
        // Use partial dot product reduction
        var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var numBlocks = (a.Length + ReductionBlockSize - 1) / ReductionBlockSize;
        var gpuPartialSums = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(numBlocks);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            lock (_gpuLock)
            {
                (_partialDotProductKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, numBlocks, gpuA.View, gpuB.View, gpuPartialSums.View, a.Length);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            // Sum partial results on CPU
            var partialSums = new float[numBlocks];
            gpuPartialSums.View.BaseView.CopyToCPU(partialSums);
            float dot = 0;
            for (int i = 0; i < numBlocks; i++)
                dot += partialSums[i];
            return dot;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.DotProduct(a, b);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuA);
            _memoryPoolFloat.Return(gpuB);
            _memoryPoolFloat.Return(gpuPartialSums);
        }
    }

    private Vector<float> SoftmaxGpuFloat(Vector<float> vector)
    {
        var result = new Vector<float>(vector.Length);
        var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            // Compute max for numerical stability
            float maxVal = float.MinValue;
            var span = vector.AsSpan();
            for (int i = 0; i < span.Length; i++)
                if (span[i] > maxVal) maxVal = span[i];

            // Compute sum(exp(x - max))
            float sumExp = 0;
            for (int i = 0; i < span.Length; i++)
                sumExp += (float)Math.Exp(span[i] - maxVal);

            lock (_gpuLock)
            {
                (_softmaxKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View, maxVal, sumExp);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Softmax(vector);
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
        var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            (_addKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_addKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> SubtractGpuDouble(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<double>(a.Length);
        var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            lock (_gpuLock)
            {
                (_subtractKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> MultiplyGpuDouble(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<double>(a.Length);
        var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            lock (_gpuLock)
            {
                (_multiplyKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> MultiplyScalarGpuDouble(Vector<double> vector, double scalar)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());
            lock (_gpuLock)
            {
                (_multiplyScalarKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, scalar, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> DivideGpuDouble(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<double>(a.Length);
        var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            lock (_gpuLock)
            {
                (_divideKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> DivideScalarGpuDouble(Vector<double> vector, double scalar)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());
            lock (_gpuLock)
            {
                (_divideScalarKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, scalar, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> SqrtGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());
            lock (_gpuLock)
            {
                (_sqrtKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> PowerGpuDouble(Vector<double> vector, double exponent)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());
            lock (_gpuLock)
            {
                (_powerKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, exponent, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> MaxGpuDouble(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<double>(a.Length);
        var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_maxKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Max(a, b);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> MinGpuDouble(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<double>(a.Length);
        var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_minKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Min(a, b);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> AbsGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_absKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Abs(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> ExpGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_expKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Exp(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> LogGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_logKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Log(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> SignGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_signKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Sign(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    // Double GPU helper methods for Phase C production operations
    private Vector<double> Log2GpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_log2KernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Log2(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> Exp2GpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_exp2KernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Exp2(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> Exp10GpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_exp10KernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Exp10(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> ExpM1GpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_expM1KernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.ExpM1(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> Log1PGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_log1PKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Log1P(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> NegateGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_negateKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Negate(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> ClampGpuDouble(Vector<double> vector, double min, double max)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_clampKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, min, max, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Clamp(vector, min, max);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> LerpGpuDouble(Vector<double> a, Vector<double> b, double t)
    {
        var result = new Vector<double>(a.Length);
        var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            lock (_gpuLock)
            {
                (_lerpKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, t, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Lerp(a, b, t);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> ReciprocalGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_reciprocalKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Reciprocal(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> ReciprocalSqrtGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_rsqrtKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.ReciprocalSqrt(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> MinMagnitudeGpuDouble(Vector<double> a, Vector<double> b)
    {
        var result = new Vector<double>(a.Length);
        var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            lock (_gpuLock)
            {
                (_minMagnitudeKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.MinMagnitude(a, b);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> MaxMagnitudeGpuDouble(Vector<double> a, Vector<double> b)
    {
        var result = new Vector<double>(a.Length);
        var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            lock (_gpuLock)
            {
                (_maxMagnitudeKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.MaxMagnitude(a, b);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> RoundGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_roundKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Round(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> FloorGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_floorKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Floor(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> CeilingGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_ceilingKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Ceiling(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> TruncateGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_truncateKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Truncate(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> FillGpuDouble(int length, double value)
    {
        var result = new Vector<double>(length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(length);

        try
        {
            lock (_gpuLock)
            {
                (_fillKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, length, gpuResult.View, value);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Fill(length, value);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private double NormGpuDouble(Vector<double> vector)
    {
        // Use partial sums for L2 norm: sqrt(sum(x^2))
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var numBlocks = (vector.Length + ReductionBlockSize - 1) / ReductionBlockSize;
        var gpuPartialSums = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(numBlocks);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                // Compute partial dot products (x dot x = sum of squares)
                (_partialDotProductKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, numBlocks, gpuVector.View, gpuVector.View, gpuPartialSums.View, vector.Length);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            // Sum partial results on CPU
            var partialSums = new double[numBlocks];
            gpuPartialSums.View.BaseView.CopyToCPU(partialSums);
            double sumOfSquares = 0;
            for (int i = 0; i < numBlocks; i++)
                sumOfSquares += partialSums[i];
            return Math.Sqrt(sumOfSquares);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Norm(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuPartialSums);
        }
    }

    private double StdDevGpuDouble(Vector<double> vector, double mean)
    {
        // Compute variance: sum((x - mean)^2) / n, then sqrt
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuTemp = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var numBlocks = (vector.Length + ReductionBlockSize - 1) / ReductionBlockSize;
        var gpuPartialSums = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(numBlocks);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                // This is a simplified approach - for production, a dedicated variance kernel would be more efficient
                (_partialSumKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, numBlocks, gpuVector.View, gpuPartialSums.View, vector.Length);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            // Compute variance on CPU with the mean
            var data = vector.AsSpan();
            double sumSquaredDiff = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                double diff = data[i] - mean;
                sumSquaredDiff += diff * diff;
            }
            return Math.Sqrt(sumSquaredDiff / vector.Length);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.StdDev(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuTemp);
            _memoryPoolDouble.Return(gpuPartialSums);
        }
    }

    private double DistanceGpuDouble(Vector<double> a, Vector<double> b)
    {
        // Euclidean distance: sqrt(sum((a-b)^2))
        var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var gpuDiff = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var numBlocks = (a.Length + ReductionBlockSize - 1) / ReductionBlockSize;
        var gpuPartialSums = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(numBlocks);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            lock (_gpuLock)
            {
                // Compute a - b
                (_subtractKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuDiff.View);
                // Compute sum of (a-b)^2
                (_partialDotProductKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, numBlocks, gpuDiff.View, gpuDiff.View, gpuPartialSums.View, a.Length);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            // Sum partial results on CPU
            var partialSums = new double[numBlocks];
            gpuPartialSums.View.BaseView.CopyToCPU(partialSums);
            double sumOfSquares = 0;
            for (int i = 0; i < numBlocks; i++)
                sumOfSquares += partialSums[i];
            return Math.Sqrt(sumOfSquares);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Distance(a, b);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuDiff);
            _memoryPoolDouble.Return(gpuPartialSums);
        }
    }

    private Vector<double> SinGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_sinKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Sin(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> CosGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_cosKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Cos(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> SinhGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_sinhKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Sinh(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private Vector<double> CoshGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_coshKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Cosh(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    private double SumGpuDouble(Vector<double> vector)
    {
        // Use partial sums reduction
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var numBlocks = (vector.Length + ReductionBlockSize - 1) / ReductionBlockSize;
        var gpuPartialSums = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(numBlocks);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            lock (_gpuLock)
            {
                (_partialSumKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, numBlocks, gpuVector.View, gpuPartialSums.View, vector.Length);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            // Sum partial results on CPU
            var partialSums = new double[numBlocks];
            gpuPartialSums.View.BaseView.CopyToCPU(partialSums);
            double sum = 0;
            for (int i = 0; i < numBlocks; i++)
                sum += partialSums[i];
            return sum;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Sum(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
            _memoryPoolDouble.Return(gpuPartialSums);
        }
    }

    private double DotProductGpuDouble(Vector<double> a, Vector<double> b)
    {
        // Use partial dot product reduction
        var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
        var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
        var numBlocks = (a.Length + ReductionBlockSize - 1) / ReductionBlockSize;
        var gpuPartialSums = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(numBlocks);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

            lock (_gpuLock)
            {
                (_partialDotProductKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, numBlocks, gpuA.View, gpuB.View, gpuPartialSums.View, a.Length);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            // Sum partial results on CPU
            var partialSums = new double[numBlocks];
            gpuPartialSums.View.BaseView.CopyToCPU(partialSums);
            double dot = 0;
            for (int i = 0; i < numBlocks; i++)
                dot += partialSums[i];
            return dot;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.DotProduct(a, b);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuPartialSums);
        }
    }

    private Vector<double> SoftmaxGpuDouble(Vector<double> vector)
    {
        var result = new Vector<double>(vector.Length);
        var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

            // Compute max for numerical stability
            double maxVal = double.MinValue;
            var span = vector.AsSpan();
            for (int i = 0; i < span.Length; i++)
                if (span[i] > maxVal) maxVal = span[i];

            // Compute sum(exp(x - max))
            double sumExp = 0;
            for (int i = 0; i < span.Length; i++)
                sumExp += Math.Exp(span[i] - maxVal);

            lock (_gpuLock)
            {
                (_softmaxKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, gpuResult.View, maxVal, sumExp);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Softmax(vector);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuVector);
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
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            (_addKernelInt ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_addKernelInt ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
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
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            (_addKernelLong ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
            // Thread-safe kernel execution (Phase B: US-GPU-019)
            lock (_gpuLock)
            {
                (_addKernelLong ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolLong.Return(gpuA);
            _memoryPoolLong.Return(gpuB);
            _memoryPoolLong.Return(gpuResult);
        }
    }

    // Int GPU operations for Subtract, Multiply, Divide
    private Vector<int> SubtractGpuInt(Vector<int> a, Vector<int> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<int>(a.Length);
        var gpuA = _memoryPoolInt!.Rent(a.Length);
        var gpuB = _memoryPoolInt.Rent(b.Length);
        var gpuResult = _memoryPoolInt.Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            lock (_gpuLock)
            {
                (_subtractKernelInt ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolInt.Return(gpuA);
            _memoryPoolInt.Return(gpuB);
            _memoryPoolInt.Return(gpuResult);
        }
    }

    private Vector<int> MultiplyGpuInt(Vector<int> a, Vector<int> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<int>(a.Length);
        var gpuA = _memoryPoolInt!.Rent(a.Length);
        var gpuB = _memoryPoolInt.Rent(b.Length);
        var gpuResult = _memoryPoolInt.Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            lock (_gpuLock)
            {
                (_multiplyKernelInt ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolInt.Return(gpuA);
            _memoryPoolInt.Return(gpuB);
            _memoryPoolInt.Return(gpuResult);
        }
    }

    private Vector<int> MultiplyScalarGpuInt(Vector<int> vector, int scalar)
    {
        var result = new Vector<int>(vector.Length);
        var gpuVector = _memoryPoolInt!.Rent(vector.Length);
        var gpuResult = _memoryPoolInt.Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());
            lock (_gpuLock)
            {
                (_multiplyScalarKernelInt ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, scalar, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolInt.Return(gpuVector);
            _memoryPoolInt.Return(gpuResult);
        }
    }

    private Vector<int> DivideGpuInt(Vector<int> a, Vector<int> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<int>(a.Length);
        var gpuA = _memoryPoolInt!.Rent(a.Length);
        var gpuB = _memoryPoolInt.Rent(b.Length);
        var gpuResult = _memoryPoolInt.Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            lock (_gpuLock)
            {
                (_divideKernelInt ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolInt.Return(gpuA);
            _memoryPoolInt.Return(gpuB);
            _memoryPoolInt.Return(gpuResult);
        }
    }

    private Vector<int> DivideScalarGpuInt(Vector<int> vector, int scalar)
    {
        var result = new Vector<int>(vector.Length);
        var gpuVector = _memoryPoolInt!.Rent(vector.Length);
        var gpuResult = _memoryPoolInt.Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());
            lock (_gpuLock)
            {
                (_divideScalarKernelInt ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, scalar, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolInt.Return(gpuVector);
            _memoryPoolInt.Return(gpuResult);
        }
    }

    // Long GPU operations for Subtract, Multiply, Divide
    private Vector<long> SubtractGpuLong(Vector<long> a, Vector<long> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<long>(a.Length);
        var gpuA = _memoryPoolLong!.Rent(a.Length);
        var gpuB = _memoryPoolLong.Rent(b.Length);
        var gpuResult = _memoryPoolLong.Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            lock (_gpuLock)
            {
                (_subtractKernelLong ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolLong.Return(gpuA);
            _memoryPoolLong.Return(gpuB);
            _memoryPoolLong.Return(gpuResult);
        }
    }

    private Vector<long> MultiplyGpuLong(Vector<long> a, Vector<long> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<long>(a.Length);
        var gpuA = _memoryPoolLong!.Rent(a.Length);
        var gpuB = _memoryPoolLong.Rent(b.Length);
        var gpuResult = _memoryPoolLong.Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            lock (_gpuLock)
            {
                (_multiplyKernelLong ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolLong.Return(gpuA);
            _memoryPoolLong.Return(gpuB);
            _memoryPoolLong.Return(gpuResult);
        }
    }

    private Vector<long> MultiplyScalarGpuLong(Vector<long> vector, long scalar)
    {
        var result = new Vector<long>(vector.Length);
        var gpuVector = _memoryPoolLong!.Rent(vector.Length);
        var gpuResult = _memoryPoolLong.Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());
            lock (_gpuLock)
            {
                (_multiplyScalarKernelLong ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, scalar, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolLong.Return(gpuVector);
            _memoryPoolLong.Return(gpuResult);
        }
    }

    private Vector<long> DivideGpuLong(Vector<long> a, Vector<long> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new Vector<long>(a.Length);
        var gpuA = _memoryPoolLong!.Rent(a.Length);
        var gpuB = _memoryPoolLong.Rent(b.Length);
        var gpuResult = _memoryPoolLong.Rent(a.Length);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
            gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
            lock (_gpuLock)
            {
                (_divideKernelLong ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolLong.Return(gpuA);
            _memoryPoolLong.Return(gpuB);
            _memoryPoolLong.Return(gpuResult);
        }
    }

    private Vector<long> DivideScalarGpuLong(Vector<long> vector, long scalar)
    {
        var result = new Vector<long>(vector.Length);
        var gpuVector = _memoryPoolLong!.Rent(vector.Length);
        var gpuResult = _memoryPoolLong.Rent(vector.Length);

        try
        {
            gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());
            lock (_gpuLock)
            {
                (_divideScalarKernelLong ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, vector.Length, gpuVector.View, scalar, gpuResult.View);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }
            gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
            return result;
        }
        finally
        {
            _memoryPoolLong.Return(gpuVector);
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
            {
                object? scalarObj = (object?)scalar;
                if (scalarObj == null) throw new ArgumentNullException(nameof(scalar));
                return (Matrix<T>)(object)MatrixMultiplyScalarGpu((Matrix<float>)(object)matrix, (float)scalarObj);
            }
            if (typeof(T) == typeof(double))
            {
                object? scalarObj = (object?)scalar;
                if (scalarObj == null) throw new ArgumentNullException(nameof(scalar));
                return (Matrix<T>)(object)MatrixMultiplyScalarGpuDouble((Matrix<double>)(object)matrix, (double)scalarObj);
            }
        }

        return _cpuFallback.MatrixMultiplyScalar(matrix, scalar);
    }

    public Matrix<T> MatrixSubtract<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a.Rows * a.Columns < _thresholds.VectorSubtract)
        {
            return _cpuFallback.MatrixSubtract(a, b);
        }

        // GPU kernel implementation for matrix subtraction pending
        // Using CPU fallback which is already vectorized using Vector operations
        return _cpuFallback.MatrixSubtract(a, b);
    }

    public T MatrixSumOfSquares<T>(Matrix<T> matrix)
    {
        if (matrix.Rows * matrix.Columns < _thresholds.MatrixMultiply)
        {
            return _cpuFallback.MatrixSumOfSquares(matrix);
        }

        // GPU kernel implementation for reduction operation pending
        // Using CPU fallback which is already vectorized using DotProduct on rows
        return _cpuFallback.MatrixSumOfSquares(matrix);
    }

    public void SwapColumns<T>(Matrix<T> matrix, int col1, int col2)
    {
        // GPU kernel implementation for column swapping
        if (typeof(T) == typeof(float))
        {
            var matrixFloat = matrix as Matrix<float>;
            if (matrixFloat != null && _accelerator != null)
            {
                SwapColumnsGpu(matrixFloat, col1, col2);
                return;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var matrixDouble = matrix as Matrix<double>;
            if (matrixDouble != null && _accelerator != null)
            {
                SwapColumnsGpuDouble(matrixDouble, col1, col2);
                return;
            }
        }

        _cpuFallback.SwapColumns(matrix, col1, col2);
    }

    private void SwapColumnsGpu(Matrix<float> matrix, int col1, int col2)
    {
        try
        {
            int rows = matrix.Rows, cols = matrix.Columns;
            
            // Rent GPU memory for the matrix
            var gpuMatrix = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);
            var gpuTemp = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows);
            
            try
            {
                // Copy matrix to GPU
                gpuMatrix.View.BaseView.CopyFromCPU(matrix.AsSpan());
                
                // Create 2D view
                var view2D = gpuMatrix.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));
                
                // Execute swap columns kernel
                lock (_gpuLock)
                {
                    (_swapColumnsKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))
                        ((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, rows, view2D, gpuTemp.View, col1, col2);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }
                
                // Copy result back
                gpuMatrix.View.BaseView.CopyToCPU(matrix.AsWritableSpan());
            }
            finally
            {
                _memoryPoolFloat.Return(gpuMatrix);
                _memoryPoolFloat.Return(gpuTemp);
            }
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted for swap columns: {ex.Message}. Falling back to CPU.");
            // CPU fallback
            for (int i = 0; i < matrix.Rows; i++)
            {
                float temp = matrix[i, col1];
                matrix[i, col1] = matrix[i, col2];
                matrix[i, col2] = temp;
            }
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            // CPU fallback
            for (int i = 0; i < matrix.Rows; i++)
            {
                float temp = matrix[i, col1];
                matrix[i, col1] = matrix[i, col2];
                matrix[i, col2] = temp;
            }
        }
    }

    private void SwapColumnsGpuDouble(Matrix<double> matrix, int col1, int col2)
    {
        try
        {
            int rows = matrix.Rows, cols = matrix.Columns;
            
            // Rent GPU memory for the matrix
            var gpuMatrix = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);
            var gpuTemp = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows);
            
            try
            {
                // Copy matrix to GPU
                gpuMatrix.View.BaseView.CopyFromCPU(matrix.AsSpan());
                
                // Create 2D view
                var view2D = gpuMatrix.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));
                
                // Execute swap columns kernel
                lock (_gpuLock)
                {
                    (_swapColumnsKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))
                        ((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, rows, view2D, gpuTemp.View, col1, col2);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }
                
                // Copy result back
                gpuMatrix.View.BaseView.CopyToCPU(matrix.AsWritableSpan());
            }
            finally
            {
                _memoryPoolDouble.Return(gpuMatrix);
                _memoryPoolDouble.Return(gpuTemp);
            }
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted for swap columns: {ex.Message}. Falling back to CPU.");
            // CPU fallback
            for (int i = 0; i < matrix.Rows; i++)
            {
                double temp = matrix[i, col1];
                matrix[i, col1] = matrix[i, col2];
                matrix[i, col2] = temp;
            }
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            // CPU fallback
            for (int i = 0; i < matrix.Rows; i++)
            {
                double temp = matrix[i, col1];
                matrix[i, col1] = matrix[i, col2];
                matrix[i, col2] = temp;
            }
        }
    }

    public void SwapRows<T>(Matrix<T> matrix, int row1, int row2)
    {
        // GPU kernel implementation for row swapping
        if (typeof(T) == typeof(float))
        {
            var matrixFloat = matrix as Matrix<float>;
            if (matrixFloat != null && _accelerator != null)
            {
                SwapRowsGpu(matrixFloat, row1, row2);
                return;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var matrixDouble = matrix as Matrix<double>;
            if (matrixDouble != null && _accelerator != null)
            {
                SwapRowsGpuDouble(matrixDouble, row1, row2);
                return;
            }
        }

        _cpuFallback.SwapRows(matrix, row1, row2);
    }

    private void SwapRowsGpu(Matrix<float> matrix, int row1, int row2)
    {
        try
        {
            int cols = matrix.Columns;
            
            // Rent GPU memory for the two rows
            var gpuRow1 = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(cols);
            var gpuRow2 = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(cols);
            
            try
            {
                // Copy rows to GPU
                gpuRow1.View.BaseView.CopyFromCPU(matrix.GetRowSpan(row1));
                gpuRow2.View.BaseView.CopyFromCPU(matrix.GetRowSpan(row2));
                
                // Execute swap kernel
                lock (_gpuLock)
                {
                    (_swapRowsKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))
                        ((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, cols, gpuRow1.View, gpuRow2.View);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }
                
                // Copy swapped rows back (row1 gets gpuRow2, row2 gets gpuRow1)
                gpuRow2.View.BaseView.CopyToCPU(matrix.GetRowSpan(row1));
                gpuRow1.View.BaseView.CopyToCPU(matrix.GetRowSpan(row2));
            }
            finally
            {
                _memoryPoolFloat.Return(gpuRow1);
                _memoryPoolFloat.Return(gpuRow2);
            }
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted for swap rows: {ex.Message}. Falling back to CPU.");
            // CPU fallback
            var span1 = matrix.GetRowSpan(row1);
            var span2 = matrix.GetRowSpan(row2);
            var tempRow = new float[matrix.Columns];
            span1.CopyTo(tempRow);
            span2.CopyTo(span1);
            tempRow.AsSpan().CopyTo(span2);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            // CPU fallback
            var span1 = matrix.GetRowSpan(row1);
            var span2 = matrix.GetRowSpan(row2);
            var tempRow = new float[matrix.Columns];
            span1.CopyTo(tempRow);
            span2.CopyTo(span1);
            tempRow.AsSpan().CopyTo(span2);
        }
    }

    private void SwapRowsGpuDouble(Matrix<double> matrix, int row1, int row2)
    {
        try
        {
            int cols = matrix.Columns;
            
            // Rent GPU memory for the two rows
            var gpuRow1 = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(cols);
            var gpuRow2 = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(cols);
            
            try
            {
                // Copy rows to GPU
                gpuRow1.View.BaseView.CopyFromCPU(matrix.GetRowSpan(row1));
                gpuRow2.View.BaseView.CopyFromCPU(matrix.GetRowSpan(row2));
                
                // Execute swap kernel
                lock (_gpuLock)
                {
                    (_swapRowsKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))
                        ((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, cols, gpuRow1.View, gpuRow2.View);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }
                
                // Copy swapped rows back (row1 gets gpuRow2, row2 gets gpuRow1)
                gpuRow2.View.BaseView.CopyToCPU(matrix.GetRowSpan(row1));
                gpuRow1.View.BaseView.CopyToCPU(matrix.GetRowSpan(row2));
            }
            finally
            {
                _memoryPoolDouble.Return(gpuRow1);
                _memoryPoolDouble.Return(gpuRow2);
            }
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted for swap rows: {ex.Message}. Falling back to CPU.");
            // CPU fallback
            var span1 = matrix.GetRowSpan(row1);
            var span2 = matrix.GetRowSpan(row2);
            var tempRow = new double[matrix.Columns];
            span1.CopyTo(tempRow);
            span2.CopyTo(span1);
            tempRow.AsSpan().CopyTo(span2);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            // CPU fallback
            var span1 = matrix.GetRowSpan(row1);
            var span2 = matrix.GetRowSpan(row2);
            var tempRow = new double[matrix.Columns];
            span1.CopyTo(tempRow);
            span2.CopyTo(span1);
            tempRow.AsSpan().CopyTo(span2);
        }
    }

    public Matrix<T> OuterProduct<T>(Vector<T> a, Vector<T> b)
    {
        // GPU kernel implementation for outer product
        if (typeof(T) == typeof(float))
        {
            var aFloat = a as Vector<float>;
            var bFloat = b as Vector<float>;
            if (aFloat != null && bFloat != null && _accelerator != null)
            {
                return (OuterProductGpu(aFloat, bFloat) as Matrix<T>)!;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var aDouble = a as Vector<double>;
            var bDouble = b as Vector<double>;
            if (aDouble != null && bDouble != null && _accelerator != null)
            {
                return (OuterProductGpuDouble(aDouble, bDouble) as Matrix<T>)!;
            }
        }

        return _cpuFallback.OuterProduct(a, b);
    }

    private Matrix<float> OuterProductGpu(Vector<float> a, Vector<float> b)
    {
        try
        {
            var result = new Matrix<float>(a.Length, b.Length);
            int m = a.Length, n = b.Length;
            
            // Rent GPU memory
            var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(m);
            var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(n);
            var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(m * n);
            
            try
            {
                // Copy vectors to GPU
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
                
                // Create 2D view for result
                var viewResult = gpuResult.View.As2DView<Stride2D.DenseX>(new Index2D(m, n), new Stride2D.DenseX(n));
                
                // Execute outer product kernel
                lock (_gpuLock)
                {
                    (_outerProductKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))
                        ((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(m, n), gpuA.View, gpuB.View, viewResult, m, n);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }
                
                // Copy result back
                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
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
            Console.WriteLine($"[GpuEngine] GPU memory exhausted for outer product: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.OuterProduct(a, b);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.OuterProduct(a, b);
        }
    }

    private Matrix<double> OuterProductGpuDouble(Vector<double> a, Vector<double> b)
    {
        try
        {
            var result = new Matrix<double>(a.Length, b.Length);
            int m = a.Length, n = b.Length;
            
            // Rent GPU memory
            var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(m);
            var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(n);
            var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(m * n);
            
            try
            {
                // Copy vectors to GPU
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());
                
                // Create 2D view for result
                var viewResult = gpuResult.View.As2DView<Stride2D.DenseX>(new Index2D(m, n), new Stride2D.DenseX(n));
                
                // Execute outer product kernel
                lock (_gpuLock)
                {
                    (_outerProductKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))
                        ((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(m, n), gpuA.View, gpuB.View, viewResult, m, n);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }
                
                // Copy result back
                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuA);
                _memoryPoolDouble.Return(gpuB);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted for outer product: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.OuterProduct(a, b);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.OuterProduct(a, b);
        }
    }

    public Vector<T> GetColumn<T>(Matrix<T> matrix, int columnIndex)
    {
        // Optimized column extraction using GetColumnAsArray
        if (typeof(T) == typeof(float))
        {
            var matrixFloat = matrix as Matrix<float>;
            if (matrixFloat != null)
            {
                var columnArray = matrixFloat.GetColumnAsArray(columnIndex);
                return (new Vector<float>(columnArray) as Vector<T>)!;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var matrixDouble = matrix as Matrix<double>;
            if (matrixDouble != null)
            {
                var columnArray = matrixDouble.GetColumnAsArray(columnIndex);
                return (new Vector<double>(columnArray) as Vector<T>)!;
            }
        }

        return _cpuFallback.GetColumn(matrix, columnIndex);
    }

    public Vector<T> GetRow<T>(Matrix<T> matrix, int rowIndex)
    {
        // Optimized using GetRowSpan for zero-copy access
        if (typeof(T) == typeof(float))
        {
            var matrixFloat = matrix as Matrix<float>;
            if (matrixFloat != null)
            {
                var rowSpan = matrixFloat.GetRowReadOnlySpan(rowIndex);
                return (new Vector<float>(rowSpan.ToArray()) as Vector<T>)!;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var matrixDouble = matrix as Matrix<double>;
            if (matrixDouble != null)
            {
                var rowSpan = matrixDouble.GetRowReadOnlySpan(rowIndex);
                return (new Vector<double>(rowSpan.ToArray()) as Vector<T>)!;
            }
        }

        return _cpuFallback.GetRow(matrix, rowIndex);
    }

    public void SetColumn<T>(Matrix<T> matrix, int columnIndex, Vector<T> values)
    {
        // Optimized column setting using direct indexer
        if (typeof(T) == typeof(float))
        {
            var matrixFloat = matrix as Matrix<float>;
            var valuesFloat = values as Vector<float>;
            if (matrixFloat != null && valuesFloat != null)
            {
                for (int i = 0; i < matrixFloat.Rows; i++)
                {
                    matrixFloat[i, columnIndex] = valuesFloat[i];
                }
                return;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var matrixDouble = matrix as Matrix<double>;
            var valuesDouble = values as Vector<double>;
            if (matrixDouble != null && valuesDouble != null)
            {
                for (int i = 0; i < matrixDouble.Rows; i++)
                {
                    matrixDouble[i, columnIndex] = valuesDouble[i];
                }
                return;
            }
        }

        _cpuFallback.SetColumn(matrix, columnIndex, values);
    }

    public void SetRow<T>(Matrix<T> matrix, int rowIndex, Vector<T> values)
    {
        // Optimized using GetRowSpan for zero-copy access
        if (typeof(T) == typeof(float))
        {
            var matrixFloat = matrix as Matrix<float>;
            var valuesFloat = values as Vector<float>;
            if (matrixFloat != null && valuesFloat != null)
            {
                var rowSpan = matrixFloat.GetRowSpan(rowIndex);
                valuesFloat.AsSpan().CopyTo(rowSpan);
                return;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var matrixDouble = matrix as Matrix<double>;
            var valuesDouble = values as Vector<double>;
            if (matrixDouble != null && valuesDouble != null)
            {
                var rowSpan = matrixDouble.GetRowSpan(rowIndex);
                valuesDouble.AsSpan().CopyTo(rowSpan);
                return;
            }
        }

        _cpuFallback.SetRow(matrix, rowIndex, values);
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
            var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(m * k);
            var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(k * n);
            var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(m * n);

            try
            {
                // Zero-copy transfer (Phase B: US-GPU-003)
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                // Create 2D views
                var viewA = gpuA.View.As2DView<Stride2D.DenseX>(new Index2D(m, k), new Stride2D.DenseX(k));
                var viewB = gpuB.View.As2DView<Stride2D.DenseX>(new Index2D(k, n), new Stride2D.DenseX(n));
                var viewResult = gpuResult.View.As2DView<Stride2D.DenseX>(new Index2D(m, n), new Stride2D.DenseX(n));

                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    // Execute pre-compiled kernel (Phase B: US-GPU-001, US-GPU-007)
                    (_matrixMultiplyKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(m, n), viewA, viewB, viewResult, k);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                // Zero-copy result transfer
                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
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
            RecordGpuFailure(ex);
            return _cpuFallback.MatrixMultiply(a, b);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
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

            var gpuMatrix = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);
            var gpuVector = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(cols);
            var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows);

            try
            {
                gpuMatrix.View.BaseView.CopyFromCPU(matrix.AsSpan());
                gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

                var viewMatrix = gpuMatrix.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));
                (_matrixVectorMultiplyKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, rows, viewMatrix, gpuVector.View, gpuResult.View, rows, cols);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_matrixVectorMultiplyKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, rows, viewMatrix, gpuVector.View, gpuResult.View, rows, cols);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuMatrix);
                _memoryPoolFloat.Return(gpuVector);
                _memoryPoolFloat.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
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

            var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);
            var gpuOutput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);

            try
            {
                gpuInput.View.BaseView.CopyFromCPU(matrix.AsSpan());

                var viewInput = gpuInput.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));
                var viewOutput = gpuOutput.View.As2DView<Stride2D.DenseX>(new Index2D(cols, rows), new Stride2D.DenseX(rows));

                (_matrixTransposeKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(rows, cols), viewInput, viewOutput);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_matrixTransposeKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(rows, cols), viewInput, viewOutput);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuOutput.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuInput);
                _memoryPoolFloat.Return(gpuOutput);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
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

            var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);
            var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);
            var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);

            try
            {
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                var viewA = gpuA.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));
                var viewB = gpuB.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));
                var viewResult = gpuResult.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));

                (_matrixAddKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(rows, cols), viewA, viewB, viewResult);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_matrixAddKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(rows, cols), viewA, viewB, viewResult);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuA);
                _memoryPoolFloat.Return(gpuB);
                _memoryPoolFloat.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
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

            var gpuMatrix = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);
            var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);

            try
            {
                gpuMatrix.View.BaseView.CopyFromCPU(matrix.AsSpan());

                var viewMatrix = gpuMatrix.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));
                var viewResult = gpuResult.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));

                (_matrixMultiplyScalarKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(rows, cols), viewMatrix, scalar, viewResult);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_matrixMultiplyScalarKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(rows, cols), viewMatrix, scalar, viewResult);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuMatrix);
                _memoryPoolFloat.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
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

            var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(m * k);
            var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(k * n);
            var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(m * n);

            try
            {
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                var viewA = gpuA.View.As2DView<Stride2D.DenseX>(new Index2D(m, k), new Stride2D.DenseX(k));
                var viewB = gpuB.View.As2DView<Stride2D.DenseX>(new Index2D(k, n), new Stride2D.DenseX(n));
                var viewResult = gpuResult.View.As2DView<Stride2D.DenseX>(new Index2D(m, n), new Stride2D.DenseX(n));

                (_matrixMultiplyKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(m, n), viewA, viewB, viewResult, k);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_matrixMultiplyKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(m, n), viewA, viewB, viewResult, k);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuA);
                _memoryPoolDouble.Return(gpuB);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
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

            var gpuMatrix = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);
            var gpuVector = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(cols);
            var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows);

            try
            {
                gpuMatrix.View.BaseView.CopyFromCPU(matrix.AsSpan());
                gpuVector.View.BaseView.CopyFromCPU(vector.AsSpan());

                var viewMatrix = gpuMatrix.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));
                (_matrixVectorMultiplyKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, rows, viewMatrix, gpuVector.View, gpuResult.View, rows, cols);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_matrixVectorMultiplyKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, rows, viewMatrix, gpuVector.View, gpuResult.View, rows, cols);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuMatrix);
                _memoryPoolDouble.Return(gpuVector);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
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

            var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);
            var gpuOutput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);

            try
            {
                gpuInput.View.BaseView.CopyFromCPU(matrix.AsSpan());

                var viewInput = gpuInput.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));
                var viewOutput = gpuOutput.View.As2DView<Stride2D.DenseX>(new Index2D(cols, rows), new Stride2D.DenseX(rows));

                (_matrixTransposeKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(rows, cols), viewInput, viewOutput);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_matrixTransposeKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(rows, cols), viewInput, viewOutput);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuOutput.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuInput);
                _memoryPoolDouble.Return(gpuOutput);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
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

            var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);
            var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);
            var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);

            try
            {
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                var viewA = gpuA.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));
                var viewB = gpuB.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));
                var viewResult = gpuResult.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));

                (_matrixAddKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(rows, cols), viewA, viewB, viewResult);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_matrixAddKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(rows, cols), viewA, viewB, viewResult);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuA);
                _memoryPoolDouble.Return(gpuB);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
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

            var gpuMatrix = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);
            var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(rows * cols);

            try
            {
                gpuMatrix.View.BaseView.CopyFromCPU(matrix.AsSpan());

                var viewMatrix = gpuMatrix.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));
                var viewResult = gpuResult.View.As2DView<Stride2D.DenseX>(new Index2D(rows, cols), new Stride2D.DenseX(cols));

                (_matrixMultiplyScalarKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(rows, cols), viewMatrix, scalar, viewResult);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_matrixMultiplyScalarKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index2D(rows, cols), viewMatrix, scalar, viewResult);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuMatrix);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix scalar multiply (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixMultiplyScalar(matrix, scalar);
        }
        catch (ArgumentException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix scalar multiply (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixMultiplyScalar(matrix, scalar);
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU matrix scalar multiply (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MatrixMultiplyScalar(matrix, scalar);
        }
    }

    #endregion

    #region Tensor Operations (Phase B: Epic 3)

    /// <inheritdoc/>
    public Tensor<T> BatchMatMul<T>(Tensor<T> a, Tensor<T> b)
    {
        // Adaptive execution: check size threshold (Phase B: US-GPU-004)
        if (Math.Max(a.Shape[1], a.Shape[2]) < _thresholds.BatchMatMul)
        {
            return _cpuFallback.BatchMatMul(a, b);
        }

        // Check GPU health and type support (Phase B: US-GPU-006)
        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)BatchMatMulGpu((Tensor<float>)(object)a, (Tensor<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)BatchMatMulGpuDouble((Tensor<double>)(object)a, (Tensor<double>)(object)b);
        }

        // Fallback to CPU for unsupported types or unhealthy GPU
        return _cpuFallback.BatchMatMul(a, b);
    }

    private Tensor<float> BatchMatMulGpu(Tensor<float> a, Tensor<float> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rank != 3 || b.Rank != 3)
        {
            throw new ArgumentException(
                $"BatchMatMul requires 3D tensors. Got ranks {a.Rank} and {b.Rank}.");
        }

        int batchSize = a.Shape[0];
        int m = a.Shape[1];
        int k = a.Shape[2];
        int k2 = b.Shape[1];
        int n = b.Shape[2];

        if (b.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Batch sizes must match. Got {batchSize} and {b.Shape[0]}.");
        }
        if (k != k2)
        {
            throw new ArgumentException(
                $"Matrix dimensions incompatible for multiplication. " +
                $"First tensor has shape [{batchSize}, {m}, {k}], " +
                $"second has shape [{b.Shape[0]}, {k2}, {n}]. " +
                $"Inner dimensions must match ({k} != {k2}).");
        }

        try
        {
            var result = new Tensor<float>(new[] { batchSize, m, n });

            // Allocate GPU buffers using memory pool (Phase B: US-GPU-002)
            var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(batchSize * m * k);
            var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(batchSize * k * n);
            var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(batchSize * m * n);

            try
            {
                // Zero-copy transfer (Phase B: US-GPU-003)
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                // Execute pre-compiled kernel (Phase B: US-GPU-001, US-GPU-013)
                (_batchMatMulKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index3D(batchSize, m, n), gpuA.View, gpuB.View, gpuResult.View, m, k, n);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_batchMatMulKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index3D(batchSize, m, n), gpuA.View, gpuB.View, gpuResult.View, m, k, n);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                // Zero-copy result transfer
                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
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
            Console.WriteLine($"[GpuEngine] GPU memory exhausted for batch matmul: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.BatchMatMul(a, b);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.BatchMatMul(a, b);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU batch matmul failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.BatchMatMul(a, b);
        }
    }

    private Tensor<double> BatchMatMulGpuDouble(Tensor<double> a, Tensor<double> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rank != 3 || b.Rank != 3)
        {
            throw new ArgumentException(
                $"BatchMatMul requires 3D tensors. Got ranks {a.Rank} and {b.Rank}.");
        }

        int batchSize = a.Shape[0];
        int m = a.Shape[1];
        int k = a.Shape[2];
        int k2 = b.Shape[1];
        int n = b.Shape[2];

        if (b.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Batch sizes must match. Got {batchSize} and {b.Shape[0]}.");
        }
        if (k != k2)
        {
            throw new ArgumentException(
                $"Matrix dimensions incompatible for multiplication. " +
                $"First tensor has shape [{batchSize}, {m}, {k}], " +
                $"second has shape [{b.Shape[0]}, {k2}, {n}]. " +
                $"Inner dimensions must match ({k} != {k2}).");
        }

        try
        {
            var result = new Tensor<double>(new[] { batchSize, m, n });

            // Allocate GPU buffers using memory pool (Phase B: US-GPU-002)
            var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(batchSize * m * k);
            var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(batchSize * k * n);
            var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(batchSize * m * n);

            try
            {
                // Zero-copy transfer (Phase B: US-GPU-003)
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                // Execute pre-compiled kernel (Phase B: US-GPU-001, US-GPU-013)
                (_batchMatMulKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index3D(batchSize, m, n), gpuA.View, gpuB.View, gpuResult.View, m, k, n);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_batchMatMulKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, new Index3D(batchSize, m, n), gpuA.View, gpuB.View, gpuResult.View, m, k, n);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                // Zero-copy result transfer
                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuA);
                _memoryPoolDouble.Return(gpuB);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"[GpuEngine] GPU memory exhausted for batch matmul (double): {ex.Message}. Falling back to CPU.");
            return _cpuFallback.BatchMatMul(a, b);
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.BatchMatMul(a, b);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU batch matmul (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.BatchMatMul(a, b);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorAdd<T>(Tensor<T> a, Tensor<T> b)
    {
        // Adaptive execution: use vector threshold (Phase B: US-GPU-004)
        if (a.Length < _thresholds.VectorAdd)
        {
            return _cpuFallback.TensorAdd(a, b);
        }

        // Check GPU health and type support (Phase B: US-GPU-006)
        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)TensorAddGpu((Tensor<float>)(object)a, (Tensor<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)TensorAddGpuDouble((Tensor<double>)(object)a, (Tensor<double>)(object)b);
        }

        return _cpuFallback.TensorAdd(a, b);
    }

    private Tensor<float> TensorAddGpu(Tensor<float> a, Tensor<float> b)
    {
        ValidateTensorShapes(a, b);

        try
        {
            var result = new Tensor<float>(a.Shape);
            var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
            var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
            var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

            try
            {
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                (_tensorAddKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_tensorAddKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuA);
                _memoryPoolFloat.Return(gpuB);
                _memoryPoolFloat.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU tensor add failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.TensorAdd(a, b);
        }
    }

    private Tensor<double> TensorAddGpuDouble(Tensor<double> a, Tensor<double> b)
    {
        ValidateTensorShapes(a, b);

        try
        {
            var result = new Tensor<double>(a.Shape);
            var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
            var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
            var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

            try
            {
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                (_tensorAddKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_tensorAddKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuA);
                _memoryPoolDouble.Return(gpuB);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU tensor add (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.TensorAdd(a, b);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSubtract<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length < _thresholds.VectorSubtract)
        {
            return _cpuFallback.TensorSubtract(a, b);
        }

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)TensorSubtractGpu((Tensor<float>)(object)a, (Tensor<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)TensorSubtractGpuDouble((Tensor<double>)(object)a, (Tensor<double>)(object)b);
        }

        return _cpuFallback.TensorSubtract(a, b);
    }

    private Tensor<float> TensorSubtractGpu(Tensor<float> a, Tensor<float> b)
    {
        ValidateTensorShapes(a, b);

        try
        {
            var result = new Tensor<float>(a.Shape);
            var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
            var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
            var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

            try
            {
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                (_tensorSubtractKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_tensorSubtractKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuA);
                _memoryPoolFloat.Return(gpuB);
                _memoryPoolFloat.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU tensor subtract failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.TensorSubtract(a, b);
        }
    }

    private Tensor<double> TensorSubtractGpuDouble(Tensor<double> a, Tensor<double> b)
    {
        ValidateTensorShapes(a, b);

        try
        {
            var result = new Tensor<double>(a.Shape);
            var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
            var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
            var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

            try
            {
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                (_tensorSubtractKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_tensorSubtractKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuA);
                _memoryPoolDouble.Return(gpuB);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU tensor subtract (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.TensorSubtract(a, b);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length < _thresholds.VectorMultiply)
        {
            return _cpuFallback.TensorMultiply(a, b);
        }

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)TensorMultiplyGpu((Tensor<float>)(object)a, (Tensor<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)TensorMultiplyGpuDouble((Tensor<double>)(object)a, (Tensor<double>)(object)b);
        }

        return _cpuFallback.TensorMultiply(a, b);
    }

    private Tensor<float> TensorMultiplyGpu(Tensor<float> a, Tensor<float> b)
    {
        ValidateTensorShapes(a, b);

        try
        {
            var result = new Tensor<float>(a.Shape);
            var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
            var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
            var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

            try
            {
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                (_tensorMultiplyKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_tensorMultiplyKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuA);
                _memoryPoolFloat.Return(gpuB);
                _memoryPoolFloat.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU tensor multiply failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.TensorMultiply(a, b);
        }
    }

    private Tensor<double> TensorMultiplyGpuDouble(Tensor<double> a, Tensor<double> b)
    {
        ValidateTensorShapes(a, b);

        try
        {
            var result = new Tensor<double>(a.Shape);
            var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
            var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
            var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

            try
            {
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                (_tensorMultiplyKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_tensorMultiplyKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuA);
                _memoryPoolDouble.Return(gpuB);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU tensor multiply (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.TensorMultiply(a, b);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMultiplyScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (tensor.Length < _thresholds.VectorMultiply)
        {
            return _cpuFallback.TensorMultiplyScalar(tensor, scalar);
        }

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)TensorMultiplyScalarGpu((Tensor<float>)(object)tensor, (float)(object)scalar!);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)TensorMultiplyScalarGpuDouble((Tensor<double>)(object)tensor, (double)(object)scalar!);
        }

        return _cpuFallback.TensorMultiplyScalar(tensor, scalar);
    }

    private Tensor<float> TensorMultiplyScalarGpu(Tensor<float> tensor, float scalar)
    {
        try
        {
            var result = new Tensor<float>(tensor.Shape);
            var gpuTensor = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(tensor.Length);
            var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(tensor.Length);

            try
            {
                gpuTensor.View.BaseView.CopyFromCPU(tensor.AsSpan());

                (_tensorMultiplyScalarKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, tensor.Length, gpuTensor.View, scalar, gpuResult.View);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_tensorMultiplyScalarKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, tensor.Length, gpuTensor.View, scalar, gpuResult.View);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuTensor);
                _memoryPoolFloat.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU tensor scalar multiply failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.TensorMultiplyScalar(tensor, scalar);
        }
    }

    private Tensor<double> TensorMultiplyScalarGpuDouble(Tensor<double> tensor, double scalar)
    {
        try
        {
            var result = new Tensor<double>(tensor.Shape);
            var gpuTensor = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(tensor.Length);
            var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(tensor.Length);

            try
            {
                gpuTensor.View.BaseView.CopyFromCPU(tensor.AsSpan());

                (_tensorMultiplyScalarKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, tensor.Length, gpuTensor.View, scalar, gpuResult.View);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_tensorMultiplyScalarKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, tensor.Length, gpuTensor.View, scalar, gpuResult.View);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuTensor);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU tensor scalar multiply (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.TensorMultiplyScalar(tensor, scalar);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length < _thresholds.VectorDivide)
        {
            return _cpuFallback.TensorDivide(a, b);
        }

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)TensorDivideGpu((Tensor<float>)(object)a, (Tensor<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)TensorDivideGpuDouble((Tensor<double>)(object)a, (Tensor<double>)(object)b);
        }

        return _cpuFallback.TensorDivide(a, b);
    }

    private Tensor<float> TensorDivideGpu(Tensor<float> a, Tensor<float> b)
    {
        ValidateTensorShapes(a, b);

        try
        {
            var result = new Tensor<float>(a.Shape);
            var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
            var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
            var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

            try
            {
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                (_tensorDivideKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_tensorDivideKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuA);
                _memoryPoolFloat.Return(gpuB);
                _memoryPoolFloat.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU tensor divide failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.TensorDivide(a, b);
        }
    }

    private Tensor<double> TensorDivideGpuDouble(Tensor<double> a, Tensor<double> b)
    {
        ValidateTensorShapes(a, b);

        try
        {
            var result = new Tensor<double>(a.Shape);
            var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);
            var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(b.Length);
            var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(a.Length);

            try
            {
                gpuA.View.BaseView.CopyFromCPU(a.AsSpan());
                gpuB.View.BaseView.CopyFromCPU(b.AsSpan());

                (_tensorDivideKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_tensorDivideKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, a.Length, gpuA.View, gpuB.View, gpuResult.View);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuResult.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuA);
                _memoryPoolDouble.Return(gpuB);
                _memoryPoolDouble.Return(gpuResult);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU tensor divide (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.TensorDivide(a, b);
        }
    }

    /// <summary>
    /// Helper method to validate that two tensors have matching shapes.
    /// </summary>
    private void ValidateTensorShapes<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Shape.Length != b.Shape.Length)
        {
            throw new ArgumentException(
                $"Tensor ranks must match. Got {a.Rank} and {b.Rank}.");
        }

        for (int i = 0; i < a.Shape.Length; i++)
        {
            if (a.Shape[i] != b.Shape[i])
            {
                throw new ArgumentException(
                    $"Tensor shapes must match. Got [{string.Join(", ", a.Shape)}] and [{string.Join(", ", b.Shape)}].");
            }
        }
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        // Adaptive execution: use pooling threshold (Phase B: US-GPU-004)
        if (input.Length < _thresholds.Pooling)
        {
            return _cpuFallback.MaxPool2D(input, poolSize, stride, padding);
        }

        // Check GPU health and type support (Phase B: US-GPU-006)
        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)MaxPool2DGpu((Tensor<float>)(object)input, poolSize, stride, padding);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)MaxPool2DGpuDouble((Tensor<double>)(object)input, poolSize, stride, padding);
        }

        return _cpuFallback.MaxPool2D(input, poolSize, stride, padding);
    }

    private Tensor<float> MaxPool2DGpu(Tensor<float> input, int poolSize, int stride, int padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4)
        {
            throw new ArgumentException($"MaxPool2D requires a 4D tensor. Got rank {input.Rank}.");
        }

        if (stride == 0) stride = poolSize;

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outputHeight = (height + 2 * padding - poolSize) / stride + 1;
        int outputWidth = (width + 2 * padding - poolSize) / stride + 1;

        try
        {
            var result = new Tensor<float>(new[] { batch, channels, outputHeight, outputWidth });
            int outputSize = batch * channels * outputHeight * outputWidth;

            var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
            var gpuOutput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(outputSize);

            try
            {
                gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());

                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_maxPool2DKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, outputSize, gpuInput.View, gpuOutput.View,
                        batch, channels, height, width, outputHeight, outputWidth, poolSize, stride, padding);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuOutput.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuInput);
                _memoryPoolFloat.Return(gpuOutput);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU max pool 2D failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MaxPool2D(input, poolSize, stride, padding);
        }
    }

    private Tensor<double> MaxPool2DGpuDouble(Tensor<double> input, int poolSize, int stride, int padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4)
        {
            throw new ArgumentException($"MaxPool2D requires a 4D tensor. Got rank {input.Rank}.");
        }

        if (stride == 0) stride = poolSize;

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outputHeight = (height + 2 * padding - poolSize) / stride + 1;
        int outputWidth = (width + 2 * padding - poolSize) / stride + 1;

        try
        {
            var result = new Tensor<double>(new[] { batch, channels, outputHeight, outputWidth });
            int outputSize = batch * channels * outputHeight * outputWidth;

            var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
            var gpuOutput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(outputSize);

            try
            {
                gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());

                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_maxPool2DKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, outputSize, gpuInput.View, gpuOutput.View,
                        batch, channels, height, width, outputHeight, outputWidth, poolSize, stride, padding);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuOutput.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuInput);
                _memoryPoolDouble.Return(gpuOutput);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU max pool 2D (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.MaxPool2D(input, poolSize, stride, padding);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (input.Length < _thresholds.Pooling)
        {
            return _cpuFallback.AvgPool2D(input, poolSize, stride, padding);
        }

        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)AvgPool2DGpu((Tensor<float>)(object)input, poolSize, stride, padding);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)AvgPool2DGpuDouble((Tensor<double>)(object)input, poolSize, stride, padding);
        }

        return _cpuFallback.AvgPool2D(input, poolSize, stride, padding);
    }

    private Tensor<float> AvgPool2DGpu(Tensor<float> input, int poolSize, int stride, int padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4)
        {
            throw new ArgumentException($"AvgPool2D requires a 4D tensor. Got rank {input.Rank}.");
        }

        if (stride == 0) stride = poolSize;

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outputHeight = (height + 2 * padding - poolSize) / stride + 1;
        int outputWidth = (width + 2 * padding - poolSize) / stride + 1;

        try
        {
            var result = new Tensor<float>(new[] { batch, channels, outputHeight, outputWidth });
            int outputSize = batch * channels * outputHeight * outputWidth;

            var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
            var gpuOutput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(outputSize);

            try
            {
                gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());

                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_avgPool2DKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, outputSize, gpuInput.View, gpuOutput.View,
                        batch, channels, height, width, outputHeight, outputWidth, poolSize, stride, padding);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuOutput.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuInput);
                _memoryPoolFloat.Return(gpuOutput);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU avg pool 2D failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.AvgPool2D(input, poolSize, stride, padding);
        }
    }

    private Tensor<double> AvgPool2DGpuDouble(Tensor<double> input, int poolSize, int stride, int padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4)
        {
            throw new ArgumentException($"AvgPool2D requires a 4D tensor. Got rank {input.Rank}.");
        }

        if (stride == 0) stride = poolSize;

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outputHeight = (height + 2 * padding - poolSize) / stride + 1;
        int outputWidth = (width + 2 * padding - poolSize) / stride + 1;

        try
        {
            var result = new Tensor<double>(new[] { batch, channels, outputHeight, outputWidth });
            int outputSize = batch * channels * outputHeight * outputWidth;

            var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
            var gpuOutput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(outputSize);

            try
            {
                gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());

                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    (_avgPool2DKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))((_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream, outputSize, gpuInput.View, gpuOutput.View,
                        batch, channels, height, width, outputHeight, outputWidth, poolSize, stride, padding);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuOutput.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuInput);
                _memoryPoolDouble.Return(gpuOutput);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU avg pool 2D (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.AvgPool2D(input, poolSize, stride, padding);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1)
    {
        // Adaptive execution: use convolution threshold (Phase B: US-GPU-004)
        if (input.Length < _thresholds.Convolution)
        {
            return _cpuFallback.Conv2D(input, kernel, stride, padding, dilation);
        }

        // Check GPU health and type support (Phase B: US-GPU-006)
        if (SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)Conv2DGpu((Tensor<float>)(object)input, (Tensor<float>)(object)kernel, stride, padding, dilation);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)Conv2DGpuDouble((Tensor<double>)(object)input, (Tensor<double>)(object)kernel, stride, padding, dilation);
        }

        return _cpuFallback.Conv2D(input, kernel, stride, padding, dilation);
    }

    private Tensor<float> Conv2DGpu(Tensor<float> input, Tensor<float> kernel, int stride, int padding, int dilation)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 4 || kernel.Rank != 4)
        {
            throw new ArgumentException($"Conv2D requires 4D tensors. Got input rank {input.Rank}, kernel rank {kernel.Rank}.");
        }

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outChannels = kernel.Shape[0];
        int kernelHeight = kernel.Shape[2];
        int kernelWidth = kernel.Shape[3];

        int effectiveKernelHeight = dilation * (kernelHeight - 1) + 1;
        int effectiveKernelWidth = dilation * (kernelWidth - 1) + 1;

        int outputHeight = (height + 2 * padding - effectiveKernelHeight) / stride + 1;
        int outputWidth = (width + 2 * padding - effectiveKernelWidth) / stride + 1;

        try
        {
            var result = new Tensor<float>(new[] { batch, outChannels, outputHeight, outputWidth });
            int outputSize = batch * outChannels * outputHeight * outputWidth;

            var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
            var gpuKernel = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(kernel.Length);
            var gpuOutput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(outputSize);

            try
            {
                gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());
                gpuKernel.View.BaseView.CopyFromCPU(kernel.AsSpan());

                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    var parameters = new Conv2DParams(batch, inChannels, height, width, outChannels,
                        outputHeight, outputWidth, kernelHeight, kernelWidth, stride, padding, dilation);
                    (_conv2DKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))(
                        (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                        outputSize, gpuInput.View, gpuKernel.View, gpuOutput.View, parameters);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuOutput.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolFloat.Return(gpuInput);
                _memoryPoolFloat.Return(gpuKernel);
                _memoryPoolFloat.Return(gpuOutput);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU Conv2D failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.Conv2D(input, kernel, stride, padding, dilation);
        }
    }

    private Tensor<double> Conv2DGpuDouble(Tensor<double> input, Tensor<double> kernel, int stride, int padding, int dilation)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 4 || kernel.Rank != 4)
        {
            throw new ArgumentException($"Conv2D requires 4D tensors. Got input rank {input.Rank}, kernel rank {kernel.Rank}.");
        }

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outChannels = kernel.Shape[0];
        int kernelHeight = kernel.Shape[2];
        int kernelWidth = kernel.Shape[3];

        int effectiveKernelHeight = dilation * (kernelHeight - 1) + 1;
        int effectiveKernelWidth = dilation * (kernelWidth - 1) + 1;

        int outputHeight = (height + 2 * padding - effectiveKernelHeight) / stride + 1;
        int outputWidth = (width + 2 * padding - effectiveKernelWidth) / stride + 1;

        try
        {
            var result = new Tensor<double>(new[] { batch, outChannels, outputHeight, outputWidth });
            int outputSize = batch * outChannels * outputHeight * outputWidth;

            var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
            var gpuKernel = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(kernel.Length);
            var gpuOutput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(outputSize);

            try
            {
                gpuInput.View.BaseView.CopyFromCPU(input.AsSpan());
                gpuKernel.View.BaseView.CopyFromCPU(kernel.AsSpan());

                // Thread-safe kernel execution (Phase B: US-GPU-019)
                lock (_gpuLock)
                {
                    var parameters = new Conv2DParams(batch, inChannels, height, width, outChannels,
                        outputHeight, outputWidth, kernelHeight, kernelWidth, stride, padding, dilation);
                    (_conv2DKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))(
                        (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                        outputSize, gpuInput.View, gpuKernel.View, gpuOutput.View, parameters);
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                gpuOutput.View.BaseView.CopyToCPU(result.AsWritableSpan());
                return result;
            }
            finally
            {
                _memoryPoolDouble.Return(gpuInput);
                _memoryPoolDouble.Return(gpuKernel);
                _memoryPoolDouble.Return(gpuOutput);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
        {
            Console.WriteLine($"[GpuEngine] GPU Conv2D (double) failed: {ex.Message}. Falling back to CPU.");
            return _cpuFallback.Conv2D(input, kernel, stride, padding, dilation);
        }
    }

    #endregion

    /// <summary>
    /// Disposes GPU resources.
    /// </summary>

    #region GPU Health Monitoring and Recovery (Phase B: US-GPU-020)

    /// <summary>
    /// Records a GPU failure and determines if recovery should be attempted.
    /// </summary>
    /// <param name="exception">The exception that caused the failure.</param>
    /// <returns>True if the GPU is now marked unhealthy.</returns>
    private bool RecordGpuFailure(Exception exception)
    {
        lock (_recoveryLock)
        {
            _consecutiveFailures++;
            Interlocked.Exchange(ref _lastFailureTimeTicks, DateTime.UtcNow.Ticks);

            Console.WriteLine($"[GpuEngine] GPU failure #{_consecutiveFailures}: {exception.Message}");

            // If we've exceeded maximum recovery attempts, permanently disable GPU
            if (_consecutiveFailures >= MaxRecoveryAttempts)
            {
                RecordGpuFailure(exception);
                return true;
            }

            // Temporarily mark unhealthy but allow recovery attempts
            Console.WriteLine($"[GpuEngine] GPU temporarily disabled. Recovery attempt {_consecutiveFailures}/{MaxRecoveryAttempts} will be tried after backoff period.");
            return false;
        }
    }

    /// <summary>
    /// Attempts to recover GPU health after a failure.
    /// </summary>
    /// <returns>True if GPU recovery succeeded.</returns>
    private bool AttemptGpuRecovery()
    {
        lock (_recoveryLock)
        {
            // If GPU is permanently disabled, don't attempt recovery
            if (!_gpuHealthy)
                return false;

            // Check if we're in backoff period
            var lastFailureTicks = Interlocked.Read(ref _lastFailureTimeTicks);
            var timeSinceFailure = DateTime.UtcNow - new DateTime(lastFailureTicks);
            if (timeSinceFailure < RecoveryBackoffPeriod)
            {
                // Still in backoff period - don't attempt recovery yet
                return false;
            }

            // Check if accelerator is still responsive
            if (_accelerator == null)
            {
                Console.WriteLine("[GpuEngine] GPU accelerator is null - cannot recover.");
                _gpuHealthy = false;
                return false;
            }

            try
            {
                // Test if GPU is responsive with a simple operation
                lock (_gpuLock)
                {
                    // Try to synchronize - if this works, GPU is healthy again
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
                }

                // Recovery successful!
                _consecutiveFailures = 0;
                Interlocked.Exchange(ref _lastFailureTimeTicks, DateTime.MinValue.Ticks);
                Console.WriteLine("[GpuEngine] GPU recovery successful! GPU operations re-enabled.");
                return true;
            }
            catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or OutOfMemoryException or DllNotFoundException or PlatformNotSupportedException)
            {
                Console.WriteLine($"[GpuEngine] GPU recovery failed: {ex.Message}");
                RecordGpuFailure(ex);
                return false;
            }
        }
    }

    /// <summary>
    /// Gets diagnostic information about GPU health status.
    /// </summary>
    /// <returns>A string containing GPU health diagnostics.</returns>
    public string GetGpuHealthDiagnostics()
    {
        if (_accelerator == null)
            return "GPU Status: Not Available (no accelerator initialized)";

        var diagnostics = new System.Text.StringBuilder();
        diagnostics.AppendLine("GPU Health Diagnostics:");
        diagnostics.AppendLine($"  Healthy: {_gpuHealthy}");
        diagnostics.AppendLine($"  Consecutive Failures: {_consecutiveFailures}/{MaxRecoveryAttempts}");

        var lastFailureTicks = Interlocked.Read(ref _lastFailureTimeTicks);
        var lastFailureTime = new DateTime(lastFailureTicks);
        diagnostics.AppendLine($"  Last Failure: {(lastFailureTicks == DateTime.MinValue.Ticks ? "Never" : lastFailureTime.ToString("yyyy-MM-dd HH:mm:ss UTC"))}");

        if (lastFailureTicks != DateTime.MinValue.Ticks)
        {
            var timeSinceFailure = DateTime.UtcNow - lastFailureTime;
            diagnostics.AppendLine($"  Time Since Failure: {timeSinceFailure.TotalSeconds:F1}s");

            if (timeSinceFailure < RecoveryBackoffPeriod)
            {
                var timeUntilRecovery = RecoveryBackoffPeriod - timeSinceFailure;
                diagnostics.AppendLine($"  Recovery Available In: {timeUntilRecovery.TotalSeconds:F1}s");
            }
            else
            {
                diagnostics.AppendLine("  Recovery Available: Yes");
            }
        }

        diagnostics.AppendLine($"  Accelerator: {_accelerator.Name}");
        diagnostics.AppendLine($"  Memory: {_accelerator.MemorySize / (1024.0 * 1024.0 * 1024.0):F2} GB");

        return diagnostics.ToString();
    }

    /// <summary>
    /// Manually triggers a GPU health check and recovery attempt if needed.
    /// </summary>
    /// <returns>True if GPU is healthy after the check.</returns>
    public bool CheckAndRecoverGpuHealth()
    {
        if (_gpuHealthy)
            return true;

        // Attempt recovery
        return AttemptGpuRecovery();
    }

    #endregion

    #region Trigonometric Span Overloads

    /// <inheritdoc/>
    public void Sin(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorFloat>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            SinGpuFloat(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorFloat>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Sin(ReadOnlySpan<double> x, Span<double> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorDouble>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            SinGpuDouble(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorDouble>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Cos(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorFloat>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            CosGpuFloat(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorFloat>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Cos(ReadOnlySpan<double> x, Span<double> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorDouble>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            CosGpuDouble(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorDouble>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Tan(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorFloat>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            TanGpuFloat(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorFloat>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Tan(ReadOnlySpan<double> x, Span<double> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorDouble>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            TanGpuDouble(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorDouble>(x, destination);
        }
    }

    /// <inheritdoc/>
    public Vector<T> Asin<T>(Vector<T> vector)
    {
        return _cpuFallback.Asin(vector);
    }

    /// <inheritdoc/>
    public void Asin(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AsinOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Asin(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AsinOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public Vector<T> Acos<T>(Vector<T> vector)
    {
        return _cpuFallback.Acos(vector);
    }

    /// <inheritdoc/>
    public void Acos(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AcosOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Acos(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AcosOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public Vector<T> Atan<T>(Vector<T> vector)
    {
        return _cpuFallback.Atan(vector);
    }

    /// <inheritdoc/>
    public void Atan(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AtanOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Atan(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AtanOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Sqrt(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorFloat>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            SqrtGpuFloat(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorFloat>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Sqrt(ReadOnlySpan<double> x, Span<double> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorDouble>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            SqrtGpuDouble(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorDouble>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Abs(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorFloat>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            AbsGpuFloat(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorFloat>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Abs(ReadOnlySpan<double> x, Span<double> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorDouble>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            AbsGpuDouble(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorDouble>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Sinh(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorFloat>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            SinhGpuFloat(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorFloat>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Sinh(ReadOnlySpan<double> x, Span<double> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorDouble>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            SinhGpuDouble(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorDouble>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Cosh(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorFloat>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            CoshGpuFloat(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorFloat>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Cosh(ReadOnlySpan<double> x, Span<double> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorDouble>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            CoshGpuDouble(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorDouble>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Tanh(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorFloat>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            TanhGpuFloat(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorFloat>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Tanh(ReadOnlySpan<double> x, Span<double> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorDouble>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            TanhGpuDouble(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorDouble>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void Asinh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AsinhOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Asinh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AsinhOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Acosh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AcoshOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Acosh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AcoshOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Atanh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AtanhOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Atanh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AtanhOperatorDouble>(x, destination);
    }

    public void Exp(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorFloat>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            ExpGpuFloat(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorFloat>(x, destination);
        }
    }

    public void Exp(ReadOnlySpan<double> x, Span<double> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorDouble>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            ExpGpuDouble(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorDouble>(x, destination);
        }
    }

    public void Log(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorFloat>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            LogGpuFloat(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorFloat>(x, destination);
        }
    }

    public void Log(ReadOnlySpan<double> x, Span<double> destination)
    {
        if (x.Length < _thresholds.VectorSqrt)
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorDouble>(x, destination);
            return;
        }

        if (SupportsGpu && _gpuHealthy)
        {
            LogGpuDouble(x, destination);
        }
        else
        {
            TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorDouble>(x, destination);
        }
    }

    /// <inheritdoc/>
    public void ExpM1(ReadOnlySpan<float> x, Span<float> destination)
    {
        // For now, use CPU fallback. Future GPU implementation can use custom kernel.
        TensorPrimitivesCore.InvokeSpanIntoSpan<ExpM1OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void ExpM1(ReadOnlySpan<double> x, Span<double> destination)
    {
        // For now, use CPU fallback. Future GPU implementation can use custom kernel.
        TensorPrimitivesCore.InvokeSpanIntoSpan<ExpM1OperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Log1P(ReadOnlySpan<float> x, Span<float> destination)
    {
        // For now, use CPU fallback. Future GPU implementation can use custom kernel.
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log1POperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Log1P(ReadOnlySpan<double> x, Span<double> destination)
    {
        // For now, use CPU fallback. Future GPU implementation can use custom kernel.
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log1POperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public Vector<T> Exp2<T>(Vector<T> vector)
    {
        return _cpuFallback.Exp2(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Exp10<T>(Vector<T> vector)
    {
        return _cpuFallback.Exp10(vector);
    }

    /// <inheritdoc/>
    public void Reciprocal(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ReciprocalOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Reciprocal(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ReciprocalOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Cbrt(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CbrtOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Cbrt(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CbrtOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Log2(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log2OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Log2(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log2OperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Log10(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log10OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Log10(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log10OperatorDouble>(x, destination);
    }

    #endregion

    #region Extended Tensor Operations

    /// <inheritdoc/>
    public Tensor<T> TensorTranspose<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank != 2)
            throw new ArgumentException($"TensorTranspose requires a 2D tensor. Got rank {tensor.Rank}.");

        // GPU transpose for supported types and large enough tensors
        if (tensor.Length >= _thresholds.MatrixMultiply && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)TensorTransposeGpuFloat((Tensor<float>)(object)tensor);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)TensorTransposeGpuDouble((Tensor<double>)(object)tensor);
        }
        return _cpuFallback.TensorTranspose(tensor);
    }

    private Tensor<float> TensorTransposeGpuFloat(Tensor<float> tensor)
    {
        int rows = tensor.Shape[0];
        int cols = tensor.Shape[1];

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(tensor.Length);
        var gpuOutput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(tensor.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(tensor.ToArray());

            lock (_gpuLock)
            {
                (_tensorTransposeKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    new Index2D(rows, cols), gpuInput.View.BaseView, gpuOutput.View.BaseView, rows, cols);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            var resultData = new float[tensor.Length];
            gpuOutput.View.BaseView.CopyToCPU(resultData);
            return new Tensor<float>([cols, rows], new Vector<float>(resultData));
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.TensorTranspose(tensor);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuOutput);
        }
    }

    private Tensor<double> TensorTransposeGpuDouble(Tensor<double> tensor)
    {
        int rows = tensor.Shape[0];
        int cols = tensor.Shape[1];

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(tensor.Length);
        var gpuOutput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(tensor.Length);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(tensor.ToArray());

            lock (_gpuLock)
            {
                (_tensorTransposeKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    new Index2D(rows, cols), gpuInput.View.BaseView, gpuOutput.View.BaseView, rows, cols);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            var resultData = new double[tensor.Length];
            gpuOutput.View.BaseView.CopyToCPU(resultData);
            return new Tensor<double>([cols, rows], new Vector<double>(resultData));
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.TensorTranspose(tensor);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuOutput);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMatMul<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException($"TensorMatMul requires 2D tensors. Got ranks {a.Rank} and {b.Rank}.");

        int m = a.Shape[0];
        int n = a.Shape[1];
        int p = b.Shape[1];

        if (n != b.Shape[0])
            throw new ArgumentException($"Matrix dimensions incompatible: [{m},{n}] x [{b.Shape[0]},{p}]");

        // GPU matrix multiplication for supported types and large enough operations
        int totalOps = m * n * p;
        if (totalOps >= _thresholds.MatrixMultiply && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)TensorMatMulGpuFloat((Tensor<float>)(object)a, (Tensor<float>)(object)b);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)TensorMatMulGpuDouble((Tensor<double>)(object)a, (Tensor<double>)(object)b);
        }
        return _cpuFallback.TensorMatMul(a, b);
    }

    private Tensor<float> TensorMatMulGpuFloat(Tensor<float> a, Tensor<float> b)
    {
        int m = a.Shape[0];
        int k = a.Shape[1];
        int n = b.Shape[1];

        var gpuA = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(m * k);
        var gpuB = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(k * n);
        var gpuResult = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(m * n);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.ToArray());
            gpuB.View.BaseView.CopyFromCPU(b.ToArray());

            // Create 2D views for GEMM
            var viewA = gpuA.View.As2DView<Stride2D.DenseX>(new Index2D(m, k), new Stride2D.DenseX(k));
            var viewB = gpuB.View.As2DView<Stride2D.DenseX>(new Index2D(k, n), new Stride2D.DenseX(n));
            var viewResult = gpuResult.View.As2DView<Stride2D.DenseX>(new Index2D(m, n), new Stride2D.DenseX(n));

            lock (_gpuLock)
            {
                // Use existing matrix multiply kernel (already optimized)
                (_matrixMultiplyKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    new Index2D(m, n), viewA, viewB, viewResult, k);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            var resultData = new float[m * n];
            gpuResult.View.BaseView.CopyToCPU(resultData);
            return new Tensor<float>([m, n], new Vector<float>(resultData));
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.TensorMatMul(a, b);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuA);
            _memoryPoolFloat.Return(gpuB);
            _memoryPoolFloat.Return(gpuResult);
        }
    }

    private Tensor<double> TensorMatMulGpuDouble(Tensor<double> a, Tensor<double> b)
    {
        int m = a.Shape[0];
        int k = a.Shape[1];
        int n = b.Shape[1];

        var gpuA = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(m * k);
        var gpuB = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(k * n);
        var gpuResult = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(m * n);

        try
        {
            gpuA.View.BaseView.CopyFromCPU(a.ToArray());
            gpuB.View.BaseView.CopyFromCPU(b.ToArray());

            var viewA = gpuA.View.As2DView<Stride2D.DenseX>(new Index2D(m, k), new Stride2D.DenseX(k));
            var viewB = gpuB.View.As2DView<Stride2D.DenseX>(new Index2D(k, n), new Stride2D.DenseX(n));
            var viewResult = gpuResult.View.As2DView<Stride2D.DenseX>(new Index2D(m, n), new Stride2D.DenseX(n));

            lock (_gpuLock)
            {
                (_matrixMultiplyKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    new Index2D(m, n), viewA, viewB, viewResult, k);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            var resultData = new double[m * n];
            gpuResult.View.BaseView.CopyToCPU(resultData);
            return new Tensor<double>([m, n], new Vector<double>(resultData));
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.TensorMatMul(a, b);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuA);
            _memoryPoolDouble.Return(gpuB);
            _memoryPoolDouble.Return(gpuResult);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] dilation)
    {
        // GPU Conv2D with asymmetric parameters
        // For now use CPU, can extend existing Conv2D GPU kernel
        return _cpuFallback.Conv2D(input, kernel, stride, padding, dilation);
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding, int[] dilation)
    {
        return _cpuFallback.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding, int[] dilation)
    {
        return _cpuFallback.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool2DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,] maxIndices)
    {
        return _cpuFallback.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool2DBackward<T>(Tensor<T> gradOutput, int[,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride)
    {
        return _cpuFallback.MaxPool2DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool2D<T>(Tensor<T> input, int[] poolSize, int[] stride)
    {
        return _cpuFallback.AvgPool2D(input, poolSize, stride);
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool2DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride)
    {
        return _cpuFallback.AvgPool2DBackward(gradOutput, inputShape, poolSize, stride);
    }

    /// <inheritdoc/>
    public Tensor<T> DepthwiseConv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding)
    {
        return _cpuFallback.DepthwiseConv2D(input, kernel, stride, padding);
    }

    /// <inheritdoc/>
    public Tensor<T> DepthwiseConv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding)
    {
        return _cpuFallback.DepthwiseConv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding);
    }

    /// <inheritdoc/>
    public Tensor<T> DepthwiseConv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding)
    {
        return _cpuFallback.DepthwiseConv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding);
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] outputPadding)
    {
        return _cpuFallback.ConvTranspose2D(input, kernel, stride, padding, outputPadding);
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding)
    {
        return _cpuFallback.ConvTranspose2DBackwardInput(gradOutput, kernel, inputShape, stride, padding);
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding)
    {
        return _cpuFallback.ConvTranspose2DBackwardKernel(gradOutput, input, kernelShape, stride, padding);
    }

    #endregion

    #region Normalization and Activation Operations (Extended)

    /// <inheritdoc/>
    public Tensor<T> Softmax<T>(Tensor<T> input, int axis = -1)
    {
        // GPU softmax along axis - can be optimized with block-level reductions
        return _cpuFallback.Softmax(input, axis);
    }

    /// <inheritdoc/>
    public Tensor<T> SoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis = -1)
    {
        return _cpuFallback.SoftmaxBackward(gradOutput, output, axis);
    }

    /// <inheritdoc/>
    public Tensor<T> BatchNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        return _cpuFallback.BatchNorm(input, gamma, beta, epsilon, out mean, out variance);
    }

    /// <inheritdoc/>
    public Tensor<T> BatchNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        return _cpuFallback.BatchNormBackward(gradOutput, input, gamma, mean, variance, epsilon, out gradGamma, out gradBeta);
    }

    /// <inheritdoc/>
    public Tensor<T> LayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        return _cpuFallback.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);
    }

    /// <inheritdoc/>
    public Tensor<T> LayerNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        return _cpuFallback.LayerNormBackward(gradOutput, input, gamma, mean, variance, epsilon, out gradGamma, out gradBeta);
    }

    #endregion

    #region Tensor Reduction Operations

    /// <inheritdoc/>
    public Tensor<T> ReduceMax<T>(Tensor<T> input, int[] axes, bool keepDims, out int[] maxIndices)
    {
        return _cpuFallback.ReduceMax(input, axes, keepDims, out maxIndices);
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceMaxBackward<T>(Tensor<T> gradOutput, int[] maxIndices, int[] inputShape)
    {
        return _cpuFallback.ReduceMaxBackward(gradOutput, maxIndices, inputShape);
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceMean<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        return _cpuFallback.ReduceMean(input, axes, keepDims);
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceMeanBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] axes)
    {
        return _cpuFallback.ReduceMeanBackward(gradOutput, inputShape, axes);
    }

    #endregion

    #region Spatial Operations

    /// <inheritdoc/>
    public Tensor<T> Upsample<T>(Tensor<T> input, int scaleH, int scaleW)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        var shape = input.Shape;
        if (shape.Length != 4)
            throw new ArgumentException("Upsample expects 4D tensor [batch, channels, height, width]");

        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        int outputSize = batch * channels * (height * scaleH) * (width * scaleW);

        // GPU upsample for supported types and large enough tensors
        if (outputSize >= _thresholds.MatrixMultiply && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)UpsampleGpuFloat((Tensor<float>)(object)input, scaleH, scaleW);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)UpsampleGpuDouble((Tensor<double>)(object)input, scaleH, scaleW);
        }
        return _cpuFallback.Upsample(input, scaleH, scaleW);
    }

    private Tensor<float> UpsampleGpuFloat(Tensor<float> input, int scaleH, int scaleW)
    {
        var shape = input.Shape;
        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        int newHeight = height * scaleH;
        int newWidth = width * scaleW;
        int outputSize = batch * channels * newHeight * newWidth;

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuOutput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(outputSize);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.ToArray());

            lock (_gpuLock)
            {
                (_upsampleKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    outputSize, gpuInput.View.BaseView, gpuOutput.View.BaseView,
                    batch, channels, height, width, scaleH, scaleW);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            var resultData = new float[outputSize];
            gpuOutput.View.BaseView.CopyToCPU(resultData);
            return new Tensor<float>([batch, channels, newHeight, newWidth], new Vector<float>(resultData));
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Upsample(input, scaleH, scaleW);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuOutput);
        }
    }

    private Tensor<double> UpsampleGpuDouble(Tensor<double> input, int scaleH, int scaleW)
    {
        var shape = input.Shape;
        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        int newHeight = height * scaleH;
        int newWidth = width * scaleW;
        int outputSize = batch * channels * newHeight * newWidth;

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuOutput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(outputSize);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.ToArray());

            lock (_gpuLock)
            {
                (_upsampleKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    outputSize, gpuInput.View.BaseView, gpuOutput.View.BaseView,
                    batch, channels, height, width, scaleH, scaleW);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            var resultData = new double[outputSize];
            gpuOutput.View.BaseView.CopyToCPU(resultData);
            return new Tensor<double>([batch, channels, newHeight, newWidth], new Vector<double>(resultData));
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.Upsample(input, scaleH, scaleW);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuOutput);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> UpsampleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int scaleH, int scaleW)
    {
        // Backward pass requires atomic operations or reduction - CPU fallback for now
        return _cpuFallback.UpsampleBackward(gradOutput, inputShape, scaleH, scaleW);
    }

    /// <inheritdoc/>
    public Tensor<T> PixelShuffle<T>(Tensor<T> input, int upscaleFactor)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        var shape = input.Shape;
        if (shape.Length != 4)
            throw new ArgumentException("PixelShuffle expects 4D tensor [batch, channels, height, width]");

        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        int r = upscaleFactor;

        if (channels % (r * r) != 0)
            throw new ArgumentException($"Number of channels ({channels}) must be divisible by r^2 ({r * r})");

        int outputSize = batch * (channels / (r * r)) * (height * r) * (width * r);

        // GPU pixel shuffle for supported types and large enough tensors
        if (outputSize >= _thresholds.MatrixMultiply && SupportsGpu && _gpuHealthy)
        {
            if (typeof(T) == typeof(float))
                return (Tensor<T>)(object)PixelShuffleGpuFloat((Tensor<float>)(object)input, upscaleFactor);
            if (typeof(T) == typeof(double))
                return (Tensor<T>)(object)PixelShuffleGpuDouble((Tensor<double>)(object)input, upscaleFactor);
        }
        return _cpuFallback.PixelShuffle(input, upscaleFactor);
    }

    private Tensor<float> PixelShuffleGpuFloat(Tensor<float> input, int upscaleFactor)
    {
        var shape = input.Shape;
        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        int r = upscaleFactor;
        int newChannels = channels / (r * r);
        int newHeight = height * r;
        int newWidth = width * r;
        int outputSize = batch * newChannels * newHeight * newWidth;

        var gpuInput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuOutput = (_memoryPoolFloat ?? throw new InvalidOperationException("GPU not initialized")).Rent(outputSize);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.ToArray());

            lock (_gpuLock)
            {
                (_pixelShuffleKernelFloat ?? throw new InvalidOperationException("Kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    outputSize, gpuInput.View.BaseView, gpuOutput.View.BaseView,
                    batch, channels, height, width, upscaleFactor);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            var resultData = new float[outputSize];
            gpuOutput.View.BaseView.CopyToCPU(resultData);
            return new Tensor<float>([batch, newChannels, newHeight, newWidth], new Vector<float>(resultData));
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.PixelShuffle(input, upscaleFactor);
        }
        finally
        {
            _memoryPoolFloat.Return(gpuInput);
            _memoryPoolFloat.Return(gpuOutput);
        }
    }

    private Tensor<double> PixelShuffleGpuDouble(Tensor<double> input, int upscaleFactor)
    {
        var shape = input.Shape;
        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        int r = upscaleFactor;
        int newChannels = channels / (r * r);
        int newHeight = height * r;
        int newWidth = width * r;
        int outputSize = batch * newChannels * newHeight * newWidth;

        var gpuInput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(input.Length);
        var gpuOutput = (_memoryPoolDouble ?? throw new InvalidOperationException("GPU not initialized")).Rent(outputSize);

        try
        {
            gpuInput.View.BaseView.CopyFromCPU(input.ToArray());

            lock (_gpuLock)
            {
                (_pixelShuffleKernelDouble ?? throw new InvalidOperationException("Kernel not initialized"))(
                    (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).DefaultStream,
                    outputSize, gpuInput.View.BaseView, gpuOutput.View.BaseView,
                    batch, channels, height, width, upscaleFactor);
                (_accelerator ?? throw new InvalidOperationException("GPU not initialized")).Synchronize();
            }

            var resultData = new double[outputSize];
            gpuOutput.View.BaseView.CopyToCPU(resultData);
            return new Tensor<double>([batch, newChannels, newHeight, newWidth], new Vector<double>(resultData));
        }
        catch (Exception ex) when (ex.Message.Contains("device") || ex.Message.Contains("accelerator"))
        {
            RecordGpuFailure(ex);
            return _cpuFallback.PixelShuffle(input, upscaleFactor);
        }
        finally
        {
            _memoryPoolDouble.Return(gpuInput);
            _memoryPoolDouble.Return(gpuOutput);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> PixelShuffleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int upscaleFactor)
    {
        // Backward pass is essentially the inverse rearrangement - CPU fallback for now
        return _cpuFallback.PixelShuffleBackward(gradOutput, inputShape, upscaleFactor);
    }

    /// <inheritdoc/>
    public Tensor<T> Crop<T>(Tensor<T> input, int top, int left, int height, int width)
    {
        return _cpuFallback.Crop(input, top, left, height, width);
    }

    /// <inheritdoc/>
    public Tensor<T> CropBackward<T>(Tensor<T> gradOutput, int[] inputShape, int top, int left)
    {
        return _cpuFallback.CropBackward(gradOutput, inputShape, top, left);
    }

    /// <inheritdoc/>
    public Tensor<T> Pad<T>(Tensor<T> input, int padTop, int padBottom, int padLeft, int padRight, T padValue)
    {
        return _cpuFallback.Pad(input, padTop, padBottom, padLeft, padRight, padValue);
    }

    /// <inheritdoc/>
    public Tensor<T> PadBackward<T>(Tensor<T> gradOutput, int padTop, int padLeft, int[] inputShape)
    {
        return _cpuFallback.PadBackward(gradOutput, padTop, padLeft, inputShape);
    }

    /// <inheritdoc/>
    public Tensor<T> Concat<T>(IReadOnlyList<Tensor<T>> tensors, int axis)
    {
        return _cpuFallback.Concat(tensors, axis);
    }

    #endregion

    #region IDisposable

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

    #endregion
}
#endif
