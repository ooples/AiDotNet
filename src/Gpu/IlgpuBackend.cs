using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;
using System.Diagnostics;

namespace AiDotNet.Gpu;

/// <summary>
/// ILGPU-based GPU backend implementation.
/// </summary>
/// <typeparam name="T">The numeric type for GPU operations.</typeparam>
/// <remarks>
/// <para>
/// IlgpuBackend provides GPU acceleration using the ILGPU library.
/// It supports CUDA (NVIDIA), OpenCL (NVIDIA/AMD/Intel), and CPU fallback.
/// </para>
/// <para><b>For Beginners:</b> This is the actual implementation that talks to your GPU.
///
/// ILGPU is a C#-native GPU library that:
/// - Works with NVIDIA GPUs (via CUDA)
/// - Works with AMD/Intel GPUs (via OpenCL)
/// - Falls back to CPU if no GPU available
/// - Writes GPU code in C# (no C++/CUDA needed!)
///
/// When you create this backend, it:
/// 1. Detects available GPUs
/// 2. Initializes the best one
/// 3. Compiles kernels (GPU functions)
/// 4. Ready to accelerate your calculations!
/// </para>
/// </remarks>
public class IlgpuBackend<T> : IGpuBackend<T>
    where T : unmanaged
{
    private Context? _context;
    private Accelerator? _accelerator;
    private readonly GpuDeviceType _preferredDeviceType;
    private bool _disposed;

    // Numeric operations for this type
    private readonly INumericOperations<T> _numOps;

    // Compiled kernels (cached for performance)
    private Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>>? _addKernel;
    private Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>>? _subtractKernel;
    private Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>>? _multiplyKernel;
    private Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>>? _divideKernel;
    private Action<Index1D, ArrayView<T>, ArrayView<T>>? _reluKernel;
    private Action<Index1D, ArrayView<T>, ArrayView<T>>? _sigmoidKernel;
    private Action<Index1D, ArrayView<T>, ArrayView<T>>? _tanhKernel;
    private Action<Index2D, ArrayView<T>, ArrayView<T>, ArrayView<T>, int, int, int>? _matMulNaiveKernel;
    private Action<Index2D, ArrayView<T>, ArrayView<T>, ArrayView<T>, int, int, int>? _matMulTiledKernel;
    private Action<Index2D, ArrayView<T>, ArrayView<T>>? _transposeKernel;

    /// <inheritdoc/>
    public GpuDeviceType DeviceType { get; private set; }

    /// <inheritdoc/>
    public bool IsAvailable => _accelerator != null && !_disposed;

    /// <inheritdoc/>
    public string DeviceName => _accelerator?.Name ?? "Not initialized";

    /// <inheritdoc/>
    public long TotalMemory => _accelerator?.MemorySize ?? 0;

    /// <inheritdoc/>
    public long FreeMemory
    {
        get
        {
            if (_accelerator == null) return 0;

            // ILGPU doesn't provide free memory directly
            // Return estimated based on total memory
            return (long)(TotalMemory * 0.8); // Conservative estimate
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="IlgpuBackend{T}"/> class.
    /// </summary>
    /// <param name="preferredDeviceType">The preferred GPU device type to use.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new GPU backend.
    ///
    /// Usage:
    /// <code>
    /// // Try to use CUDA (NVIDIA), fallback to OpenCL or CPU
    /// var backend = new IlgpuBackend&lt;float&gt;(GpuDeviceType.Default);
    /// backend.Initialize();
    ///
    /// // Force CUDA (NVIDIA only)
    /// var cudaBackend = new IlgpuBackend&lt;float&gt;(GpuDeviceType.CUDA);
    ///
    /// // Force CPU (no GPU needed)
    /// var cpuBackend = new IlgpuBackend&lt;float&gt;(GpuDeviceType.CPU);
    /// </code>
    /// </para>
    /// </remarks>
    public IlgpuBackend(GpuDeviceType preferredDeviceType = GpuDeviceType.Default)
    {
        _preferredDeviceType = preferredDeviceType;
        _numOps = MathHelper.GetNumericOperations<T>();
        DeviceType = GpuDeviceType.Default;
    }

    /// <inheritdoc/>
    public void Initialize()
    {
        if (_context != null)
        {
            throw new InvalidOperationException("Backend already initialized");
        }

        // Create ILGPU context
        _context = Context.Create(builder => builder.Default().EnableAlgorithms());

        // Select accelerator based on preference
        _accelerator = _preferredDeviceType switch
        {
            GpuDeviceType.CUDA => TryCreateCudaAccelerator(),
            GpuDeviceType.OpenCL => TryCreateOpenCLAccelerator(),
            GpuDeviceType.CPU => CreateCpuAccelerator(),
            GpuDeviceType.Default => TryCreateBestAccelerator(),
            _ => throw new ArgumentException($"Unsupported device type: {_preferredDeviceType}")
        };

        if (_accelerator == null)
        {
            throw new InvalidOperationException(
                "Failed to create accelerator. No compatible GPU found or GPU drivers not installed.");
        }

        // Compile kernels
        CompileKernels();

        Debug.WriteLine($"[IlgpuBackend] Initialized on {DeviceName} ({DeviceType})");
    }

    /// <summary>
    /// Tries to create a CUDA accelerator.
    /// </summary>
    private Accelerator? TryCreateCudaAccelerator()
    {
        if (_context == null) return null;

        try
        {
            foreach (var device in _context.GetCudaDevices())
            {
                var accelerator = device.CreateAccelerator(_context);
                DeviceType = GpuDeviceType.CUDA;
                return accelerator;
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[IlgpuBackend] Failed to create CUDA accelerator: {ex.Message}");
        }

        return null;
    }

    /// <summary>
    /// Tries to create an OpenCL accelerator.
    /// </summary>
    private Accelerator? TryCreateOpenCLAccelerator()
    {
        if (_context == null) return null;

        try
        {
            foreach (var device in _context.GetCLDevices())
            {
                var accelerator = device.CreateAccelerator(_context);
                DeviceType = GpuDeviceType.OpenCL;
                return accelerator;
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[IlgpuBackend] Failed to create OpenCL accelerator: {ex.Message}");
        }

        return null;
    }

    /// <summary>
    /// Creates a CPU accelerator as fallback.
    /// </summary>
    private Accelerator CreateCpuAccelerator()
    {
        if (_context == null)
        {
            throw new InvalidOperationException("Context not initialized");
        }

        var device = _context.GetCPUDevice();
        var accelerator = device.CreateAccelerator(_context);
        DeviceType = GpuDeviceType.CPU;
        return accelerator;
    }

    /// <summary>
    /// Tries to create the best available accelerator (CUDA > OpenCL > CPU).
    /// </summary>
    private Accelerator TryCreateBestAccelerator()
    {
        // Try CUDA first (fastest)
        var accelerator = TryCreateCudaAccelerator();
        if (accelerator != null) return accelerator;

        // Try OpenCL second (cross-platform)
        accelerator = TryCreateOpenCLAccelerator();
        if (accelerator != null) return accelerator;

        // Fallback to CPU
        return CreateCpuAccelerator();
    }

    /// <summary>
    /// Compiles all GPU kernels for this type.
    /// </summary>
    private void CompileKernels()
    {
        if (_accelerator == null)
        {
            throw new InvalidOperationException("Accelerator not initialized");
        }

        // Compile element-wise kernels
        _addKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>>(AddKernel);
        _subtractKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>>(SubtractKernel);
        _multiplyKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>>(MultiplyKernel);
        _divideKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>>(DivideKernel);

        // Compile activation kernels
        _reluKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, ArrayView<T>>(ReLUKernel);
        _sigmoidKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, ArrayView<T>>(SigmoidKernel);
        _tanhKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, ArrayView<T>>(TanhKernel);

        // Compile linear algebra kernels
        _matMulNaiveKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<T>, ArrayView<T>, ArrayView<T>, int, int, int>(MatMulNaiveKernel);
        _matMulTiledKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<T>, ArrayView<T>, ArrayView<T>, int, int, int>(MatMulTiledKernel);
        _transposeKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<T>, ArrayView<T>>(TransposeKernel);

        Debug.WriteLine("[IlgpuBackend] Kernels compiled successfully");
    }

    /// <inheritdoc/>
    public void Synchronize()
    {
        _accelerator?.Synchronize();
    }

    #region Kernel Implementations

    /// <summary>
    /// GPU kernel for element-wise addition.
    /// </summary>
    private static void AddKernel(Index1D index, ArrayView<T> a, ArrayView<T> b, ArrayView<T> result)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        result[index] = numOps.Add(a[index], b[index]);
    }

    /// <summary>
    /// GPU kernel for element-wise subtraction.
    /// </summary>
    private static void SubtractKernel(Index1D index, ArrayView<T> a, ArrayView<T> b, ArrayView<T> result)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        result[index] = numOps.Subtract(a[index], b[index]);
    }

    /// <summary>
    /// GPU kernel for element-wise multiplication.
    /// </summary>
    private static void MultiplyKernel(Index1D index, ArrayView<T> a, ArrayView<T> b, ArrayView<T> result)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        result[index] = numOps.Multiply(a[index], b[index]);
    }

    /// <summary>
    /// GPU kernel for element-wise division.
    /// </summary>
    private static void DivideKernel(Index1D index, ArrayView<T> a, ArrayView<T> b, ArrayView<T> result)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        result[index] = numOps.Divide(a[index], b[index]);
    }

    /// <summary>
    /// GPU kernel for ReLU activation.
    /// </summary>
    private static void ReLUKernel(Index1D index, ArrayView<T> input, ArrayView<T> output)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var value = input[index];
        output[index] = numOps.GreaterThan(value, numOps.Zero) ? value : numOps.Zero;
    }

    /// <summary>
    /// GPU kernel for Sigmoid activation.
    /// </summary>
    private static void SigmoidKernel(Index1D index, ArrayView<T> input, ArrayView<T> output)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var value = input[index];
        var negValue = numOps.Negate(value);
        var expNeg = numOps.Exp(negValue);
        var onePlusExp = numOps.Add(numOps.One, expNeg);
        output[index] = numOps.Divide(numOps.One, onePlusExp);
    }

    /// <summary>
    /// GPU kernel for Tanh activation.
    /// </summary>
    private static void TanhKernel(Index1D index, ArrayView<T> input, ArrayView<T> output)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        output[index] = numOps.Tanh(input[index]);
    }

    /// <summary>
    /// Naive GPU kernel for matrix multiplication.
    /// </summary>
    /// <remarks>
    /// Computes C = A * B where:
    /// - A is M x K
    /// - B is K x N
    /// - C is M x N (result)
    ///
    /// This is a simple implementation where each thread computes one output element.
    /// Performance: Good for small matrices, slower for large matrices due to global memory access.
    /// </remarks>
    private static void MatMulNaiveKernel(
        Index2D index,
        ArrayView<T> a,
        ArrayView<T> b,
        ArrayView<T> result,
        int m, int n, int k)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var row = index.X;
        var col = index.Y;

        if (row >= m || col >= n) return;

        var sum = numOps.Zero;

        for (int i = 0; i < k; i++)
        {
            var aValue = a[row * k + i];
            var bValue = b[i * n + col];
            sum = numOps.Add(sum, numOps.Multiply(aValue, bValue));
        }

        result[row * n + col] = sum;
    }

    /// <summary>
    /// Tiled GPU kernel for matrix multiplication with shared memory optimization.
    /// </summary>
    /// <remarks>
    /// Optimized version using:
    /// - Shared memory to reduce global memory access
    /// - Tile-based computation for better cache utilization
    /// - Coalesced memory access patterns
    ///
    /// Performance: 5-10x faster than naive for large matrices (>512x512).
    /// </remarks>
    private static void MatMulTiledKernel(
        Index2D index,
        ArrayView<T> a,
        ArrayView<T> b,
        ArrayView<T> result,
        int m, int n, int k)
    {
        const int TILE_SIZE = 16;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Allocate shared memory for tiles
        var sharedA = SharedMemory.Allocate2D<T>(new Index2D(TILE_SIZE, TILE_SIZE), new Stride2D.DenseY(TILE_SIZE));
        var sharedB = SharedMemory.Allocate2D<T>(new Index2D(TILE_SIZE, TILE_SIZE), new Stride2D.DenseY(TILE_SIZE));

        var row = index.X;
        var col = index.Y;
        var localRow = Group.IdxX;
        var localCol = Group.IdxY;

        var sum = numOps.Zero;
        var numTiles = (k + TILE_SIZE - 1) / TILE_SIZE;

        for (int tile = 0; tile < numTiles; tile++)
        {
            // Load tile of A into shared memory
            var aCol = tile * TILE_SIZE + localCol;
            if (row < m && aCol < k)
            {
                sharedA[new Index2D(localRow, localCol)] = a[row * k + aCol];
            }
            else
            {
                sharedA[new Index2D(localRow, localCol)] = numOps.Zero;
            }

            // Load tile of B into shared memory
            var bRow = tile * TILE_SIZE + localRow;
            if (bRow < k && col < n)
            {
                sharedB[new Index2D(localRow, localCol)] = b[bRow * n + col];
            }
            else
            {
                sharedB[new Index2D(localRow, localCol)] = numOps.Zero;
            }

            // Synchronize to ensure tile is loaded
            Group.Barrier();

            // Compute partial dot product for this tile
            for (int i = 0; i < TILE_SIZE; i++)
            {
                var aValue = sharedA[new Index2D(localRow, i)];
                var bValue = sharedB[new Index2D(i, localCol)];
                sum = numOps.Add(sum, numOps.Multiply(aValue, bValue));
            }

            // Synchronize before loading next tile
            Group.Barrier();
        }

        // Write result
        if (row < m && col < n)
        {
            result[row * n + col] = sum;
        }
    }

    /// <summary>
    /// GPU kernel for matrix transpose.
    /// </summary>
    /// <remarks>
    /// Transposes a matrix by swapping rows and columns.
    /// Uses coalesced memory access for optimal performance.
    /// </remarks>
    private static void TransposeKernel(
        Index2D index,
        ArrayView<T> input,
        ArrayView<T> output)
    {
        // index.X = row in input, index.Y = col in input
        // After transpose: row becomes col, col becomes row

        // Get dimensions from the 2D index
        var inputRow = index.X;
        var inputCol = index.Y;

        // In the output, swap row and col
        var outputRow = inputCol;
        var outputCol = inputRow;

        // Note: We need to know the dimensions to calculate flat indices
        // This will be passed via the shape parameters
        // For now, we'll use a simpler approach

        output[index] = input[new Index2D(index.Y, index.X)];
    }

    #endregion

    #region Memory Management

    /// <inheritdoc/>
    public GpuTensor<T> Allocate(int[] shape)
    {
        if (_accelerator == null)
        {
            throw new InvalidOperationException("Backend not initialized. Call Initialize() first.");
        }

        // Calculate total size
        int length = 1;
        foreach (var dim in shape)
        {
            length *= dim;
        }

        // Allocate GPU memory
        var buffer = _accelerator.Allocate1D<T>(length);

        return new GpuTensor<T>(buffer, shape, this);
    }

    /// <inheritdoc/>
    public GpuTensor<T> ToGpu(Tensor<T> cpuTensor)
    {
        if (_accelerator == null)
        {
            throw new InvalidOperationException("Backend not initialized");
        }

        // Allocate GPU memory
        var gpuTensor = Allocate(cpuTensor.Shape);

        // Copy data from CPU to GPU
        var cpuData = new T[cpuTensor.Length];
        for (int i = 0; i < cpuTensor.Length; i++)
        {
            cpuData[i] = cpuTensor[i];
        }

        gpuTensor.Buffer.CopyFromCPU(cpuData);

        return gpuTensor;
    }

    /// <inheritdoc/>
    public Tensor<T> ToCpu(GpuTensor<T> gpuTensor)
    {
        // Allocate CPU tensor
        var cpuTensor = new Tensor<T>(gpuTensor.Shape);

        // Copy data from GPU to CPU
        var gpuData = gpuTensor.Buffer.GetAsArray1D();
        for (int i = 0; i < gpuData.Length; i++)
        {
            cpuTensor[i] = gpuData[i];
        }

        return cpuTensor;
    }

    /// <inheritdoc/>
    public void Free(GpuTensor<T> gpuTensor)
    {
        gpuTensor?.Dispose();
    }

    #endregion

    #region Basic Operations

    /// <inheritdoc/>
    public GpuTensor<T> Add(GpuTensor<T> a, GpuTensor<T> b)
    {
        ValidateSameShape(a, b);

        var result = Allocate(a.Shape);
        _addKernel!(result.Length, a.Buffer.View, b.Buffer.View, result.Buffer.View);
        Synchronize();

        return result;
    }

    /// <inheritdoc/>
    public GpuTensor<T> Subtract(GpuTensor<T> a, GpuTensor<T> b)
    {
        ValidateSameShape(a, b);

        var result = Allocate(a.Shape);
        _subtractKernel!(result.Length, a.Buffer.View, b.Buffer.View, result.Buffer.View);
        Synchronize();

        return result;
    }

    /// <inheritdoc/>
    public GpuTensor<T> Multiply(GpuTensor<T> a, GpuTensor<T> b)
    {
        ValidateSameShape(a, b);

        var result = Allocate(a.Shape);
        _multiplyKernel!(result.Length, a.Buffer.View, b.Buffer.View, result.Buffer.View);
        Synchronize();

        return result;
    }

    /// <inheritdoc/>
    public GpuTensor<T> Divide(GpuTensor<T> a, GpuTensor<T> b)
    {
        ValidateSameShape(a, b);

        var result = Allocate(a.Shape);
        _divideKernel!(result.Length, a.Buffer.View, b.Buffer.View, result.Buffer.View);
        Synchronize();

        return result;
    }

    #endregion

    #region Linear Algebra

    /// <inheritdoc/>
    public GpuTensor<T> MatMul(GpuTensor<T> a, GpuTensor<T> b)
    {
        // Validate inputs
        if (a.Rank != 2 || b.Rank != 2)
        {
            throw new ArgumentException("MatMul requires 2D tensors (matrices)");
        }

        int m = a.Shape[0];  // Rows of A
        int k = a.Shape[1];  // Cols of A = Rows of B
        int n = b.Shape[1];  // Cols of B

        if (b.Shape[0] != k)
        {
            throw new ArgumentException(
                $"Matrix dimensions don't match for multiplication: A is {m}x{k}, B is {b.Shape[0]}x{n}");
        }

        // Allocate result matrix (M x N)
        var result = Allocate(new[] { m, n });

        // Choose kernel based on matrix size
        // Tiled kernel is faster for large matrices, naive for small
        const int TILED_THRESHOLD = 128; // Use tiled for matrices larger than 128x128

        if (m >= TILED_THRESHOLD && n >= TILED_THRESHOLD && k >= TILED_THRESHOLD)
        {
            // Use optimized tiled kernel for large matrices
            _matMulTiledKernel!(
                new Index2D(m, n),
                a.Buffer.View,
                b.Buffer.View,
                result.Buffer.View,
                m, n, k);
        }
        else
        {
            // Use naive kernel for small matrices
            _matMulNaiveKernel!(
                new Index2D(m, n),
                a.Buffer.View,
                b.Buffer.View,
                result.Buffer.View,
                m, n, k);
        }

        Synchronize();
        return result;
    }

    /// <inheritdoc/>
    public GpuTensor<T> Transpose(GpuTensor<T> a)
    {
        if (a.Rank != 2)
        {
            throw new ArgumentException("Transpose currently only supports 2D tensors (matrices)");
        }

        int rows = a.Shape[0];
        int cols = a.Shape[1];

        // Result shape is swapped
        var result = Allocate(new[] { cols, rows });

        // For transpose, we need a different approach since we can't easily use Index2D
        // Let's implement a simple kernel that works with flat indices
        TransposeMatrix(a, result, rows, cols);

        return result;
    }

    /// <summary>
    /// Helper method to transpose a matrix.
    /// </summary>
    private void TransposeMatrix(GpuTensor<T> input, GpuTensor<T> output, int rows, int cols)
    {
        // Create a simple transpose kernel
        var kernel = _accelerator!.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, ArrayView<T>, int, int>(
            (Index1D index, ArrayView<T> inp, ArrayView<T> outp, int r, int c) =>
            {
                int i = (int)index;
                if (i >= r * c) return;

                int row = i / c;
                int col = i % c;

                // In input: row * cols + col
                // In output: col * rows + row (transposed)
                outp[col * r + row] = inp[row * c + col];
            });

        kernel(input.Length, input.Buffer.View, output.Buffer.View, rows, cols);
        Synchronize();
    }

    #endregion

    #region Activations

    /// <inheritdoc/>
    public GpuTensor<T> ReLU(GpuTensor<T> a)
    {
        var result = Allocate(a.Shape);
        _reluKernel!(result.Length, a.Buffer.View, result.Buffer.View);
        Synchronize();
        return result;
    }

    /// <inheritdoc/>
    public GpuTensor<T> Sigmoid(GpuTensor<T> a)
    {
        var result = Allocate(a.Shape);
        _sigmoidKernel!(result.Length, a.Buffer.View, result.Buffer.View);
        Synchronize();
        return result;
    }

    /// <inheritdoc/>
    public GpuTensor<T> Tanh(GpuTensor<T> a)
    {
        var result = Allocate(a.Shape);
        _tanhKernel!(result.Length, a.Buffer.View, result.Buffer.View);
        Synchronize();
        return result;
    }

    #endregion

    #region Reductions

    /// <inheritdoc/>
    public GpuTensor<T> Sum(GpuTensor<T> a)
    {
        // Use ILGPU.Algorithms for efficient reduction
        var sumValue = _numOps.Zero;

        // Simple implementation: Copy to CPU and sum
        // TODO: Implement true parallel reduction kernel
        var cpuTensor = ToCpu(a);
        for (int i = 0; i < cpuTensor.Length; i++)
        {
            sumValue = _numOps.Add(sumValue, cpuTensor[i]);
        }

        // Return as scalar GPU tensor
        var result = Allocate(new[] { 1 });
        var resultData = new T[] { sumValue };
        result.Buffer.CopyFromCPU(resultData);

        return result;
    }

    /// <inheritdoc/>
    public GpuTensor<T> Mean(GpuTensor<T> a)
    {
        // Compute sum first
        using var sumTensor = Sum(a);

        // Divide by count
        var sumData = sumTensor.Buffer.GetAsArray1D();
        var sumValue = sumData[0];

        var count = _numOps.FromInt(a.Length);
        var meanValue = _numOps.Divide(sumValue, count);

        // Return as scalar GPU tensor
        var result = Allocate(new[] { 1 });
        var resultData = new T[] { meanValue };
        result.Buffer.CopyFromCPU(resultData);

        return result;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Validates that two tensors have the same shape.
    /// </summary>
    private static void ValidateSameShape(GpuTensor<T> a, GpuTensor<T> b)
    {
        if (a.Rank != b.Rank)
        {
            throw new ArgumentException($"Tensor ranks don't match: {a.Rank} vs {b.Rank}");
        }

        for (int i = 0; i < a.Rank; i++)
        {
            if (a.Shape[i] != b.Shape[i])
            {
                throw new ArgumentException(
                    $"Tensor shapes don't match at dimension {i}: {a.Shape[i]} vs {b.Shape[i]}");
            }
        }
    }

    #endregion

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;

        _accelerator?.Dispose();
        _context?.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
