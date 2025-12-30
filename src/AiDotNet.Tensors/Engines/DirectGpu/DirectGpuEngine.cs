using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Operators;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// High-performance GPU engine that talks directly to hardware via P/Invoke.
/// Supports any generic type T by converting to float at the GPU boundary.
/// </summary>
/// <remarks>
/// <para><b>Design Philosophy:</b></para>
/// <para>
/// This engine provides the fastest GPU path by:
/// 1. Using float32-only kernels (optimal GPU performance)
/// 2. Converting generic types at the boundary (preserves clean API)
/// 3. Implementing optimizations CLBlast misses (tensor cores, fusion, double-buffering)
/// </para>
/// <para><b>Fallback Tiers:</b></para>
/// <code>
/// DirectGpuEngine (this) - custom optimized kernels
///     ↓ fallback
/// CLBlast - tuned but missing optimizations
///     ↓ fallback
/// ILGPU - general purpose
///     ↓ fallback
/// CPU - always available
/// </code>
/// </remarks>
public sealed class DirectGpuEngine : IDisposable
{
    private readonly IDirectGpuBackend? _backend;
    private readonly KernelFusionManager _fusionManager;
    private readonly bool _isAvailable;
    private bool _disposed;

    /// <summary>
    /// Gets whether the direct GPU engine is available.
    /// </summary>
    public bool IsAvailable => _isAvailable && _backend != null;

    /// <summary>
    /// Gets the kernel fusion manager for tracking and fusing operation sequences.
    /// </summary>
    public KernelFusionManager FusionManager => _fusionManager;

    /// <summary>
    /// Gets the backend name (OpenCL, CUDA, etc.).
    /// </summary>
    public string BackendName => _backend?.BackendName ?? "None";

    /// <summary>
    /// Gets the GPU device name.
    /// </summary>
    public string DeviceName => _backend?.DeviceName ?? "None";

    /// <summary>
    /// Gets the GPU vendor.
    /// </summary>
    public string DeviceVendor => _backend?.DeviceVendor ?? "None";

    /// <summary>
    /// Gets the number of compute units.
    /// </summary>
    public int ComputeUnits => _backend?.ComputeUnits ?? 0;

    /// <summary>
    /// Gets global memory in GB.
    /// </summary>
    public double GlobalMemoryGB => (_backend?.GlobalMemoryBytes ?? 0) / (1024.0 * 1024 * 1024);

    /// <summary>
    /// Initializes the DirectGpuEngine, automatically selecting the best available backend.
    /// </summary>
    /// <remarks>
    /// Backend selection order:
    /// 1. HIP (AMD GPUs with MFMA support - MI100/200/300, RDNA3)
    /// 2. OpenCL (works on AMD, Intel, and NVIDIA)
    /// 3. Future: CUDA (NVIDIA-specific optimizations)
    /// </remarks>
    public DirectGpuEngine()
    {
        // Initialize fusion manager
        _fusionManager = new KernelFusionManager();

        Console.WriteLine("[DirectGpuEngine] Initializing GPU backends...");

        // Try backends in order of preference for maximum performance
        // NOTE: OpenCL is preferred for now as it has optimized GEMM kernels
        // HIP support is experimental and may have stability issues on some AMD GPUs

        // 1. Try OpenCL first (works on AMD, Intel, and NVIDIA)
        // Our optimized kernels with double buffering, KREG, and vectorized loads are in OpenCL
        try
        {
            Console.WriteLine("[DirectGpuEngine] Checking OpenCL availability...");
            Console.WriteLine($"[DirectGpuEngine] OpenClBackend.IsOpenClAvailable = {OpenClBackend.IsOpenClAvailable}");

            Console.WriteLine("[DirectGpuEngine] Creating OpenCL backend...");
            var openClBackend = new OpenClBackend();
            Console.WriteLine($"[DirectGpuEngine] OpenCL backend created, IsAvailable = {openClBackend.IsAvailable}");

            if (openClBackend.IsAvailable)
            {
                _backend = openClBackend;
                _isAvailable = true;
                Console.WriteLine($"[DirectGpuEngine] SUCCESS: Using OpenCL backend on {openClBackend.DeviceName}");
                System.Diagnostics.Debug.WriteLine($"DirectGpuEngine: Using OpenCL backend on {openClBackend.DeviceName}");
                return;
            }
            Console.WriteLine("[DirectGpuEngine] OpenCL backend created but not available, disposing...");
            openClBackend.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[DirectGpuEngine] OpenCL backend initialization failed: {ex.GetType().Name}: {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"[DirectGpuEngine] Inner exception: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}");
            }
            System.Diagnostics.Debug.WriteLine($"OpenCL backend initialization failed: {ex.Message}");
        }

        // 2. Try HIP backend for AMD GPUs (experimental)
        // HIP provides MFMA support on MI100/200/300 GPUs
        try
        {
            Console.WriteLine("[DirectGpuEngine] Checking HIP availability...");
            if (HipBackend.IsHipAvailable)
            {
                Console.WriteLine("[DirectGpuEngine] HIP is available, creating HIP backend...");
                var hipBackend = new HipBackend();
                if (hipBackend.IsAvailable)
                {
                    _backend = hipBackend;
                    _isAvailable = true;
                    Console.WriteLine($"[DirectGpuEngine] SUCCESS: Using HIP backend with {hipBackend.Architecture} architecture");
                    System.Diagnostics.Debug.WriteLine($"DirectGpuEngine: Using HIP backend with {hipBackend.Architecture} architecture");
                    return;
                }
                Console.WriteLine("[DirectGpuEngine] HIP backend created but not available, disposing...");
                hipBackend.Dispose();
            }
            else
            {
                Console.WriteLine("[DirectGpuEngine] HIP is not available on this system");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[DirectGpuEngine] HIP backend initialization failed: {ex.GetType().Name}: {ex.Message}");
            System.Diagnostics.Debug.WriteLine($"HIP backend initialization failed: {ex.Message}");
        }

        // Future: Try CUDA backend
        // try
        // {
        //     var cudaBackend = new CudaBackend();
        //     if (cudaBackend.IsAvailable)
        //     {
        //         _backend = cudaBackend;
        //         _isAvailable = true;
        //         return;
        //     }
        //     cudaBackend.Dispose();
        // }
        // catch { }

        Console.WriteLine("[DirectGpuEngine] No GPU backends available. Falling back to CPU.");
        _isAvailable = false;
    }

    #region Type Conversion (Generic T → float → T)

    /// <summary>
    /// Converts a generic array to float array for GPU processing.
    /// Uses vectorized span-based conversion via IVectorizedOperations.ToFloatSpan.
    /// </summary>
    public static float[] ToFloatArray<T>(T[] data)
    {
        if (data is float[] floatData)
            return floatData;

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new float[data.Length];
        numOps.ToFloatSpan(new ReadOnlySpan<T>(data), new Span<float>(result));
        return result;
    }

    /// <summary>
    /// Converts a float array back to the generic type.
    /// Uses vectorized span-based conversion via IVectorizedOperations.FromFloatSpan.
    /// </summary>
    public static T[] FromFloatArray<T>(float[] data)
    {
        if (typeof(T) == typeof(float))
            return (T[])(object)data;

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[data.Length];
        numOps.FromFloatSpan(new ReadOnlySpan<float>(data), new Span<T>(result));
        return result;
    }

    #endregion

    #region Matrix Operations

    /// <summary>
    /// Matrix multiplication: C = A * B
    /// </summary>
    /// <typeparam name="T">Element type (will be converted to float for GPU).</typeparam>
    /// <param name="A">Matrix A data (M x K, row-major).</param>
    /// <param name="B">Matrix B data (K x N, row-major).</param>
    /// <param name="M">Rows of A.</param>
    /// <param name="K">Columns of A / Rows of B.</param>
    /// <param name="N">Columns of B.</param>
    /// <returns>Result matrix C (M x N), or null if GPU unavailable.</returns>
    public T[]? MatMul<T>(T[] A, T[] B, int M, int K, int N)
    {
        if (!IsAvailable || _backend == null)
            return null;

        // Convert to float
        float[] aFloat = ToFloatArray(A);
        float[] bFloat = ToFloatArray(B);

        // Allocate GPU buffers
        using var bufferA = _backend.AllocateBuffer(aFloat);
        using var bufferB = _backend.AllocateBuffer(bFloat);

        // Execute GEMM
        using var bufferC = _backend.MatMul(bufferA, bufferB, M, N, K);

        // Download result
        float[] resultFloat = _backend.DownloadBuffer(bufferC);

        // Convert back to T
        return FromFloatArray<T>(resultFloat);
    }

    /// <summary>
    /// Matrix multiplication with pre-allocated GPU weight buffer (for cached weights).
    /// </summary>
    public T[]? MatMulWithCachedWeights<T>(T[] input, IGpuBuffer cachedWeights, int M, int K, int N)
    {
        if (!IsAvailable || _backend == null)
            return null;

        float[] inputFloat = ToFloatArray(input);
        using var bufferInput = _backend.AllocateBuffer(inputFloat);
        using var bufferC = _backend.MatMul(bufferInput, cachedWeights, M, N, K);

        float[] resultFloat = _backend.DownloadBuffer(bufferC);
        return FromFloatArray<T>(resultFloat);
    }

    /// <summary>
    /// Fused Dense layer forward: output = Activation(input * weights + bias)
    /// Uses the kernel fusion manager to select the optimal fused kernel.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="input">Input data (batchSize x inputFeatures).</param>
    /// <param name="weights">Weight matrix (inputFeatures x outputFeatures).</param>
    /// <param name="bias">Bias vector (outputFeatures).</param>
    /// <param name="batchSize">Batch size.</param>
    /// <param name="inputFeatures">Input feature count.</param>
    /// <param name="outputFeatures">Output feature count.</param>
    /// <param name="activation">Activation function type for fused kernel selection.</param>
    /// <returns>Output data (batchSize x outputFeatures), or null if GPU unavailable.</returns>
    public T[]? DenseForwardFused<T>(
        T[] input, T[] weights, T[] bias,
        int batchSize, int inputFeatures, int outputFeatures,
        ActivationType activation = ActivationType.None)
    {
        if (!IsAvailable || _backend == null)
            return null;

        float[] inputFloat = ToFloatArray(input);
        float[] weightsFloat = ToFloatArray(weights);
        float[] biasFloat = ToFloatArray(bias);

        using var bufferInput = _backend.AllocateBuffer(inputFloat);
        using var bufferWeights = _backend.AllocateBuffer(weightsFloat);
        using var bufferBias = _backend.AllocateBuffer(biasFloat);

        // Use fusion manager to find the optimal kernel
        var kernelName = _fusionManager.GetGemmBiasActivationKernel(activation);

        if (kernelName != null)
        {
            IGpuBuffer resultBuffer;

            // Execute the fused kernel
            switch (activation)
            {
                case ActivationType.ReLU:
                    resultBuffer = _backend.GemmBiasRelu(bufferInput, bufferWeights, bufferBias, batchSize, outputFeatures, inputFeatures);
                    break;
                case ActivationType.GELU:
                    resultBuffer = _backend.GemmBiasGelu(bufferInput, bufferWeights, bufferBias, batchSize, outputFeatures, inputFeatures);
                    break;
                case ActivationType.Sigmoid:
                    resultBuffer = _backend.GemmBiasSigmoid(bufferInput, bufferWeights, bufferBias, batchSize, outputFeatures, inputFeatures);
                    break;
                case ActivationType.Tanh:
                    resultBuffer = _backend.GemmBiasTanh(bufferInput, bufferWeights, bufferBias, batchSize, outputFeatures, inputFeatures);
                    break;
                case ActivationType.None:
                default:
                    // No activation - just GEMM + bias (still fused)
                    using (var bufferC = _backend.MatMul(bufferInput, bufferWeights, batchSize, outputFeatures, inputFeatures))
                    {
                        // TODO: Add bias (for now, just return GEMM result)
                        float[] tempResult = _backend.DownloadBuffer(bufferC);
                        // Add bias on CPU for now
                        for (int b = 0; b < batchSize; b++)
                        {
                            for (int o = 0; o < outputFeatures; o++)
                            {
                                tempResult[b * outputFeatures + o] += biasFloat[o];
                            }
                        }
                        return FromFloatArray<T>(tempResult);
                    }
            }

            using (resultBuffer)
            {
                float[] resultFloat = _backend.DownloadBuffer(resultBuffer);
                return FromFloatArray<T>(resultFloat);
            }
        }

        // Fallback: no fused kernel available, execute separately
        var unfusedResult = ExecuteUnfusedDenseForwardInternal(bufferInput, bufferWeights, biasFloat, batchSize, inputFeatures, outputFeatures, activation);
        return unfusedResult != null ? FromFloatArray<T>(unfusedResult) : null;
    }

    /// <summary>
    /// Executes a Dense forward pass without kernel fusion (fallback path).
    /// </summary>
    private float[]? ExecuteUnfusedDenseForwardInternal(
        IGpuBuffer bufferInput, IGpuBuffer bufferWeights,
        float[] biasFloat, int batchSize, int inputFeatures, int outputFeatures,
        ActivationType activation)
    {
        if (_backend == null)
            return null;

        // Step 1: GEMM
        using var bufferC = _backend.MatMul(bufferInput, bufferWeights, batchSize, outputFeatures, inputFeatures);

        // Step 2: Download and add bias (CPU for now)
        float[] result = _backend.DownloadBuffer(bufferC);
        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < outputFeatures; o++)
            {
                result[b * outputFeatures + o] += biasFloat[o];
            }
        }

        // Step 3: Apply activation (if any)
        if (activation != ActivationType.None)
        {
            using var bufferResult = _backend.AllocateBuffer(result);
            using var bufferOutput = _backend.AllocateBuffer(result.Length);

            switch (activation)
            {
                case ActivationType.ReLU:
                    _backend.Relu(bufferResult, bufferOutput, result.Length);
                    break;
                case ActivationType.GELU:
                    _backend.Gelu(bufferResult, bufferOutput, result.Length);
                    break;
                case ActivationType.Sigmoid:
                    _backend.Sigmoid(bufferResult, bufferOutput, result.Length);
                    break;
                case ActivationType.Tanh:
                    _backend.Tanh(bufferResult, bufferOutput, result.Length);
                    break;
            }

            result = _backend.DownloadBuffer(bufferOutput);
        }

        return result;
    }

    /// <summary>
    /// Executes a sequence of operations, automatically fusing where possible.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="operations">The operations to execute.</param>
    /// <param name="input">Initial input data.</param>
    /// <param name="additionalBuffers">Additional buffers needed for the operations (e.g., weights, bias).</param>
    /// <param name="dimensions">Dimension parameters (M, K, N, etc.).</param>
    /// <returns>The result, or null if GPU unavailable.</returns>
    public T[]? ExecuteWithFusion<T>(
        IReadOnlyList<FusableOperation> operations,
        T[] input,
        IReadOnlyDictionary<string, T[]>? additionalBuffers = null,
        IReadOnlyDictionary<string, int>? dimensions = null)
    {
        if (!IsAvailable || _backend == null || operations == null || operations.Count == 0)
            return null;

        // Try to fuse the operations
        var fusionResult = _fusionManager.TryFuseOperations(operations);

        if (!fusionResult.IsFused)
        {
            // No fusion available - execute individually
            return ExecuteOperationsSequentially(operations, input, additionalBuffers, dimensions);
        }

        // For now, return null for complex fusion sequences that aren't directly supported
        // The caller should use DenseForwardFused for GEMM+Bias+Activation patterns
        System.Diagnostics.Debug.WriteLine(
            $"Fusion available: {fusionResult.FusedKernelName} " +
            $"(fused {fusionResult.FusedOperationCount} ops, {fusionResult.RemainingOperations.Count} remaining)");

        // TODO: Implement generic fused execution based on kernel name
        return null;
    }

    /// <summary>
    /// Executes operations sequentially without fusion (fallback).
    /// </summary>
    private T[]? ExecuteOperationsSequentially<T>(
        IReadOnlyList<FusableOperation> operations,
        T[] input,
        IReadOnlyDictionary<string, T[]>? additionalBuffers,
        IReadOnlyDictionary<string, int>? dimensions)
    {
        // Simplified sequential execution - can be extended
        float[] current = ToFloatArray(input);

        foreach (var op in operations)
        {
            switch (op.OperationType)
            {
                case GpuOperationType.ReLU:
                    using (var bufferA = _backend!.AllocateBuffer(current))
                    using (var bufferB = _backend.AllocateBuffer(current.Length))
                    {
                        _backend.Relu(bufferA, bufferB, current.Length);
                        current = _backend.DownloadBuffer(bufferB);
                    }
                    break;

                case GpuOperationType.Sigmoid:
                    using (var bufferA = _backend!.AllocateBuffer(current))
                    using (var bufferB = _backend.AllocateBuffer(current.Length))
                    {
                        _backend.Sigmoid(bufferA, bufferB, current.Length);
                        current = _backend.DownloadBuffer(bufferB);
                    }
                    break;

                case GpuOperationType.Tanh:
                    using (var bufferA = _backend!.AllocateBuffer(current))
                    using (var bufferB = _backend.AllocateBuffer(current.Length))
                    {
                        _backend.Tanh(bufferA, bufferB, current.Length);
                        current = _backend.DownloadBuffer(bufferB);
                    }
                    break;

                case GpuOperationType.GELU:
                    using (var bufferA = _backend!.AllocateBuffer(current))
                    using (var bufferB = _backend.AllocateBuffer(current.Length))
                    {
                        _backend.Gelu(bufferA, bufferB, current.Length);
                        current = _backend.DownloadBuffer(bufferB);
                    }
                    break;

                // Add more operation types as needed
                default:
                    System.Diagnostics.Debug.WriteLine($"Unsupported operation type for sequential execution: {op.OperationType}");
                    break;
            }
        }

        return FromFloatArray<T>(current);
    }

    #endregion

    #region Activation Operations

    /// <summary>
    /// ReLU activation: output = max(0, input)
    /// </summary>
    public T[]? Relu<T>(T[] input)
    {
        if (!IsAvailable || _backend == null)
            return null;

        float[] inputFloat = ToFloatArray(input);
        using var bufferA = _backend.AllocateBuffer(inputFloat);
        using var bufferB = _backend.AllocateBuffer(input.Length);

        _backend.Relu(bufferA, bufferB, input.Length);

        float[] resultFloat = _backend.DownloadBuffer(bufferB);
        return FromFloatArray<T>(resultFloat);
    }

    /// <summary>
    /// Sigmoid activation: output = 1 / (1 + exp(-input))
    /// </summary>
    public T[]? Sigmoid<T>(T[] input)
    {
        if (!IsAvailable || _backend == null)
            return null;

        float[] inputFloat = ToFloatArray(input);
        using var bufferA = _backend.AllocateBuffer(inputFloat);
        using var bufferB = _backend.AllocateBuffer(input.Length);

        _backend.Sigmoid(bufferA, bufferB, input.Length);

        float[] resultFloat = _backend.DownloadBuffer(bufferB);
        return FromFloatArray<T>(resultFloat);
    }

    /// <summary>
    /// Softmax activation along last dimension.
    /// </summary>
    public T[]? Softmax<T>(T[] input, int batchSize, int features)
    {
        if (!IsAvailable || _backend == null)
            return null;

        float[] inputFloat = ToFloatArray(input);
        using var bufferA = _backend.AllocateBuffer(inputFloat);
        using var bufferB = _backend.AllocateBuffer(input.Length);

        _backend.Softmax(bufferA, bufferB, batchSize, features);

        float[] resultFloat = _backend.DownloadBuffer(bufferB);
        return FromFloatArray<T>(resultFloat);
    }

    #endregion

    #region GPU Buffer Management (for cached weights)

    /// <summary>
    /// Allocates a persistent GPU buffer for cached weights.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="data">Data to upload.</param>
    /// <returns>GPU buffer handle, or null if unavailable.</returns>
    public IGpuBuffer? AllocatePersistentBuffer<T>(T[] data)
    {
        if (!IsAvailable || _backend == null)
            return null;

        float[] floatData = ToFloatArray(data);
        return _backend.AllocateBuffer(floatData);
    }

    #endregion

    #region Diagnostics

    /// <summary>
    /// Gets a diagnostic string with GPU information.
    /// </summary>
    public string GetDiagnostics()
    {
        if (!IsAvailable || _backend == null)
            return "DirectGpuEngine: Not available (no compatible GPU found)";

        return $"""
            DirectGpuEngine: Available
            Backend: {BackendName}
            Device: {DeviceName}
            Vendor: {DeviceVendor}
            Compute Units: {ComputeUnits}
            Global Memory: {GlobalMemoryGB:F2} GB
            Local Memory: {(_backend.LocalMemoryBytes / 1024.0):F0} KB
            Registered Fusion Patterns: {_fusionManager.RegisteredPatterns.Count}
            """;
    }

    /// <summary>
    /// Gets detailed fusion statistics.
    /// </summary>
    public string GetFusionStatistics()
    {
        return _fusionManager.GetStatistics();
    }

    #endregion

    public void Dispose()
    {
        if (_disposed) return;
        _backend?.Dispose();
        _disposed = true;
    }
}
