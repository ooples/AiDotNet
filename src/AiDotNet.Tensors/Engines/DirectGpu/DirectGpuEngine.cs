using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
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
/// CPU - always available
/// </code>
/// </remarks>
public sealed class DirectGpuEngine : IDisposable
{
    private readonly IDirectGpuBackend? _backend;
    private readonly KernelFusionManager _fusionManager;
    private readonly bool _isAvailable;
    private bool _disposed;
    private const string BackendOrderEnvVar = "AIDOTNET_DIRECTGPU_BACKENDS";
    private static readonly string[] DefaultBackendOrder = new[] { "cuda", "opencl", "hip" };

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
    /// Gets the underlying backend for direct GPU operations.
    /// Use this for GPU-resident operations to avoid CPU-GPU transfer overhead.
    /// </summary>
    public IDirectGpuBackend? Backend => _backend;

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
    /// Gets global memory in bytes.
    /// </summary>
    public long GlobalMemoryBytes => _backend?.GlobalMemoryBytes ?? 0;

    /// <summary>
    /// Gets local (shared) memory per workgroup in bytes.
    /// </summary>
    public long LocalMemoryBytes => _backend?.LocalMemoryBytes ?? 0;

    /// <summary>
    /// Initializes the DirectGpuEngine, automatically selecting the best available backend.
    /// </summary>
    /// <remarks>
    /// Backend selection order:
    /// 1. CUDA (NVIDIA GPUs with NVRTC available)
    /// 2. OpenCL (works on AMD, Intel, and NVIDIA)
    /// 3. HIP (AMD GPUs with MFMA support - MI100/200/300, RDNA3)
    /// </remarks>
    public DirectGpuEngine()
    {
        // Initialize fusion manager
        _fusionManager = new KernelFusionManager();

        Console.WriteLine("[DirectGpuEngine] Initializing GPU backends...");

        var backendOrder = GetBackendOrderFromEnv();
        if (backendOrder.Count == 0)
        {
            Console.WriteLine($"[DirectGpuEngine] Direct GPU backends disabled via {BackendOrderEnvVar}.");
            _isAvailable = false;
            return;
        }

        Console.WriteLine($"[DirectGpuEngine] Backend order: {string.Join(", ", backendOrder)}");
        foreach (var backendName in backendOrder)
        {
            var backend = TryCreateBackend(backendName);
            if (backend != null)
            {
                _backend = backend;
                _isAvailable = true;
                return;
            }
        }

        Console.WriteLine("[DirectGpuEngine] No GPU backends available. Falling back to CPU.");
        _isAvailable = false;
    }

    private static IReadOnlyList<string> GetBackendOrderFromEnv()
    {
        string? env = Environment.GetEnvironmentVariable(BackendOrderEnvVar);
        if (string.IsNullOrWhiteSpace(env))
            return DefaultBackendOrder;

        var tokens = env.Split(new[] { ',', ';', ' ', '|' }, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length == 0)
            return DefaultBackendOrder;

        foreach (var token in tokens)
        {
            var normalized = token.Trim().ToLowerInvariant();
            if (normalized is "none" or "disable" or "disabled")
                return Array.Empty<string>();
        }

        var result = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var token in tokens)
        {
            var normalized = token.Trim().ToLowerInvariant();
            switch (normalized)
            {
                case "auto":
                case "default":
                    continue;
                case "cuda":
                case "nvidia":
                case "nv":
                    AddBackend(result, seen, "cuda");
                    break;
                case "opencl":
                case "ocl":
                    AddBackend(result, seen, "opencl");
                    break;
                case "hip":
                case "rocm":
                    AddBackend(result, seen, "hip");
                    break;
                case "":
                    break;
                default:
                    Console.WriteLine($"[DirectGpuEngine] Unknown backend token '{token}' in {BackendOrderEnvVar}. Expected: cuda, opencl, hip, auto, none.");
                    break;
            }
        }

        return result.Count > 0 ? result : DefaultBackendOrder;
    }

    private static void AddBackend(List<string> backends, HashSet<string> seen, string name)
    {
        if (seen.Add(name))
            backends.Add(name);
    }

    private static IDirectGpuBackend? TryCreateBackend(string backendName)
    {
        return backendName.ToLowerInvariant() switch
        {
            "cuda" => TryCreateCudaBackend(),
            "opencl" => TryCreateOpenClBackend(),
            "hip" => TryCreateHipBackend(),
            _ => null
        };
    }

    private static IDirectGpuBackend? TryCreateCudaBackend()
    {
        try
        {
            Console.WriteLine("[DirectGpuEngine] Checking CUDA availability...");
            if (CudaBackend.IsCudaAvailable)
            {
                Console.WriteLine("[DirectGpuEngine] Creating CUDA backend...");
                var cudaBackend = new CudaBackend();
                Console.WriteLine($"[DirectGpuEngine] CUDA backend created, IsAvailable = {cudaBackend.IsAvailable}");

                if (cudaBackend.IsAvailable)
                {
                    Console.WriteLine($"[DirectGpuEngine] SUCCESS: Using CUDA backend on {cudaBackend.DeviceName}");
                    System.Diagnostics.Debug.WriteLine($"DirectGpuEngine: Using CUDA backend on {cudaBackend.DeviceName}");
                    return cudaBackend;
                }
                Console.WriteLine("[DirectGpuEngine] CUDA backend created but not available, disposing...");
                cudaBackend.Dispose();
            }
            else
            {
                Console.WriteLine("[DirectGpuEngine] CUDA is not available on this system");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[DirectGpuEngine] CUDA backend initialization failed: {ex.GetType().Name}: {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"[DirectGpuEngine] Inner exception: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}");
            }
            System.Diagnostics.Debug.WriteLine($"CUDA backend initialization failed: {ex.Message}");
        }

        return null;
    }

    private static IDirectGpuBackend? TryCreateOpenClBackend()
    {
        try
        {
            Console.WriteLine("[DirectGpuEngine] Checking OpenCL availability...");
            Console.WriteLine($"[DirectGpuEngine] OpenClBackend.IsOpenClAvailable = {OpenClBackend.IsOpenClAvailable}");

            Console.WriteLine("[DirectGpuEngine] Creating OpenCL backend...");
            var openClBackend = new OpenClBackend();
            Console.WriteLine($"[DirectGpuEngine] OpenCL backend created, IsAvailable = {openClBackend.IsAvailable}");

            if (openClBackend.IsAvailable)
            {
                Console.WriteLine($"[DirectGpuEngine] SUCCESS: Using OpenCL backend on {openClBackend.DeviceName}");
                System.Diagnostics.Debug.WriteLine($"DirectGpuEngine: Using OpenCL backend on {openClBackend.DeviceName}");
                return openClBackend;
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

        return null;
    }

    private static IDirectGpuBackend? TryCreateHipBackend()
    {
        try
        {
            Console.WriteLine("[DirectGpuEngine] Checking HIP availability...");
            if (HipBackend.IsHipAvailable)
            {
                Console.WriteLine("[DirectGpuEngine] HIP is available, creating HIP backend...");
                var hipBackend = new HipBackend();
                if (hipBackend.IsAvailable)
                {
                    Console.WriteLine($"[DirectGpuEngine] SUCCESS: Using HIP backend with {hipBackend.Architecture} architecture");
                    System.Diagnostics.Debug.WriteLine($"DirectGpuEngine: Using HIP backend with {hipBackend.Architecture} architecture");
                    return hipBackend;
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

        return null;
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
                    // No activation - GEMM + bias (bias added on CPU after GPU GEMM)
                    using (var bufferC = _backend.MatMul(bufferInput, bufferWeights, batchSize, outputFeatures, inputFeatures))
                    {
                        float[] tempResult = _backend.DownloadBuffer(bufferC);
                        // Add bias (GPU bias broadcast kernel would be more efficient)
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

        if (fusionResult.Pattern != null && fusionResult.RemainingOperations.Count == 0)
        {
            if (TryExecuteFusedGemmBiasActivation(
                fusionResult.Pattern.Operations,
                input,
                additionalBuffers,
                dimensions,
                out var fusedResult))
            {
                return fusedResult;
            }
        }

        System.Diagnostics.Debug.WriteLine(
            $"Fusion available but falling back to sequential execution: {fusionResult.FusedKernelName} " +
            $"(fused {fusionResult.FusedOperationCount} ops, {fusionResult.RemainingOperations.Count} remaining)");

        return ExecuteOperationsSequentially(operations, input, additionalBuffers, dimensions);
    }

    private bool TryExecuteFusedGemmBiasActivation<T>(
        IReadOnlyList<FusableOperation> operations,
        T[] input,
        IReadOnlyDictionary<string, T[]>? additionalBuffers,
        IReadOnlyDictionary<string, int>? dimensions,
        out T[]? result)
    {
        result = null;
        if (operations == null || operations.Count < 2 || operations.Count > 3)
            return false;
        if (operations[0].OperationType != GpuOperationType.Gemm ||
            operations[1].OperationType != GpuOperationType.BiasAdd)
            return false;
        if (!string.IsNullOrEmpty(operations[0].Metadata) ||
            !string.IsNullOrEmpty(operations[1].Metadata))
            return false;

        ActivationType activation = ActivationType.None;
        if (operations.Count == 3)
        {
            if (!string.IsNullOrEmpty(operations[2].Metadata))
                return false;
            activation = operations[2].OperationType switch
            {
                GpuOperationType.ReLU => ActivationType.ReLU,
                GpuOperationType.GELU => ActivationType.GELU,
                GpuOperationType.Sigmoid => ActivationType.Sigmoid,
                GpuOperationType.Tanh => ActivationType.Tanh,
                _ => ActivationType.None
            };

            if (activation == ActivationType.None)
                return false;
        }

        if (!TryGetDenseInputs(
            input,
            additionalBuffers,
            dimensions,
            out var weights,
            out var bias,
            out var batchSize,
            out var inputFeatures,
            out var outputFeatures))
        {
            return false;
        }

        result = DenseForwardFused(
            input,
            weights,
            bias,
            batchSize,
            inputFeatures,
            outputFeatures,
            activation);
        return result != null;
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
        if (_backend == null)
            return null;

        float[] current = ToFloatArray(input);

        foreach (var op in operations)
        {
            if (!string.IsNullOrEmpty(op.Metadata))
            {
                System.Diagnostics.Debug.WriteLine($"Unsupported operation metadata for sequential execution: {op}");
                return null;
            }

            switch (op.OperationType)
            {
                case GpuOperationType.Gemm:
                    if (!TryGetBuffer(additionalBuffers, out var weights, "weights", "weight", "w", "matrixB"))
                    {
                        System.Diagnostics.Debug.WriteLine("Missing weights for GEMM execution.");
                        return null;
                    }
                    if (!TryGetGemmDimensions(dimensions, out var m, out var k, out var n))
                    {
                        System.Diagnostics.Debug.WriteLine("Missing GEMM dimensions for sequential execution.");
                        return null;
                    }

                    float[] weightsFloat = ToFloatArray(weights);
                    using (var bufferA = _backend.AllocateBuffer(current))
                    using (var bufferB = _backend.AllocateBuffer(weightsFloat))
                    using (var bufferC = _backend.MatMul(bufferA, bufferB, m, n, k))
                    {
                        current = _backend.DownloadBuffer(bufferC);
                    }
                    break;

                case GpuOperationType.BiasAdd:
                    if (!TryGetBuffer(additionalBuffers, out var bias, "bias", "b"))
                    {
                        System.Diagnostics.Debug.WriteLine("Missing bias for BiasAdd execution.");
                        return null;
                    }

                    float[] biasFloat = ToFloatArray(bias);
                    if (!TryGetOutputFeatureCount(dimensions, biasFloat, current.Length, out var featureCount))
                    {
                        System.Diagnostics.Debug.WriteLine("Missing output feature count for BiasAdd execution.");
                        return null;
                    }

                    for (int i = 0; i < current.Length; i++)
                    {
                        current[i] += biasFloat[i % featureCount];
                    }
                    break;

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

    private static bool TryGetDenseInputs<T>(
        T[] input,
        IReadOnlyDictionary<string, T[]>? additionalBuffers,
        IReadOnlyDictionary<string, int>? dimensions,
        out T[] weights,
        out T[] bias,
        out int batchSize,
        out int inputFeatures,
        out int outputFeatures)
    {
        weights = Array.Empty<T>();
        bias = Array.Empty<T>();
        batchSize = 0;
        inputFeatures = 0;
        outputFeatures = 0;

        if (!TryGetBuffer(additionalBuffers, out weights, "weights", "weight", "w", "matrixB"))
            return false;
        if (!TryGetBuffer(additionalBuffers, out bias, "bias", "b"))
            return false;

        if (TryGetDimension(dimensions, out batchSize, "batchSize", "batch", "m") &&
            TryGetDimension(dimensions, out inputFeatures, "inputFeatures", "input", "k") &&
            TryGetDimension(dimensions, out outputFeatures, "outputFeatures", "output", "n"))
        {
            return ValidateDenseDimensions(input.Length, weights.Length, bias.Length, batchSize, inputFeatures, outputFeatures);
        }

        if (bias.Length == 0)
            return false;

        outputFeatures = bias.Length;
        if (weights.Length % outputFeatures != 0)
            return false;

        inputFeatures = weights.Length / outputFeatures;
        if (inputFeatures <= 0 || input.Length % inputFeatures != 0)
            return false;

        batchSize = input.Length / inputFeatures;
        return ValidateDenseDimensions(input.Length, weights.Length, bias.Length, batchSize, inputFeatures, outputFeatures);
    }

    private static bool ValidateDenseDimensions(int inputLength, int weightsLength, int biasLength, int batchSize, int inputFeatures, int outputFeatures)
    {
        if (batchSize <= 0 || inputFeatures <= 0 || outputFeatures <= 0)
            return false;
        if (inputLength != batchSize * inputFeatures)
            return false;
        if (weightsLength != inputFeatures * outputFeatures)
            return false;
        return biasLength == outputFeatures;
    }

    private static bool TryGetGemmDimensions(
        IReadOnlyDictionary<string, int>? dimensions,
        out int m,
        out int k,
        out int n)
    {
        m = 0;
        k = 0;
        n = 0;

        return TryGetDimension(dimensions, out m, "m", "batchSize", "batch") &&
               TryGetDimension(dimensions, out k, "k", "inputFeatures", "input") &&
               TryGetDimension(dimensions, out n, "n", "outputFeatures", "output");
    }

    private static bool TryGetOutputFeatureCount(
        IReadOnlyDictionary<string, int>? dimensions,
        float[] bias,
        int currentLength,
        out int outputFeatures)
    {
        outputFeatures = 0;
        if (TryGetDimension(dimensions, out outputFeatures, "outputFeatures", "output", "n", "features", "cols", "columns"))
        {
            return outputFeatures > 0;
        }

        if (bias.Length > 0 && currentLength % bias.Length == 0)
        {
            outputFeatures = bias.Length;
            return true;
        }

        return false;
    }

    private static bool TryGetDimension(
        IReadOnlyDictionary<string, int>? dimensions,
        out int value,
        params string[] keys)
    {
        value = 0;
        if (dimensions == null || keys == null || keys.Length == 0)
            return false;

        foreach (var key in keys)
        {
            if (dimensions.TryGetValue(key, out value))
                return true;
        }

        foreach (var pair in dimensions)
        {
            foreach (var key in keys)
            {
                if (string.Equals(pair.Key, key, StringComparison.OrdinalIgnoreCase))
                {
                    value = pair.Value;
                    return true;
                }
            }
        }

        return false;
    }

    private static bool TryGetBuffer<T>(
        IReadOnlyDictionary<string, T[]>? buffers,
        out T[] buffer,
        params string[] keys)
    {
        buffer = Array.Empty<T>();
        if (buffers == null || keys == null || keys.Length == 0)
            return false;

        foreach (var key in keys)
        {
            if (buffers.TryGetValue(key, out var candidate) && candidate != null)
            {
                buffer = candidate;
                return true;
            }
        }

        foreach (var pair in buffers)
        {
            foreach (var key in keys)
            {
                if (string.Equals(pair.Key, key, StringComparison.OrdinalIgnoreCase) && pair.Value != null)
                {
                    buffer = pair.Value;
                    return true;
                }
            }
        }

        return false;
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
