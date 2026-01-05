namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Configuration options for GPU execution behavior.
/// Can be configured via code or environment variables.
/// </summary>
public sealed class GpuExecutionOptions
{
    /// <summary>
    /// Environment variable prefix for configuration.
    /// </summary>
    private const string EnvPrefix = "AIDOTNET_GPU_";

    /// <summary>
    /// Gets or sets the minimum number of elements for GPU to be used.
    /// Operations with fewer elements will use CPU for lower overhead.
    /// Default: 4096. Environment variable: AIDOTNET_GPU_MIN_ELEMENTS
    /// </summary>
    public int MinGpuElements { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the maximum number of compute streams to use.
    /// More streams enable more parallelism but use more resources.
    /// Default: 3. Environment variable: AIDOTNET_GPU_STREAMS
    /// </summary>
    public int MaxComputeStreams { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to force GPU execution regardless of size.
    /// Useful for benchmarking. Default: false. Environment variable: AIDOTNET_GPU_FORCE_GPU
    /// </summary>
    public bool ForceGpu { get; set; }

    /// <summary>
    /// Gets or sets whether to force CPU execution regardless of size.
    /// Useful for debugging. Default: false. Environment variable: AIDOTNET_GPU_FORCE_CPU
    /// </summary>
    public bool ForceCpu { get; set; }

    /// <summary>
    /// Gets or sets whether to enable execution graph compilation.
    /// Graphs enable operation fusion and scheduling optimization.
    /// Default: true. Environment variable: AIDOTNET_GPU_ENABLE_GRAPH
    /// </summary>
    public bool EnableGraphCompilation { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable automatic kernel fusion.
    /// Fused kernels reduce launch overhead for common patterns (GEMM+Bias+ReLU).
    /// Default: true. Environment variable: AIDOTNET_GPU_ENABLE_FUSION
    /// </summary>
    public bool EnableAutoFusion { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum fraction of GPU memory to use (0.0 to 1.0).
    /// Exceeding this triggers memory pressure handling.
    /// Default: 0.8 (80%). Environment variable: AIDOTNET_GPU_MAX_MEMORY
    /// </summary>
    public double MaxMemoryUsage { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets whether to enable tensor prefetching.
    /// Prefetch uploads data before it's needed.
    /// Default: true. Environment variable: AIDOTNET_GPU_ENABLE_PREFETCH
    /// </summary>
    public bool EnablePrefetch { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable compute/transfer overlap.
    /// Uses separate streams for compute and data transfer.
    /// Default: true. Environment variable: AIDOTNET_GPU_ENABLE_OVERLAP
    /// </summary>
    public bool EnableComputeTransferOverlap { get; set; } = true;

    /// <summary>
    /// Gets or sets the execution mode.
    /// Default: Auto (selects best mode based on capabilities).
    /// Environment variable: AIDOTNET_GPU_EXECUTION_MODE
    /// </summary>
    public GpuExecutionMode ExecutionMode { get; set; } = GpuExecutionMode.Auto;

    /// <summary>
    /// Gets or sets whether to keep tensors GPU-resident between operations.
    /// Reduces upload/download overhead for chained operations.
    /// Default: true. Environment variable: AIDOTNET_GPU_RESIDENT
    /// </summary>
    public bool EnableGpuResidency { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of transfer streams to use.
    /// Separate streams for H2D and D2H transfers.
    /// Default: 2. Environment variable: AIDOTNET_GPU_TRANSFER_STREAMS
    /// </summary>
    public int TransferStreams { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to enable profiling/timing.
    /// Adds overhead but provides detailed performance data.
    /// Default: false. Environment variable: AIDOTNET_GPU_PROFILING
    /// </summary>
    public bool EnableProfiling { get; set; }

    /// <summary>
    /// Gets or sets the batch size for graph execution.
    /// Larger batches allow more optimization but use more memory.
    /// Default: 32. Environment variable: AIDOTNET_GPU_BATCH_SIZE
    /// </summary>
    public int GraphBatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets whether to cache compiled graphs.
    /// Avoids recompilation for repeated execution patterns.
    /// Default: true. Environment variable: AIDOTNET_GPU_CACHE_GRAPHS
    /// </summary>
    public bool CacheCompiledGraphs { get; set; } = true;

    /// <summary>
    /// Creates options with default values.
    /// </summary>
    public GpuExecutionOptions()
    {
    }

    /// <summary>
    /// Creates options initialized from environment variables.
    /// </summary>
    /// <returns>Options configured from environment variables.</returns>
    public static GpuExecutionOptions FromEnvironment()
    {
        var options = new GpuExecutionOptions();

        if (TryGetEnvInt("MIN_ELEMENTS", out int minElements))
        {
            options.MinGpuElements = minElements;
        }

        if (TryGetEnvInt("STREAMS", out int streams))
        {
            options.MaxComputeStreams = streams;
        }

        if (TryGetEnvBool("FORCE_GPU", out bool forceGpu))
        {
            options.ForceGpu = forceGpu;
        }

        if (TryGetEnvBool("FORCE_CPU", out bool forceCpu))
        {
            options.ForceCpu = forceCpu;
        }

        if (TryGetEnvBool("ENABLE_GRAPH", out bool enableGraph))
        {
            options.EnableGraphCompilation = enableGraph;
        }

        if (TryGetEnvBool("ENABLE_FUSION", out bool enableFusion))
        {
            options.EnableAutoFusion = enableFusion;
        }

        if (TryGetEnvDouble("MAX_MEMORY", out double maxMemory))
        {
            options.MaxMemoryUsage = Math.Max(0.0, Math.Min(1.0, maxMemory));
        }

        if (TryGetEnvBool("ENABLE_PREFETCH", out bool enablePrefetch))
        {
            options.EnablePrefetch = enablePrefetch;
        }

        if (TryGetEnvBool("ENABLE_OVERLAP", out bool enableOverlap))
        {
            options.EnableComputeTransferOverlap = enableOverlap;
        }

        if (TryGetEnvEnum<GpuExecutionMode>("EXECUTION_MODE", out var mode))
        {
            options.ExecutionMode = mode;
        }

        if (TryGetEnvBool("RESIDENT", out bool resident))
        {
            options.EnableGpuResidency = resident;
        }

        if (TryGetEnvInt("TRANSFER_STREAMS", out int transferStreams))
        {
            options.TransferStreams = transferStreams;
        }

        if (TryGetEnvBool("PROFILING", out bool profiling))
        {
            options.EnableProfiling = profiling;
        }

        if (TryGetEnvInt("BATCH_SIZE", out int batchSize))
        {
            options.GraphBatchSize = batchSize;
        }

        if (TryGetEnvBool("CACHE_GRAPHS", out bool cacheGraphs))
        {
            options.CacheCompiledGraphs = cacheGraphs;
        }

        return options;
    }

    /// <summary>
    /// Creates a copy of these options.
    /// </summary>
    /// <returns>A new options instance with the same values.</returns>
    public GpuExecutionOptions Clone()
    {
        return new GpuExecutionOptions
        {
            MinGpuElements = MinGpuElements,
            MaxComputeStreams = MaxComputeStreams,
            ForceGpu = ForceGpu,
            ForceCpu = ForceCpu,
            EnableGraphCompilation = EnableGraphCompilation,
            EnableAutoFusion = EnableAutoFusion,
            MaxMemoryUsage = MaxMemoryUsage,
            EnablePrefetch = EnablePrefetch,
            EnableComputeTransferOverlap = EnableComputeTransferOverlap,
            ExecutionMode = ExecutionMode,
            EnableGpuResidency = EnableGpuResidency,
            TransferStreams = TransferStreams,
            EnableProfiling = EnableProfiling,
            GraphBatchSize = GraphBatchSize,
            CacheCompiledGraphs = CacheCompiledGraphs
        };
    }

    /// <summary>
    /// Validates the options and throws if invalid.
    /// </summary>
    public void Validate()
    {
        if (MinGpuElements < 0)
        {
            throw new ArgumentException("MinGpuElements must be non-negative.", nameof(MinGpuElements));
        }

        if (MaxComputeStreams < 1)
        {
            throw new ArgumentException("MaxComputeStreams must be at least 1.", nameof(MaxComputeStreams));
        }

        if (MaxMemoryUsage < 0 || MaxMemoryUsage > 1)
        {
            throw new ArgumentException("MaxMemoryUsage must be between 0 and 1.", nameof(MaxMemoryUsage));
        }

        if (TransferStreams < 1)
        {
            throw new ArgumentException("TransferStreams must be at least 1.", nameof(TransferStreams));
        }

        if (GraphBatchSize < 1)
        {
            throw new ArgumentException("GraphBatchSize must be at least 1.", nameof(GraphBatchSize));
        }

        if (ForceGpu && ForceCpu)
        {
            throw new ArgumentException("Cannot set both ForceGpu and ForceCpu.");
        }
    }

    private static bool TryGetEnvInt(string name, out int value)
    {
        var envValue = Environment.GetEnvironmentVariable(EnvPrefix + name);
        if (!string.IsNullOrEmpty(envValue) && int.TryParse(envValue, out value))
        {
            return true;
        }
        value = 0;
        return false;
    }

    private static bool TryGetEnvDouble(string name, out double value)
    {
        var envValue = Environment.GetEnvironmentVariable(EnvPrefix + name);
        if (!string.IsNullOrEmpty(envValue) && double.TryParse(envValue, out value))
        {
            return true;
        }
        value = 0;
        return false;
    }

    private static bool TryGetEnvBool(string name, out bool value)
    {
        var envValue = Environment.GetEnvironmentVariable(EnvPrefix + name);
        if (!string.IsNullOrEmpty(envValue))
        {
            value = envValue.Equals("true", StringComparison.OrdinalIgnoreCase) ||
                    envValue.Equals("1", StringComparison.Ordinal) ||
                    envValue.Equals("yes", StringComparison.OrdinalIgnoreCase);
            return true;
        }
        value = false;
        return false;
    }

    private static bool TryGetEnvEnum<TEnum>(string name, out TEnum value) where TEnum : struct, Enum
    {
        var envValue = Environment.GetEnvironmentVariable(EnvPrefix + name);
        if (!string.IsNullOrEmpty(envValue) && Enum.TryParse<TEnum>(envValue, ignoreCase: true, out value))
        {
            return true;
        }
        value = default;
        return false;
    }
}
