using AiDotNet.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Engines.Gpu.Graph;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for GPU execution optimization system (Phases 2-3).
/// Tests graph construction, configuration, and cache performance.
/// Target: Verify low overhead for graph compilation infrastructure.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net90)]
public class GpuExecutionBenchmarks
{
    // Matrix dimensions for benchmarks
    [Params(256, 1024, 4096)]
    public int MatrixSize { get; set; }

    [Params(10, 50)]
    public int LayerCount { get; set; }

    private float[] _inputData = null!;
    private float[] _weightData = null!;
    private float[] _biasData = null!;
    private GpuExecutionOptions _eagerOptions = null!;
    private GpuExecutionOptions _deferredOptions = null!;
    private GpuExecutionOptions _fusionOptions = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // Initialize data arrays
        int size = MatrixSize * MatrixSize;
        _inputData = new float[size];
        _weightData = new float[size];
        _biasData = new float[MatrixSize];

        for (int i = 0; i < size; i++)
        {
            _inputData[i] = (float)(random.NextDouble() * 2 - 1);
            _weightData[i] = (float)(random.NextDouble() * 0.1 - 0.05);
        }

        for (int i = 0; i < MatrixSize; i++)
        {
            _biasData[i] = (float)(random.NextDouble() * 0.1);
        }

        // Setup different execution options
        _eagerOptions = new GpuExecutionOptions
        {
            ExecutionMode = GpuExecutionMode.Eager,
            EnableGraphCompilation = false,
            EnableAutoFusion = false,
            EnableComputeTransferOverlap = false,
            MaxComputeStreams = 1
        };

        _deferredOptions = new GpuExecutionOptions
        {
            ExecutionMode = GpuExecutionMode.Deferred,
            EnableGraphCompilation = true,
            EnableAutoFusion = false,
            EnableComputeTransferOverlap = true,
            MaxComputeStreams = 3
        };

        _fusionOptions = new GpuExecutionOptions
        {
            ExecutionMode = GpuExecutionMode.Deferred,
            EnableGraphCompilation = true,
            EnableAutoFusion = true,
            EnableComputeTransferOverlap = true,
            MaxComputeStreams = 3,
            EnablePrefetch = true
        };
    }

    #region Execution Graph Construction Benchmarks

    /// <summary>
    /// Benchmark: Creating an execution graph builder.
    /// </summary>
    [Benchmark]
    public ExecutionGraphBuilder GraphBuilder_Create()
    {
        return new ExecutionGraphBuilder();
    }

    /// <summary>
    /// Benchmark: Building an empty execution graph.
    /// </summary>
    [Benchmark]
    public ExecutionGraph GraphBuilder_BuildEmpty()
    {
        using var builder = new ExecutionGraphBuilder();
        return builder.Build();
    }

    #endregion

    #region GPU Execution Options Configuration Benchmarks

    /// <summary>
    /// Benchmark: Creating GpuExecutionOptions with defaults.
    /// </summary>
    [Benchmark(Baseline = true)]
    public GpuExecutionOptions Options_CreateDefault()
    {
        return new GpuExecutionOptions();
    }

    /// <summary>
    /// Benchmark: Creating GpuExecutionOptions from environment variables.
    /// </summary>
    [Benchmark]
    public GpuExecutionOptions Options_FromEnvironment()
    {
        return GpuExecutionOptions.FromEnvironment();
    }

    /// <summary>
    /// Benchmark: Cloning GpuExecutionOptions.
    /// </summary>
    [Benchmark]
    public GpuExecutionOptions Options_Clone()
    {
        return _fusionOptions.Clone();
    }

    /// <summary>
    /// Benchmark: Validating GpuExecutionOptions.
    /// </summary>
    [Benchmark]
    public void Options_Validate()
    {
        _fusionOptions.Validate();
    }

    #endregion

    #region Configuration Integration Benchmarks

    /// <summary>
    /// Benchmark: Creating GpuAccelerationConfig with defaults.
    /// </summary>
    [Benchmark]
    public GpuAccelerationConfig Config_CreateDefault()
    {
        return new GpuAccelerationConfig();
    }

    /// <summary>
    /// Benchmark: Creating GpuAccelerationConfig with all advanced options.
    /// </summary>
    [Benchmark]
    public GpuAccelerationConfig Config_CreateAdvanced()
    {
        return new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.Auto,
            UsageLevel = GpuUsageLevel.Aggressive,
            ExecutionMode = GpuExecutionModeConfig.Deferred,
            EnableGraphCompilation = true,
            EnableAutoFusion = true,
            EnableComputeTransferOverlap = true,
            MaxComputeStreams = 4,
            MinGpuElements = 2048,
            MaxGpuMemoryUsage = 0.9,
            EnablePrefetch = true,
            CacheCompiledGraphs = true,
            EnableProfiling = false
        };
    }

    /// <summary>
    /// Benchmark: GpuAccelerationConfig.ToString() for logging.
    /// </summary>
    [Benchmark]
    public string Config_ToString()
    {
        var config = new GpuAccelerationConfig
        {
            ExecutionMode = GpuExecutionModeConfig.Deferred,
            EnableGraphCompilation = true,
            EnableAutoFusion = true
        };

        return config.ToString();
    }

    #endregion

    #region Compiled Graph Cache Benchmarks

    /// <summary>
    /// Benchmark: Creating a compiled graph cache.
    /// </summary>
    [Benchmark]
    public CompiledGraphCache Cache_Create()
    {
        return new CompiledGraphCache(maxSize: 100);
    }

    /// <summary>
    /// Benchmark: Cache miss lookup (key not found).
    /// </summary>
    [Benchmark]
    public bool Cache_LookupMiss()
    {
        var cache = new CompiledGraphCache(maxSize: 100);
        return cache.TryGet("nonexistent_key", out _);
    }

    /// <summary>
    /// Benchmark: Cache hit lookup after adding.
    /// </summary>
    [Benchmark]
    public ExecutionGraph? Cache_LookupHit()
    {
        var cache = new CompiledGraphCache(maxSize: 100);

        using var builder = new ExecutionGraphBuilder();
        var graph = builder.Build();
        string key = "test_graph";
        cache.Add(key, graph);

        cache.TryGet(key, out var cachedGraph);
        return cachedGraph;
    }

    /// <summary>
    /// Benchmark: Adding a graph to the cache.
    /// </summary>
    [Benchmark]
    public void Cache_Add()
    {
        var cache = new CompiledGraphCache(maxSize: 100);

        using var builder = new ExecutionGraphBuilder();
        var graph = builder.Build();
        string key = $"test_graph_{Guid.NewGuid()}";

        cache.Add(key, graph);
    }

    /// <summary>
    /// Benchmark: Cache with LRU eviction under pressure.
    /// </summary>
    [Benchmark]
    public int Cache_LruEviction()
    {
        var cache = new CompiledGraphCache(maxSize: 10);

        // Add more than maxSize to trigger eviction
        for (int i = 0; i < 20; i++)
        {
            using var builder = new ExecutionGraphBuilder();
            var graph = builder.Build();
            cache.Add($"graph_{i}", graph);
        }

        return cache.Count;
    }

    #endregion

    #region Execution Graph Properties Benchmarks

    /// <summary>
    /// Benchmark: Getting topological order from empty graph.
    /// </summary>
    [Benchmark]
    public IReadOnlyList<ExecutionNode> ExecutionGraph_EmptyTopologicalOrder()
    {
        using var builder = new ExecutionGraphBuilder();
        var graph = builder.Build();

        return graph.TopologicalOrder;
    }

    /// <summary>
    /// Benchmark: Getting nodes at level from empty graph.
    /// </summary>
    [Benchmark]
    public IReadOnlyList<ExecutionNode> ExecutionGraph_EmptyGetNodesAtLevel()
    {
        using var builder = new ExecutionGraphBuilder();
        var graph = builder.Build();

        return graph.GetNodesAtLevel(0);
    }

    /// <summary>
    /// Benchmark: Getting graph statistics from empty graph.
    /// </summary>
    [Benchmark]
    public (int levels, int critical, int parallel) ExecutionGraph_EmptyStatistics()
    {
        using var builder = new ExecutionGraphBuilder();
        var graph = builder.Build();

        return (graph.LevelCount, graph.CriticalPathLength, graph.MaxParallelism);
    }

    #endregion

    // Note: GPU Tensor Registry benchmarks require an actual GPU backend and are skipped here.
    // These would be run as part of GPU-specific benchmarks when hardware is available.

    #region Threshold Decision Benchmarks

    /// <summary>
    /// Benchmark: GPU threshold decision check (should use GPU).
    /// </summary>
    [Benchmark]
    public bool Threshold_ShouldUseGpu_Large()
    {
        var options = new GpuExecutionOptions { MinGpuElements = 4096 };
        int elementCount = 100000; // Large - should use GPU
        return elementCount >= options.MinGpuElements && !options.ForceCpu;
    }

    /// <summary>
    /// Benchmark: GPU threshold decision check (should use CPU).
    /// </summary>
    [Benchmark]
    public bool Threshold_ShouldUseGpu_Small()
    {
        var options = new GpuExecutionOptions { MinGpuElements = 4096 };
        int elementCount = 1000; // Small - should use CPU
        return elementCount >= options.MinGpuElements && !options.ForceCpu;
    }

    /// <summary>
    /// Benchmark: GPU threshold with force GPU override.
    /// </summary>
    [Benchmark]
    public bool Threshold_ForceGpu()
    {
        var options = new GpuExecutionOptions { MinGpuElements = 4096, ForceGpu = true };
        int elementCount = 100; // Small but forced GPU
        return options.ForceGpu || (elementCount >= options.MinGpuElements && !options.ForceCpu);
    }

    #endregion
}

// Note: GpuConfigConversionBenchmarks removed - ToExecutionOptions() is internal
// and cannot be accessed from the benchmark project. Conversion benchmarks would
// need to be done via internal tests or by making the method public.

/// <summary>
/// Benchmarks for GpuExecutionOptions validation and cloning.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net90)]
public class GpuOptionsOperationsBenchmarks
{
    private GpuExecutionOptions _validOptions = null!;
    private GpuExecutionOptions _complexOptions = null!;

    [GlobalSetup]
    public void Setup()
    {
        _validOptions = new GpuExecutionOptions
        {
            MinGpuElements = 4096,
            MaxComputeStreams = 3,
            EnableGraphCompilation = true
        };

        _complexOptions = new GpuExecutionOptions
        {
            MinGpuElements = 2048,
            MaxComputeStreams = 8,
            ForceGpu = false,
            ForceCpu = false,
            EnableGraphCompilation = true,
            EnableAutoFusion = true,
            MaxMemoryUsage = 0.85,
            EnablePrefetch = true,
            EnableComputeTransferOverlap = true,
            ExecutionMode = GpuExecutionMode.Deferred,
            EnableGpuResidency = true,
            TransferStreams = 4,
            EnableProfiling = true,
            GraphBatchSize = 64,
            CacheCompiledGraphs = true
        };
    }

    [Benchmark(Baseline = true)]
    public void ValidateSimpleOptions()
    {
        _validOptions.Validate();
    }

    [Benchmark]
    public void ValidateComplexOptions()
    {
        _complexOptions.Validate();
    }

    [Benchmark]
    public GpuExecutionOptions CloneSimpleOptions()
    {
        return _validOptions.Clone();
    }

    [Benchmark]
    public GpuExecutionOptions CloneComplexOptions()
    {
        return _complexOptions.Clone();
    }
}
