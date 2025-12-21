namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for performance profiling during model training and inference.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Profiling measures how long different parts of your ML code take to run.
/// Think of it like a stopwatch for your code - it helps you find bottlenecks and optimize performance.
///
/// What gets tracked:
/// - Operation timing: How long each training step, forward pass, backward pass takes
/// - Memory allocations: How much memory is used during training
/// - Call hierarchy: Which operations call which other operations
/// - Percentiles: P50 (median), P95, P99 timing for statistical analysis
///
/// Why it's important:
/// - Find bottlenecks in your training pipeline
/// - Compare performance before and after optimizations
/// - Identify memory-intensive operations
/// - Understand your model's computational profile
///
/// The profiler uses production-ready algorithms:
/// - Welford's algorithm for O(1) streaming mean/variance
/// - Reservoir sampling for bounded-memory percentile estimation
/// - Configurable sampling for high-frequency operations
/// </para>
/// </remarks>
public class ProfilingConfig
{
    /// <summary>
    /// Gets or sets whether profiling is enabled (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set to true to collect performance data during training/inference.
    /// Profiling adds some overhead, so it's disabled by default. Enable when debugging performance.
    /// </para>
    /// </remarks>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the sampling rate for high-frequency operations (default: 1.0 = 100%).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> What percentage of operations to profile (0.0 to 1.0).
    /// 1.0 = profile everything, 0.1 = profile 10% of operations.
    /// Use lower values for very hot paths to reduce overhead.
    /// </para>
    /// </remarks>
    public double SamplingRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the reservoir size for percentile estimation (default: 1000).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many samples to keep for calculating P50/P95/P99.
    /// Larger values give more accurate percentiles but use more memory.
    /// 1000 samples is usually sufficient for accurate estimates.
    /// </para>
    /// </remarks>
    public int ReservoirSize { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the maximum number of unique operations to track (default: 10000).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Maximum distinct operation names the profiler will track.
    /// Prevents unbounded memory growth if your code generates many unique operation names.
    /// 10000 is typically more than enough for any ML workload.
    /// </para>
    /// </remarks>
    public int MaxOperations { get; set; } = 10000;

    /// <summary>
    /// Gets or sets whether to track parent-child call relationships (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tracks which operations call which other operations.
    /// Enables call tree analysis to see the hierarchy of operations.
    /// Adds slight overhead but provides valuable context.
    /// </para>
    /// </remarks>
    public bool TrackCallHierarchy { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to track memory allocations (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Records memory usage for each profiled operation.
    /// Helps identify memory-intensive operations during training.
    /// Combined with MemoryTracker for detailed analysis.
    /// </para>
    /// </remarks>
    public bool TrackAllocations { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include detailed timing breakdowns (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Track sub-operation timings within each operation.
    /// More detailed but adds overhead. Use for deep performance analysis.
    /// </para>
    /// </remarks>
    public bool DetailedTiming { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to auto-enable profiling in debug builds (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, profiling is automatically enabled in debug builds
    /// even if Enabled is false. Convenient for development without changing config.
    /// </para>
    /// </remarks>
    public bool AutoEnableInDebug { get; set; } = true;

    /// <summary>
    /// Gets or sets custom tags to include with profiling data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Add custom labels to profiling data (e.g., model name, experiment ID).
    /// Useful for comparing profiles across different runs or configurations.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> CustomTags { get; set; } = new();
}
