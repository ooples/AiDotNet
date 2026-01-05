namespace AiDotNet.Tensors.Engines.Gpu.Graph.Optimization;

/// <summary>
/// Interface for graph optimization passes.
/// Each pass transforms the graph to improve execution efficiency.
/// </summary>
public interface IGraphOptimizationPass
{
    /// <summary>
    /// Gets the name of this optimization pass.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the priority of this pass (lower = runs earlier).
    /// </summary>
    int Priority { get; }

    /// <summary>
    /// Gets or sets whether this pass is enabled.
    /// </summary>
    bool IsEnabled { get; set; }

    /// <summary>
    /// Applies the optimization pass to the graph.
    /// </summary>
    /// <param name="nodes">The nodes to optimize.</param>
    /// <param name="context">The optimization context.</param>
    /// <returns>The optimized list of nodes.</returns>
    List<ExecutionNode> Apply(List<ExecutionNode> nodes, OptimizationContext context);
}

/// <summary>
/// Context for optimization passes containing shared state and configuration.
/// </summary>
public sealed class OptimizationContext
{
    /// <summary>
    /// Gets the execution options.
    /// </summary>
    public GpuExecutionOptions Options { get; }

    /// <summary>
    /// Gets the async GPU backend.
    /// </summary>
    public IAsyncGpuBackend Backend { get; }

    /// <summary>
    /// Gets or sets statistics collected during optimization.
    /// </summary>
    public OptimizationStatistics Statistics { get; }

    /// <summary>
    /// Gets the stream pool for stream assignment.
    /// </summary>
    public GpuStreamPool? StreamPool { get; set; }

    /// <summary>
    /// Creates a new optimization context.
    /// </summary>
    public OptimizationContext(GpuExecutionOptions options, IAsyncGpuBackend backend)
    {
        Options = options ?? throw new ArgumentNullException(nameof(options));
        Backend = backend ?? throw new ArgumentNullException(nameof(backend));
        Statistics = new OptimizationStatistics();
    }
}

/// <summary>
/// Statistics collected during graph optimization.
/// </summary>
public sealed class OptimizationStatistics
{
    /// <summary>
    /// Number of nodes before optimization.
    /// </summary>
    public int OriginalNodeCount { get; set; }

    /// <summary>
    /// Number of nodes after optimization.
    /// </summary>
    public int OptimizedNodeCount { get; set; }

    /// <summary>
    /// Number of nodes fused.
    /// </summary>
    public int NodesFused { get; set; }

    /// <summary>
    /// Number of nodes eliminated.
    /// </summary>
    public int NodesEliminated { get; set; }

    /// <summary>
    /// Estimated cost reduction percentage.
    /// </summary>
    public double EstimatedCostReduction { get; set; }

    /// <summary>
    /// Time spent in optimization passes.
    /// </summary>
    public TimeSpan OptimizationTime { get; set; }

    /// <summary>
    /// Individual pass statistics.
    /// </summary>
    public Dictionary<string, PassStatistics> PassStats { get; } = new();

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"Optimization: {OriginalNodeCount} -> {OptimizedNodeCount} nodes " +
               $"({NodesFused} fused, {NodesEliminated} eliminated, " +
               $"{EstimatedCostReduction:P1} cost reduction) in {OptimizationTime.TotalMilliseconds:F1}ms";
    }
}

/// <summary>
/// Statistics for a single optimization pass.
/// </summary>
public sealed class PassStatistics
{
    /// <summary>
    /// Name of the pass.
    /// </summary>
    public string PassName { get; init; } = string.Empty;

    /// <summary>
    /// Time spent in this pass.
    /// </summary>
    public TimeSpan Duration { get; set; }

    /// <summary>
    /// Number of transformations applied.
    /// </summary>
    public int TransformationsApplied { get; set; }

    /// <summary>
    /// Node count change from this pass.
    /// </summary>
    public int NodeCountDelta { get; set; }
}
