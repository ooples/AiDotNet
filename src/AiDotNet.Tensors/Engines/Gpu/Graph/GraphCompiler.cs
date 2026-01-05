using System.Diagnostics;
using AiDotNet.Tensors.Engines.Gpu.Graph.Optimization;

namespace AiDotNet.Tensors.Engines.Gpu.Graph;

/// <summary>
/// Compiles and optimizes execution graphs for efficient GPU execution.
/// Orchestrates multiple optimization passes and produces optimized graphs.
/// </summary>
public sealed class GraphCompiler
{
    private readonly GpuExecutionOptions _options;
    private readonly List<IGraphOptimizationPass> _passes;

    /// <summary>
    /// Gets the optimization passes registered with this compiler.
    /// </summary>
    public IReadOnlyList<IGraphOptimizationPass> Passes => _passes;

    /// <summary>
    /// Creates a new graph compiler with default optimization passes.
    /// </summary>
    /// <param name="options">Execution options controlling optimization behavior.</param>
    public GraphCompiler(GpuExecutionOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _passes = CreateDefaultPasses(options);
    }

    /// <summary>
    /// Creates a graph compiler with custom optimization passes.
    /// </summary>
    /// <param name="options">Execution options.</param>
    /// <param name="passes">Custom passes to use.</param>
    public GraphCompiler(GpuExecutionOptions options, IEnumerable<IGraphOptimizationPass> passes)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _passes = passes?.OrderBy(p => p.Priority).ToList() ?? new List<IGraphOptimizationPass>();
    }

    /// <summary>
    /// Compiles and optimizes a list of execution nodes into an optimized graph.
    /// </summary>
    /// <param name="nodes">The nodes to compile.</param>
    /// <param name="backend">The async GPU backend.</param>
    /// <returns>An optimized execution graph.</returns>
    public ExecutionGraph Compile(List<ExecutionNode> nodes, IAsyncGpuBackend backend)
    {
        var (graph, _) = CompileWithStatistics(nodes, backend);
        return graph;
    }

    /// <summary>
    /// Compiles and optimizes a list of execution nodes, returning both the graph and optimization statistics.
    /// </summary>
    /// <param name="nodes">The nodes to compile.</param>
    /// <param name="backend">The async GPU backend.</param>
    /// <returns>A tuple containing the optimized execution graph and optimization statistics.</returns>
    public (ExecutionGraph Graph, OptimizationStatistics Statistics) CompileWithStatistics(List<ExecutionNode> nodes, IAsyncGpuBackend backend)
    {
        if (nodes == null)
        {
            throw new ArgumentNullException(nameof(nodes));
        }

        if (backend == null)
        {
            throw new ArgumentNullException(nameof(backend));
        }

        var context = new OptimizationContext(_options, backend);
        context.Statistics.OriginalNodeCount = nodes.Count;

        var sw = Stopwatch.StartNew();

        GpuStreamPool? streamPool = null;
        try
        {
            // Create stream pool if needed
            if (_options.EnableComputeTransferOverlap)
            {
                streamPool = new GpuStreamPool(backend, _options);
                context.StreamPool = streamPool;
            }

            // Initialize pass statistics
            foreach (var pass in _passes)
            {
                context.Statistics.PassStats[pass.Name] = new PassStatistics { PassName = pass.Name };
            }

            // Run optimization passes
            var optimizedNodes = nodes;
            foreach (var pass in _passes)
            {
                if (!pass.IsEnabled)
                {
                    continue;
                }

                var passSw = Stopwatch.StartNew();
                int beforeCount = optimizedNodes.Count;

                optimizedNodes = pass.Apply(optimizedNodes, context);

                passSw.Stop();

                if (context.Statistics.PassStats.TryGetValue(pass.Name, out var stats))
                {
                    stats.Duration = passSw.Elapsed;
                    if (stats.NodeCountDelta == 0)
                    {
                        stats.NodeCountDelta = optimizedNodes.Count - beforeCount;
                    }
                }
            }

            sw.Stop();

            // Compute final statistics
            context.Statistics.OptimizedNodeCount = optimizedNodes.Count;
            context.Statistics.OptimizationTime = sw.Elapsed;

            int originalCost = nodes.Sum(n => n.EstimatedCost);
            int optimizedCost = optimizedNodes.Sum(n => n.EstimatedCost);
            context.Statistics.EstimatedCostReduction = originalCost > 0
                ? (double)(originalCost - optimizedCost) / originalCost
                : 0;

            // Create the optimized graph
            return (new ExecutionGraph(optimizedNodes), context.Statistics);
        }
        finally
        {
            // Dispose the stream pool after compilation
            streamPool?.Dispose();
        }
    }

    /// <summary>
    /// Compiles a graph builder's recorded operations.
    /// </summary>
    /// <param name="builder">The graph builder with recorded operations.</param>
    /// <param name="backend">The async GPU backend.</param>
    /// <returns>An optimized execution graph.</returns>
    public ExecutionGraph Compile(ExecutionGraphBuilder builder, IAsyncGpuBackend backend)
    {
        if (builder == null)
        {
            throw new ArgumentNullException(nameof(builder));
        }

        return Compile(builder.Nodes.ToList(), backend);
    }

    /// <summary>
    /// Adds a custom optimization pass.
    /// </summary>
    /// <param name="pass">The pass to add.</param>
    public void AddPass(IGraphOptimizationPass pass)
    {
        if (pass == null)
        {
            throw new ArgumentNullException(nameof(pass));
        }

        _passes.Add(pass);
        _passes.Sort((a, b) => a.Priority.CompareTo(b.Priority));
    }

    /// <summary>
    /// Removes an optimization pass by name.
    /// </summary>
    /// <param name="passName">The name of the pass to remove.</param>
    /// <returns>True if a pass was removed.</returns>
    public bool RemovePass(string passName)
    {
        return _passes.RemoveAll(p => p.Name == passName) > 0;
    }

    /// <summary>
    /// Enables or disables a pass by name.
    /// </summary>
    /// <param name="passName">The name of the pass.</param>
    /// <param name="enabled">Whether to enable the pass.</param>
    public void SetPassEnabled(string passName, bool enabled)
    {
        var pass = _passes.FirstOrDefault(p => p.Name == passName);
        if (pass != null)
        {
            pass.IsEnabled = enabled;
        }
    }

    private static List<IGraphOptimizationPass> CreateDefaultPasses(GpuExecutionOptions options)
    {
        var passes = new List<IGraphOptimizationPass>
        {
            new DeadCodeEliminationPass { IsEnabled = true },
            new KernelFusionPass { IsEnabled = options.EnableAutoFusion },
            new OperationReorderingPass { IsEnabled = true },
            new StreamAssignmentPass { IsEnabled = options.EnableComputeTransferOverlap },
            new MemoryPlanningPass { IsEnabled = true },
            new PrefetchPass { IsEnabled = options.EnablePrefetch }
        };

        return passes.OrderBy(p => p.Priority).ToList();
    }
}
