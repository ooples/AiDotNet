namespace AiDotNet.Tensors.Engines.Gpu.Graph.Optimization;

/// <summary>
/// Optimization pass that prefetches data to hide transfer latency.
/// </summary>
public sealed class PrefetchPass : IGraphOptimizationPass
{
    /// <inheritdoc/>
    public string Name => "Prefetch";

    /// <inheritdoc/>
    public int Priority => 500; // Run late

    /// <inheritdoc/>
    public bool IsEnabled { get; set; } = true;

    /// <inheritdoc/>
    public List<ExecutionNode> Apply(List<ExecutionNode> nodes, OptimizationContext context)
    {
        if (!IsEnabled || !context.Options.EnablePrefetch)
        {
            return nodes;
        }

        // Find H2D transfers that can be moved earlier
        var h2dNodes = nodes.OfType<TransferNode>()
            .Where(n => n.TransferType == TransferDirection.HostToDevice)
            .ToList();

        if (h2dNodes.Count == 0)
        {
            return nodes;
        }

        // For each H2D transfer, find the earliest point it can be scheduled
        // (after all its dependencies)
        var result = new List<ExecutionNode>(nodes);
        int prefetchCount = 0;

        foreach (var h2d in h2dNodes)
        {
            int currentIndex = result.IndexOf(h2d);
            if (currentIndex < 0)
            {
                continue;
            }

            // Find earliest valid position
            int earliestIndex = 0;
            foreach (var dep in h2d.Dependencies)
            {
                int depIndex = result.IndexOf(dep);
                if (depIndex >= 0)
                {
                    earliestIndex = Math.Max(earliestIndex, depIndex + 1);
                }
            }

            // Move if beneficial
            if (earliestIndex < currentIndex - 1) // Worth moving at least 2 positions
            {
                result.RemoveAt(currentIndex);
                result.Insert(earliestIndex, h2d);
                prefetchCount++;
            }
        }

        if (context.Statistics.PassStats.TryGetValue(Name, out var stats))
        {
            stats.TransformationsApplied = prefetchCount;
        }

        return result;
    }
}
