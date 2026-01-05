using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu.Graph.Optimization;

/// <summary>
/// Optimization pass that plans buffer reuse to minimize memory allocation.
/// Analyzes liveness of buffers and identifies reuse opportunities.
/// </summary>
public sealed class MemoryPlanningPass : IGraphOptimizationPass
{
    /// <inheritdoc/>
    public string Name => "MemoryPlanning";

    /// <inheritdoc/>
    public int Priority => 400; // Run after stream assignment

    /// <inheritdoc/>
    public bool IsEnabled { get; set; } = true;

    /// <inheritdoc/>
    public List<ExecutionNode> Apply(List<ExecutionNode> nodes, OptimizationContext context)
    {
        if (!IsEnabled)
        {
            return nodes;
        }

        // Compute buffer liveness
        var liveness = ComputeBufferLiveness(nodes);

        // Find reuse opportunities
        var reuseMap = ComputeReuseOpportunities(liveness, context);

        // Apply memory planning (doesn't modify nodes, but records for execution)
        context.Statistics.PassStats[Name] = new PassStatistics
        {
            PassName = Name,
            TransformationsApplied = reuseMap.Count
        };

        return nodes;
    }

    private static Dictionary<IGpuBuffer, BufferLiveness> ComputeBufferLiveness(List<ExecutionNode> nodes)
    {
        var liveness = new Dictionary<IGpuBuffer, BufferLiveness>();

        for (int i = 0; i < nodes.Count; i++)
        {
            var node = nodes[i];

            // Record first use (definition) for output buffers
            foreach (var output in node.OutputTensors)
            {
                if (!liveness.ContainsKey(output.Buffer))
                {
                    liveness[output.Buffer] = new BufferLiveness
                    {
                        Buffer = output.Buffer,
                        FirstUseIndex = i,
                        LastUseIndex = i,
                        Size = output.Buffer.Size,
                        Role = output.Role
                    };
                }
            }

            // Record last use for input buffers
            foreach (var input in node.InputTensors)
            {
                if (liveness.TryGetValue(input.Buffer, out var info))
                {
                    info.LastUseIndex = Math.Max(info.LastUseIndex, i);
                }
                else
                {
                    // External input - assume live for entire graph
                    liveness[input.Buffer] = new BufferLiveness
                    {
                        Buffer = input.Buffer,
                        FirstUseIndex = 0,
                        LastUseIndex = nodes.Count - 1,
                        Size = input.Buffer.Size,
                        Role = input.Role,
                        IsExternal = true
                    };
                }
            }
        }

        return liveness;
    }

    private static Dictionary<IGpuBuffer, IGpuBuffer> ComputeReuseOpportunities(
        Dictionary<IGpuBuffer, BufferLiveness> liveness,
        OptimizationContext context)
    {
        var reuseMap = new Dictionary<IGpuBuffer, IGpuBuffer>();

        // Sort by size (largest first) to maximize reuse
        var sortedBuffers = liveness.Values
            .Where(l => !l.IsExternal && l.Role != GpuTensorRole.Weight && l.Role != GpuTensorRole.Bias)
            .OrderByDescending(l => l.Size)
            .ToList();

        var availableBuffers = new List<BufferLiveness>();

        foreach (var buffer in sortedBuffers)
        {
            // Find a reusable buffer that is:
            // 1. Not live when this buffer is first used
            // 2. Large enough to hold this buffer's data
            var reusable = availableBuffers
                .FirstOrDefault(a =>
                    a.LastUseIndex < buffer.FirstUseIndex &&
                    a.Size >= buffer.Size);

            if (reusable != null)
            {
                // Reuse this buffer
                reuseMap[buffer.Buffer] = reusable.Buffer;

                // Update the reused buffer's last use
                reusable.LastUseIndex = buffer.LastUseIndex;
            }
            else
            {
                // Add to available pool for future reuse
                availableBuffers.Add(buffer);
            }
        }

        return reuseMap;
    }

    private sealed class BufferLiveness
    {
        public IGpuBuffer Buffer { get; init; } = null!;
        public int FirstUseIndex { get; set; }
        public int LastUseIndex { get; set; }
        public int Size { get; init; }
        public GpuTensorRole Role { get; init; }
        public bool IsExternal { get; init; }
    }
}

/// <summary>
/// Optimization pass that eliminates dead code (unused computations).
/// </summary>
public sealed class DeadCodeEliminationPass : IGraphOptimizationPass
{
    /// <inheritdoc/>
    public string Name => "DeadCodeElimination";

    /// <inheritdoc/>
    public int Priority => 50; // Run very early

    /// <inheritdoc/>
    public bool IsEnabled { get; set; } = true;

    /// <inheritdoc/>
    public List<ExecutionNode> Apply(List<ExecutionNode> nodes, OptimizationContext context)
    {
        if (!IsEnabled)
        {
            return nodes;
        }

        // Find all nodes that produce outputs needed by other nodes or are terminal outputs
        var neededNodes = new HashSet<int>();

        // Mark all nodes with dependents as needed
        foreach (var node in nodes)
        {
            if (node.Dependents.Count > 0)
            {
                neededNodes.Add(node.NodeId);
            }
        }

        // Mark terminal nodes (D2H transfers, barriers) as needed
        foreach (var node in nodes)
        {
            if (node.NodeType == ExecutionNodeType.TransferD2H ||
                node.NodeType == ExecutionNodeType.Barrier)
            {
                MarkNeeded(node, neededNodes);
            }
        }

        // Mark the last few nodes as needed (they're probably the outputs)
        int skipCount = Math.Max(0, nodes.Count - 3);
        var lastNodes = nodes.Skip(skipCount);
        foreach (var node in lastNodes)
        {
            MarkNeeded(node, neededNodes);
        }

        // Filter out unneeded nodes
        var result = nodes.Where(n => neededNodes.Contains(n.NodeId)).ToList();

        int eliminated = nodes.Count - result.Count;
        context.Statistics.NodesEliminated += eliminated;

        if (context.Statistics.PassStats.TryGetValue(Name, out var stats))
        {
            stats.TransformationsApplied = eliminated;
            stats.NodeCountDelta = -eliminated;
        }

        return result;
    }

    private static void MarkNeeded(ExecutionNode node, HashSet<int> needed)
    {
        if (!needed.Add(node.NodeId))
        {
            return; // Already marked
        }

        // Mark all dependencies as needed
        foreach (var dep in node.Dependencies)
        {
            MarkNeeded(dep, needed);
        }
    }
}

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
