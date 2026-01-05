namespace AiDotNet.Tensors.Engines.Gpu.Graph.Optimization;

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
