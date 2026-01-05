namespace AiDotNet.Tensors.Engines.Gpu.Graph.Optimization;

/// <summary>
/// Optimization pass that reorders operations to maximize parallelism.
/// </summary>
public sealed class OperationReorderingPass : IGraphOptimizationPass
{
    /// <inheritdoc/>
    public string Name => "OperationReordering";

    /// <inheritdoc/>
    public int Priority => 200; // Run after fusion, before stream assignment

    /// <inheritdoc/>
    public bool IsEnabled { get; set; } = true;

    /// <inheritdoc/>
    public List<ExecutionNode> Apply(List<ExecutionNode> nodes, OptimizationContext context)
    {
        if (!IsEnabled)
        {
            return nodes;
        }

        // Compute critical path
        var criticalPath = ComputeCriticalPath(nodes);

        // Prioritize critical path nodes
        var result = ReorderForCriticalPath(nodes, criticalPath);

        // Move transfers to maximize overlap
        result = OptimizeTransferOrder(result);

        // Update statistics
        if (context.Statistics.PassStats.TryGetValue(Name, out var stats))
        {
            stats.TransformationsApplied = nodes.Count; // Processed all nodes
        }

        return result;
    }

    private static HashSet<int> ComputeCriticalPath(List<ExecutionNode> nodes)
    {
        var criticalPath = new HashSet<int>();
        var nodeCosts = new Dictionary<int, int>();

        // Compute cost from each node to the end
        void ComputeCostToEnd(ExecutionNode node)
        {
            if (nodeCosts.ContainsKey(node.NodeId))
            {
                return;
            }

            foreach (var dep in node.Dependents)
            {
                ComputeCostToEnd(dep);
            }

            int maxDepCost = node.Dependents
                .Select(d => nodeCosts.TryGetValue(d.NodeId, out var c) ? c : 0)
                .DefaultIfEmpty(0)
                .Max();

            nodeCosts[node.NodeId] = node.EstimatedCost + maxDepCost;
        }

        foreach (var node in nodes)
        {
            ComputeCostToEnd(node);
        }

        // Find nodes on the critical path (highest total cost at each level)
        if (nodes.Count > 0)
        {
            var maxCost = nodeCosts.Values.Max();
            var startNodes = nodes.Where(n =>
                nodeCosts.TryGetValue(n.NodeId, out var c) && c == maxCost).ToList();

            void TracePath(ExecutionNode node)
            {
                criticalPath.Add(node.NodeId);

                if (node.Dependents.Count > 0)
                {
                    var nextNode = node.Dependents
                        .OrderByDescending(d => nodeCosts.TryGetValue(d.NodeId, out var c) ? c : 0)
                        .First();
                    TracePath(nextNode);
                }
            }

            foreach (var start in startNodes)
            {
                TracePath(start);
            }
        }

        return criticalPath;
    }

    private static List<ExecutionNode> ReorderForCriticalPath(
        List<ExecutionNode> nodes,
        HashSet<int> criticalPath)
    {
        // Topological sort with critical path priority
        var visited = new HashSet<int>();
        var result = new List<ExecutionNode>();

        void Visit(ExecutionNode node)
        {
            if (visited.Contains(node.NodeId))
            {
                return;
            }

            visited.Add(node.NodeId);

            // Visit dependencies first, prioritizing non-critical path
            var deps = node.Dependencies
                .OrderBy(d => criticalPath.Contains(d.NodeId) ? 1 : 0)
                .ThenBy(d => d.EstimatedCost);

            foreach (var dep in deps)
            {
                Visit(dep);
            }

            result.Add(node);
        }

        // Start with critical path nodes
        var sortedNodes = nodes
            .OrderByDescending(n => criticalPath.Contains(n.NodeId))
            .ThenByDescending(n => n.EstimatedCost);

        foreach (var node in sortedNodes)
        {
            Visit(node);
        }

        return result;
    }

    private static List<ExecutionNode> OptimizeTransferOrder(List<ExecutionNode> nodes)
    {
        if (nodes.Count < 2)
        {
            return nodes;
        }

        var result = new List<ExecutionNode>(nodes);
        var nodePositions = BuildPositionMap(result);

        // Process H2D transfers: move as early as possible (prefetch)
        var h2dNodes = result
            .Where(n => n.NodeType == ExecutionNodeType.TransferH2D)
            .ToList();

        foreach (var h2d in h2dNodes)
        {
            int currentPos = nodePositions[h2d.NodeId];
            int earliestPos = ComputeEarliestValidPosition(h2d, nodePositions);

            if (earliestPos < currentPos)
            {
                MoveNode(result, nodePositions, currentPos, earliestPos);
            }
        }

        // Process D2H transfers: move as late as possible (delay until needed)
        var d2hNodes = result
            .Where(n => n.NodeType == ExecutionNodeType.TransferD2H)
            .ToList();

        foreach (var d2h in d2hNodes)
        {
            int currentPos = nodePositions[d2h.NodeId];
            int latestPos = ComputeLatestValidPosition(d2h, nodePositions, result.Count);

            if (latestPos > currentPos)
            {
                MoveNode(result, nodePositions, currentPos, latestPos);
            }
        }

        return result;
    }

    private static Dictionary<int, int> BuildPositionMap(List<ExecutionNode> nodes)
    {
        var positions = new Dictionary<int, int>();
        for (int i = 0; i < nodes.Count; i++)
        {
            positions[nodes[i].NodeId] = i;
        }
        return positions;
    }

    private static int ComputeEarliestValidPosition(ExecutionNode node, Dictionary<int, int> positions)
    {
        if (node.Dependencies.Count == 0)
        {
            return 0;
        }

        int earliestPos = 0;
        foreach (var dep in node.Dependencies)
        {
            if (positions.TryGetValue(dep.NodeId, out int depPos))
            {
                earliestPos = Math.Max(earliestPos, depPos + 1);
            }
        }

        return earliestPos;
    }

    private static int ComputeLatestValidPosition(ExecutionNode node, Dictionary<int, int> positions, int maxPos)
    {
        if (node.Dependents.Count == 0)
        {
            return maxPos - 1;
        }

        int latestPos = maxPos - 1;
        foreach (var dependent in node.Dependents)
        {
            if (positions.TryGetValue(dependent.NodeId, out int depPos))
            {
                latestPos = Math.Min(latestPos, depPos - 1);
            }
        }

        return Math.Max(0, latestPos);
    }

    private static void MoveNode(
        List<ExecutionNode> nodes,
        Dictionary<int, int> positions,
        int fromIndex,
        int toIndex)
    {
        if (fromIndex == toIndex || fromIndex < 0 || toIndex < 0 ||
            fromIndex >= nodes.Count || toIndex >= nodes.Count)
        {
            return;
        }

        var node = nodes[fromIndex];
        nodes.RemoveAt(fromIndex);
        nodes.Insert(toIndex, node);

        // Update position map for affected nodes
        if (fromIndex < toIndex)
        {
            for (int i = fromIndex; i <= toIndex; i++)
            {
                positions[nodes[i].NodeId] = i;
            }
        }
        else
        {
            for (int i = toIndex; i <= fromIndex; i++)
            {
                positions[nodes[i].NodeId] = i;
            }
        }
    }
}
