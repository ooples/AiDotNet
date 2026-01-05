namespace AiDotNet.Tensors.Engines.Gpu.Graph.Optimization;

/// <summary>
/// Optimization pass that assigns nodes to GPU streams for parallel execution.
/// Aims to maximize compute/transfer overlap and multi-stream parallelism.
/// </summary>
public sealed class StreamAssignmentPass : IGraphOptimizationPass
{
    /// <inheritdoc/>
    public string Name => "StreamAssignment";

    /// <inheritdoc/>
    public int Priority => 300; // Run after fusion

    /// <inheritdoc/>
    public bool IsEnabled { get; set; } = true;

    /// <inheritdoc/>
    public List<ExecutionNode> Apply(List<ExecutionNode> nodes, OptimizationContext context)
    {
        if (!IsEnabled || !context.Options.EnableComputeTransferOverlap)
        {
            // Assign all to default stream
            foreach (var node in nodes)
            {
                node.AssignedStream = context.StreamPool?.DefaultComputeStream;
            }
            return nodes;
        }

        // Compute levels for parallel execution analysis
        var nodeLevels = ComputeNodeLevels(nodes);
        var maxComputeStreams = context.Options.MaxComputeStreams;
        int assignmentCount = 0;

        // Assign streams based on node type and level
        foreach (var node in nodes)
        {
            var streamType = GetOptimalStreamType(node);
            node.AssignedStream = streamType switch
            {
                GpuStreamType.HostToDevice => context.StreamPool?.DefaultH2DStream,
                GpuStreamType.DeviceToHost => context.StreamPool?.DefaultD2HStream,
                _ => AssignComputeStream(node, nodeLevels, context.StreamPool, maxComputeStreams)
            };
            assignmentCount++;
        }

        // Add synchronization nodes where needed
        var result = InsertSynchronizationNodes(nodes, nodeLevels);

        // Update statistics
        if (context.Statistics.PassStats.TryGetValue(Name, out var stats))
        {
            stats.TransformationsApplied = assignmentCount;
        }

        return result;
    }

    private static Dictionary<int, int> ComputeNodeLevels(List<ExecutionNode> nodes)
    {
        var levels = new Dictionary<int, int>();

        // Compute topological order
        var visited = new HashSet<int>();
        var order = new List<ExecutionNode>();

        void Visit(ExecutionNode node)
        {
            if (visited.Contains(node.NodeId))
            {
                return;
            }

            visited.Add(node.NodeId);

            foreach (var dep in node.Dependencies)
            {
                Visit(dep);
            }

            order.Add(node);
        }

        foreach (var node in nodes)
        {
            Visit(node);
        }

        // Compute levels
        foreach (var node in order)
        {
            int level = 0;
            foreach (var dep in node.Dependencies)
            {
                if (levels.TryGetValue(dep.NodeId, out var depLevel))
                {
                    level = Math.Max(level, depLevel + 1);
                }
            }
            levels[node.NodeId] = level;
        }

        return levels;
    }

    private static GpuStreamType GetOptimalStreamType(ExecutionNode node)
    {
        return node.NodeType switch
        {
            ExecutionNodeType.TransferH2D => GpuStreamType.HostToDevice,
            ExecutionNodeType.TransferD2H => GpuStreamType.DeviceToHost,
            ExecutionNodeType.Copy => GpuStreamType.DeviceToDevice,
            _ => GpuStreamType.Compute
        };
    }

    private static IGpuStream? AssignComputeStream(
        ExecutionNode node,
        Dictionary<int, int> nodeLevels,
        GpuStreamPool? pool,
        int maxStreams)
    {
        if (pool == null)
        {
            return null;
        }

        // For nodes at the same level, distribute across streams
        if (!nodeLevels.TryGetValue(node.NodeId, out var level))
        {
            return pool.DefaultComputeStream;
        }

        // Simple round-robin for nodes at same level
        // More sophisticated: consider dependencies and data locality
        int streamIdx = level % maxStreams;

        // Use default for first stream, acquire for others
        return streamIdx == 0
            ? pool.DefaultComputeStream
            : pool.AcquireStream(GpuStreamType.Compute);
    }

    private List<ExecutionNode> InsertSynchronizationNodes(
        List<ExecutionNode> nodes,
        Dictionary<int, int> nodeLevels)
    {
        var result = new List<ExecutionNode>();
        var streamGroups = nodes.GroupBy(n => n.AssignedStream).ToList();

        if (streamGroups.Count <= 1)
        {
            // Single stream, no sync needed
            return nodes;
        }

        // Group nodes by level
        var levelGroups = nodes
            .GroupBy(n => nodeLevels.TryGetValue(n.NodeId, out var l) ? l : 0)
            .OrderBy(g => g.Key)
            .ToList();

        foreach (var levelGroup in levelGroups)
        {
            var levelNodes = levelGroup.ToList();

            // Check if this level has cross-stream dependencies from previous level
            var prevLevelNodes = result.Where(n =>
                nodeLevels.TryGetValue(n.NodeId, out var l) && l == levelGroup.Key - 1).ToList();

            // For each node in this level, check if it needs to wait for nodes on other streams
            foreach (var node in levelNodes)
            {
                var crossStreamDeps = node.Dependencies
                    .Where(d => d.AssignedStream != node.AssignedStream)
                    .ToList();

                // Dependencies are handled by the ExecutionNode.ExecuteAsync method
                // which uses events for cross-stream sync
                result.Add(node);
            }
        }

        return result;
    }
}

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
        // Note: Naive reordering (moving all H2D to start and D2H to end) was removed
        // because it ignores the dependency graph and can violate topological order.
        // For example, if a D2H transfer result is needed by a subsequent compute node,
        // or if an H2D transfer depends on a prior computation.
        //
        // The correct approach is to preserve the topological ordering from ReorderForCriticalPath
        // and let the stream assignment handle parallelism. A more sophisticated implementation
        // would need to compute valid insertion points based on dependency constraints.
        //
        // For now, preserve the topologically-sorted order.
        return nodes;
    }
}
