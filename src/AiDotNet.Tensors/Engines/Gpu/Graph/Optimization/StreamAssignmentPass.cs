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

        // During optimization, we don't assign actual streams - we leave AssignedStream as null
        // so that ExecutionGraph.GetStreamForNode() will acquire streams at runtime.
        // This allows proper multi-stream execution where streams are acquired and released
        // during graph execution, not during compilation.
        //
        // The ExecutionGraph will:
        // 1. Acquire streams from the pool for compute operations
        // 2. Track acquired streams
        // 3. Release them after execution completes
        //
        // Returning null here tells the execution layer to dynamically acquire a stream.
        return null;
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
