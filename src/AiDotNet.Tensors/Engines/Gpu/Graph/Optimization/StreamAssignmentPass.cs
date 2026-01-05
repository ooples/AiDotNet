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

        // Compute topological order with cycle detection using three-color DFS
        // - Not visited: node not in visiting or visited
        // - Visiting (gray): node is in visiting set (currently on recursion stack)
        // - Visited (black): node is in visited set (fully processed)
        var visiting = new HashSet<int>();
        var visited = new HashSet<int>();
        var order = new List<ExecutionNode>();

        bool Visit(ExecutionNode node)
        {
            if (visited.Contains(node.NodeId))
            {
                return true; // Already processed, no cycle here
            }

            if (visiting.Contains(node.NodeId))
            {
                // Node is on the current recursion stack - cycle detected!
                throw new InvalidOperationException(
                    $"Circular dependency detected involving node {node.Name} (NodeId: {node.NodeId}). " +
                    "Cannot compute topological order for stream assignment.");
            }

            visiting.Add(node.NodeId);

            foreach (var dep in node.Dependencies)
            {
                if (!Visit(dep))
                {
                    return false;
                }
            }

            visiting.Remove(node.NodeId);
            visited.Add(node.NodeId);
            order.Add(node);
            return true;
        }

        foreach (var node in nodes)
        {
            if (!visited.Contains(node.NodeId))
            {
                Visit(node);
            }
        }

        // Compute levels based on topological order
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

        // Assign the default compute stream for all compute operations.
        // This ensures InsertSynchronizationNodes can properly insert barriers
        // for cross-stream dependencies between compute and transfer operations.
        //
        // Note: We assign streams during optimization (not runtime) because:
        // 1. It enables proper barrier insertion for cross-stream synchronization
        // 2. Transfer streams (H2D, D2H) run on separate default streams, enabling
        //    compute/transfer overlap which is the primary performance benefit
        // 3. Fine-grained multi-stream compute parallelism can be added later if needed
        //
        // The primary optimization goal is compute/transfer overlap, which is achieved
        // by having compute on DefaultComputeStream and transfers on DefaultH2DStream
        // and DefaultD2HStream. This configuration allows uploads, compute, and downloads
        // to overlap in the GPU pipeline.

        return pool.DefaultComputeStream;
    }

    private List<ExecutionNode> InsertSynchronizationNodes(
        List<ExecutionNode> nodes,
        Dictionary<int, int> nodeLevels)
    {
        var result = new List<ExecutionNode>();

        // Get unique streams (excluding null)
        var uniqueStreams = nodes
            .Select(n => n.AssignedStream)
            .Where(s => s != null)
            .Distinct()
            .ToList();

        if (uniqueStreams.Count <= 1)
        {
            // Single stream (or all null), no explicit sync needed
            return nodes;
        }

        // Group nodes by level
        var levelGroups = nodes
            .GroupBy(n => nodeLevels.TryGetValue(n.NodeId, out var l) ? l : 0)
            .OrderBy(g => g.Key)
            .ToList();

        // Track which streams have been used at each level for sync point insertion
        var lastLevelWithActivity = new Dictionary<IGpuStream, int>();

        foreach (var levelGroup in levelGroups)
        {
            int currentLevel = levelGroup.Key;
            var levelNodes = levelGroup.ToList();

            // Collect all cross-stream dependencies for this level
            var crossStreamSyncNeeded = new Dictionary<IGpuStream, HashSet<IGpuStream>>();

            foreach (var node in levelNodes)
            {
                if (node.AssignedStream == null)
                {
                    continue;
                }

                // Find dependencies on other streams
                var crossStreamDeps = node.Dependencies
                    .Where(d => d.AssignedStream != null &&
                                d.AssignedStream != node.AssignedStream)
                    .ToList();

                if (crossStreamDeps.Count > 0)
                {
                    if (!crossStreamSyncNeeded.TryGetValue(node.AssignedStream, out var waitForStreams))
                    {
                        waitForStreams = new HashSet<IGpuStream>();
                        crossStreamSyncNeeded[node.AssignedStream] = waitForStreams;
                    }

                    foreach (var dep in crossStreamDeps)
                    {
                        if (dep.AssignedStream != null)
                        {
                            waitForStreams.Add(dep.AssignedStream);
                        }
                    }
                }
            }

            // Insert barrier nodes for cross-stream synchronization before the level's nodes
            foreach (var kvp in crossStreamSyncNeeded)
            {
                IGpuStream targetStream = kvp.Key;
                var streamsToWaitFor = kvp.Value.ToList();

                if (streamsToWaitFor.Count > 0)
                {
                    // Create a barrier node that synchronizes the required streams
                    var barrier = new BarrierNode(streamsToWaitFor);
                    barrier.AssignedStream = targetStream;

                    // Add dependencies on the last nodes from the source streams
                    foreach (var srcStream in streamsToWaitFor)
                    {
                        var lastNodeOnStream = result
                            .LastOrDefault(n => n.AssignedStream == srcStream);

                        if (lastNodeOnStream != null)
                        {
                            barrier.AddDependency(lastNodeOnStream);
                        }
                    }

                    result.Add(barrier);

                    // Make nodes on this stream at this level depend on the barrier
                    foreach (var node in levelNodes.Where(n => n.AssignedStream == targetStream))
                    {
                        node.AddDependency(barrier);
                    }
                }
            }

            // Add all nodes for this level
            foreach (var node in levelNodes)
            {
                result.Add(node);

                // Track stream activity
                if (node.AssignedStream != null)
                {
                    lastLevelWithActivity[node.AssignedStream] = currentLevel;
                }
            }
        }

        return result;
    }
}
