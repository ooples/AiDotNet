using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu.Graph;

/// <summary>
/// Represents a compiled execution graph of GPU operations.
/// Provides topological execution with optional stream parallelism.
/// </summary>
public sealed class ExecutionGraph : IDisposable
{
    private readonly List<ExecutionNode> _nodes;
    private readonly List<ExecutionNode> _topologicalOrder;
    private readonly Dictionary<int, List<ExecutionNode>> _levelNodes;
    private bool _disposed;

    /// <summary>
    /// Gets all nodes in the graph.
    /// </summary>
    public IReadOnlyList<ExecutionNode> Nodes => _nodes;

    /// <summary>
    /// Gets nodes in topological execution order.
    /// </summary>
    public IReadOnlyList<ExecutionNode> TopologicalOrder => _topologicalOrder;

    /// <summary>
    /// Gets the number of execution levels (for parallel execution analysis).
    /// </summary>
    public int LevelCount => _levelNodes.Count;

    /// <summary>
    /// Gets the total estimated execution cost.
    /// </summary>
    public int TotalEstimatedCost { get; }

    /// <summary>
    /// Gets the critical path length (longest dependency chain).
    /// </summary>
    public int CriticalPathLength { get; }

    /// <summary>
    /// Gets the potential parallelism (max nodes at any level).
    /// </summary>
    public int MaxParallelism { get; }

    /// <summary>
    /// Creates a new execution graph from a list of nodes.
    /// </summary>
    /// <param name="nodes">The nodes in the graph.</param>
    internal ExecutionGraph(List<ExecutionNode> nodes)
    {
        _nodes = nodes ?? throw new ArgumentNullException(nameof(nodes));
        _topologicalOrder = ComputeTopologicalOrder();
        _levelNodes = ComputeLevels();

        TotalEstimatedCost = _nodes.Sum(n => n.EstimatedCost);
        CriticalPathLength = _levelNodes.Count;
        MaxParallelism = _levelNodes.Values.Max(level => level.Count);
    }

    /// <summary>
    /// Executes the graph synchronously on the default stream.
    /// </summary>
    /// <param name="backend">The GPU backend to use.</param>
    public void Execute(IDirectGpuBackend backend)
    {
        ThrowIfDisposed();

        if (backend == null)
        {
            throw new ArgumentNullException(nameof(backend));
        }

        // Execute in topological order
        foreach (var node in _topologicalOrder)
        {
            node.Execute(backend);
        }

        // Final synchronization
        backend.Synchronize();
    }

    /// <summary>
    /// Executes the graph with multi-stream parallelism.
    /// </summary>
    /// <param name="backend">The GPU backend to use.</param>
    /// <param name="streamPool">Pool of streams for parallel execution.</param>
    public void Execute(IDirectGpuBackend backend, GpuStreamPool streamPool)
    {
        ThrowIfDisposed();

        if (backend == null)
        {
            throw new ArgumentNullException(nameof(backend));
        }

        if (streamPool == null)
        {
            throw new ArgumentNullException(nameof(streamPool));
        }

        // Execute level by level
        foreach (var level in _levelNodes.Keys.OrderBy(k => k))
        {
            var levelNodesList = _levelNodes[level];

            // Execute nodes at this level - can be parallel across streams
            foreach (var node in levelNodesList)
            {
                var stream = GetStreamForNode(node, streamPool);
                node.ExecuteAsync(backend, stream);
            }
        }

        // Final synchronization
        streamPool.SynchronizeAll();
    }

    /// <summary>
    /// Executes the graph asynchronously using the async backend.
    /// </summary>
    /// <param name="backend">The async GPU backend to use.</param>
    /// <param name="streamPool">Pool of streams for parallel execution.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task ExecuteAsync(
        IAsyncGpuBackend backend,
        GpuStreamPool streamPool,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (backend == null)
        {
            throw new ArgumentNullException(nameof(backend));
        }

        if (streamPool == null)
        {
            throw new ArgumentNullException(nameof(streamPool));
        }

        // Execute level by level with async yielding for cancellation checks
        foreach (var level in _levelNodes.Keys.OrderBy(k => k))
        {
            cancellationToken.ThrowIfCancellationRequested();

            var levelNodesList = _levelNodes[level];

            // Launch all nodes at this level
            foreach (var node in levelNodesList)
            {
                var stream = GetStreamForNode(node, streamPool);
                node.ExecuteAsync(backend, stream);
            }

            // Yield to allow other work
            await Task.Yield();
        }

        // Final synchronization
        streamPool.SynchronizeAll();
    }

    /// <summary>
    /// Gets nodes that can execute at a specific level.
    /// </summary>
    /// <param name="level">The execution level (0-based).</param>
    /// <returns>Nodes at the specified level.</returns>
    public IReadOnlyList<ExecutionNode> GetNodesAtLevel(int level)
    {
        if (_levelNodes.TryGetValue(level, out var nodes))
        {
            return nodes;
        }
        return Array.Empty<ExecutionNode>();
    }

    /// <summary>
    /// Gets the stream assignment for optimal execution.
    /// </summary>
    public Dictionary<ExecutionNode, GpuStreamType> GetStreamAssignments()
    {
        var assignments = new Dictionary<ExecutionNode, GpuStreamType>();

        foreach (var node in _nodes)
        {
            var streamType = DetermineOptimalStreamType(node);
            assignments[node] = streamType;
        }

        return assignments;
    }

    /// <summary>
    /// Creates a sub-graph containing only the specified nodes and their dependencies.
    /// </summary>
    public ExecutionGraph CreateSubGraph(IEnumerable<ExecutionNode> nodes)
    {
        var nodeSet = new HashSet<ExecutionNode>(nodes);
        var subGraphNodes = new List<ExecutionNode>();

        // Include all dependencies
        var visited = new HashSet<int>();
        void CollectDependencies(ExecutionNode node)
        {
            if (!visited.Add(node.NodeId))
            {
                return;
            }

            foreach (var dep in node.Dependencies)
            {
                CollectDependencies(dep);
            }

            subGraphNodes.Add(node);
        }

        foreach (var node in nodeSet)
        {
            CollectDependencies(node);
        }

        return new ExecutionGraph(subGraphNodes);
    }

    private List<ExecutionNode> ComputeTopologicalOrder()
    {
        var result = new List<ExecutionNode>();
        var visited = new HashSet<int>();
        var inProgress = new HashSet<int>();

        void Visit(ExecutionNode node)
        {
            if (visited.Contains(node.NodeId))
            {
                return;
            }

            if (inProgress.Contains(node.NodeId))
            {
                throw new InvalidOperationException(
                    $"Circular dependency detected at node {node.Name}");
            }

            inProgress.Add(node.NodeId);

            foreach (var dep in node.Dependencies)
            {
                Visit(dep);
            }

            inProgress.Remove(node.NodeId);
            visited.Add(node.NodeId);
            result.Add(node);
        }

        foreach (var node in _nodes)
        {
            Visit(node);
        }

        return result;
    }

    private Dictionary<int, List<ExecutionNode>> ComputeLevels()
    {
        var levels = new Dictionary<int, List<ExecutionNode>>();
        var nodeLevel = new Dictionary<int, int>();

        // Compute level for each node (max level of dependencies + 1)
        foreach (var node in _topologicalOrder)
        {
            int level = 0;
            foreach (var dep in node.Dependencies)
            {
                if (nodeLevel.TryGetValue(dep.NodeId, out var depLevel))
                {
                    level = Math.Max(level, depLevel + 1);
                }
            }

            nodeLevel[node.NodeId] = level;

            if (!levels.TryGetValue(level, out var levelList))
            {
                levelList = new List<ExecutionNode>();
                levels[level] = levelList;
            }
            levelList.Add(node);
        }

        return levels;
    }

    private IGpuStream GetStreamForNode(ExecutionNode node, GpuStreamPool pool)
    {
        // Use assigned stream if available
        if (node.AssignedStream != null)
        {
            return node.AssignedStream;
        }

        // Determine stream type based on node type
        var streamType = DetermineOptimalStreamType(node);

        return streamType switch
        {
            GpuStreamType.HostToDevice => pool.DefaultH2DStream,
            GpuStreamType.DeviceToHost => pool.DefaultD2HStream,
            _ => pool.AcquireStream(GpuStreamType.Compute)
        };
    }

    private static GpuStreamType DetermineOptimalStreamType(ExecutionNode node)
    {
        return node.NodeType switch
        {
            ExecutionNodeType.TransferH2D => GpuStreamType.HostToDevice,
            ExecutionNodeType.TransferD2H => GpuStreamType.DeviceToHost,
            ExecutionNodeType.Copy => GpuStreamType.DeviceToDevice,
            _ => GpuStreamType.Compute
        };
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(ExecutionGraph));
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // Dispose any sync points
        foreach (var node in _nodes)
        {
            node.CompletionSync?.Dispose();
        }
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"ExecutionGraph(Nodes: {_nodes.Count}, Levels: {LevelCount}, " +
               $"MaxParallelism: {MaxParallelism}, Cost: {TotalEstimatedCost})";
    }
}

/// <summary>
/// Statistics about graph execution.
/// </summary>
public sealed class GraphExecutionStats
{
    /// <summary>
    /// Total execution time in milliseconds.
    /// </summary>
    public double TotalTimeMs { get; init; }

    /// <summary>
    /// Time spent in compute operations.
    /// </summary>
    public double ComputeTimeMs { get; init; }

    /// <summary>
    /// Time spent in memory transfers.
    /// </summary>
    public double TransferTimeMs { get; init; }

    /// <summary>
    /// Time spent waiting for synchronization.
    /// </summary>
    public double SyncTimeMs { get; init; }

    /// <summary>
    /// Number of nodes executed.
    /// </summary>
    public int NodesExecuted { get; init; }

    /// <summary>
    /// Number of streams used.
    /// </summary>
    public int StreamsUsed { get; init; }

    /// <summary>
    /// Peak memory usage during execution.
    /// </summary>
    public long PeakMemoryBytes { get; init; }

    /// <summary>
    /// Achieved parallelism (average concurrent nodes).
    /// </summary>
    public double AchievedParallelism { get; init; }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"Execution: {TotalTimeMs:F2}ms (Compute: {ComputeTimeMs:F2}ms, " +
               $"Transfer: {TransferTimeMs:F2}ms, Sync: {SyncTimeMs:F2}ms), " +
               $"Nodes: {NodesExecuted}, Streams: {StreamsUsed}, " +
               $"Parallelism: {AchievedParallelism:F1}";
    }
}
