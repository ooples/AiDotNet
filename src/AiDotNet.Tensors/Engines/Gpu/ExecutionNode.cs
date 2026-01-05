using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Abstract base class for nodes in an execution graph.
/// Each node represents an operation that can be executed on the GPU.
/// </summary>
public abstract class ExecutionNode
{
    private static int _nextNodeId;
    private readonly List<ExecutionNode> _dependencies = new();
    private readonly List<ExecutionNode> _dependents = new();

    /// <summary>
    /// Gets the unique identifier for this node within the graph.
    /// </summary>
    public int NodeId { get; }

    /// <summary>
    /// Gets the type of this execution node.
    /// </summary>
    public abstract ExecutionNodeType NodeType { get; }

    /// <summary>
    /// Gets the stream assigned to this node for execution.
    /// Set by the stream assignment optimization pass.
    /// </summary>
    public IGpuStream? AssignedStream { get; set; }

    /// <summary>
    /// Gets the sync point created after this node's execution.
    /// Used for dependency management between streams.
    /// </summary>
    public GpuSyncPoint? CompletionSync { get; protected set; }

    /// <summary>
    /// Gets the nodes that must complete before this node can execute.
    /// </summary>
    public IReadOnlyList<ExecutionNode> Dependencies => _dependencies;

    /// <summary>
    /// Gets the nodes that depend on this node's completion.
    /// </summary>
    public IReadOnlyList<ExecutionNode> Dependents => _dependents;

    /// <summary>
    /// Gets the estimated cost of this operation for scheduling decisions.
    /// Higher values indicate more expensive operations.
    /// </summary>
    public virtual int EstimatedCost => 1;

    /// <summary>
    /// Gets whether this node can be fused with other nodes.
    /// </summary>
    public virtual bool CanFuse => false;

    /// <summary>
    /// Gets the input tensors for this node.
    /// </summary>
    public virtual IReadOnlyList<IGpuTensor> InputTensors => Array.Empty<IGpuTensor>();

    /// <summary>
    /// Gets the output tensors produced by this node.
    /// </summary>
    public virtual IReadOnlyList<IGpuTensor> OutputTensors => Array.Empty<IGpuTensor>();

    /// <summary>
    /// Gets a descriptive name for this node (for debugging/profiling).
    /// </summary>
    public virtual string Name => $"{NodeType}_{NodeId}";

    protected ExecutionNode()
    {
        NodeId = Interlocked.Increment(ref _nextNodeId);
    }

    /// <summary>
    /// Adds a dependency on another node.
    /// This node will not execute until the dependency completes.
    /// </summary>
    /// <param name="dependency">The node this node depends on.</param>
    public void AddDependency(ExecutionNode dependency)
    {
        if (dependency == null)
        {
            throw new ArgumentNullException(nameof(dependency));
        }

        if (dependency == this)
        {
            throw new ArgumentException("A node cannot depend on itself.", nameof(dependency));
        }

        if (!_dependencies.Contains(dependency))
        {
            _dependencies.Add(dependency);
            dependency._dependents.Add(this);
        }
    }

    /// <summary>
    /// Removes a dependency on another node.
    /// </summary>
    /// <param name="dependency">The node to remove from dependencies.</param>
    public void RemoveDependency(ExecutionNode dependency)
    {
        if (_dependencies.Remove(dependency))
        {
            dependency._dependents.Remove(this);
        }
    }

    /// <summary>
    /// Executes this node on the GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    public abstract void Execute(IDirectGpuBackend backend);

    /// <summary>
    /// Executes this node asynchronously on the specified stream.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="stream">The stream to execute on.</param>
    public virtual void ExecuteAsync(IDirectGpuBackend backend, IGpuStream stream)
    {
        // Default implementation: ensure dependencies are waited on, then execute
        foreach (var dep in _dependencies)
        {
            if (dep.CompletionSync != null && dep.AssignedStream != stream)
            {
                dep.CompletionSync.MakeStreamWait(stream);
            }
        }

        AssignedStream = stream;
        Execute(backend);

        // Record completion for dependents on other streams
        if (_dependents.Any(d => d.AssignedStream != stream))
        {
            var evt = stream.RecordEvent();
            CompletionSync = new EventSyncPoint(evt, stream);
        }
    }

    /// <summary>
    /// Validates that this node is properly configured for execution.
    /// </summary>
    /// <returns>True if valid, false otherwise.</returns>
    public virtual bool Validate()
    {
        var visiting = new HashSet<int>();
        var visited = new HashSet<int>();
        return !HasCircularDependency(this, visiting, visited);
    }

    private static bool HasCircularDependency(
        ExecutionNode node,
        HashSet<int> visiting,
        HashSet<int> visited)
    {
        if (visited.Contains(node.NodeId))
        {
            return false;
        }

        if (!visiting.Add(node.NodeId))
        {
            return true;
        }

        foreach (var dep in node._dependencies)
        {
            if (HasCircularDependency(dep, visiting, visited))
            {
                return true;
            }
        }

        visiting.Remove(node.NodeId);
        visited.Add(node.NodeId);
        return false;
    }

    /// <summary>
    /// Creates a clone of this node for graph optimization.
    /// </summary>
    /// <returns>A new node with the same configuration but no dependencies.</returns>
    public abstract ExecutionNode Clone();

    public override string ToString()
    {
        return $"{Name} (deps: {_dependencies.Count}, dependents: {_dependents.Count})";
    }

    /// <summary>
    /// Simple sync point implementation wrapping an event.
    /// </summary>
    private sealed class EventSyncPoint : GpuSyncPoint
    {
        private readonly IGpuEvent _event;
        private readonly IGpuStream _stream;

        public EventSyncPoint(IGpuEvent evt, IGpuStream stream)
        {
            _event = evt;
            _stream = stream;
        }

        public override bool IsComplete => _event.IsComplete;
        public override IGpuStream? Stream => _stream;
        public override IGpuEvent? Event => _event;

        public override void Wait()
        {
            _event.Synchronize();
        }

        public override bool Poll()
        {
            return _event.Query();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _event.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
