using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu.Graph;

/// <summary>
/// Represents a buffer allocation node in the execution graph.
/// Used for allocating and deallocating GPU memory.
/// </summary>
public sealed class AllocationNode : ExecutionNode
{
    private readonly int _size;
    private readonly bool _isDeallocation;
    private IGpuBuffer? _allocatedBuffer;
    private readonly IGpuBuffer? _bufferToDeallocate;

    /// <inheritdoc/>
    public override ExecutionNodeType NodeType =>
        _isDeallocation ? ExecutionNodeType.Deallocate : ExecutionNodeType.Allocate;

    /// <summary>
    /// Gets the size of the allocation in elements.
    /// </summary>
    public int Size => _size;

    /// <summary>
    /// Gets the allocated buffer after execution (for allocation nodes).
    /// </summary>
    public IGpuBuffer? AllocatedBuffer => _allocatedBuffer;

    /// <summary>
    /// Gets whether this is a deallocation node.
    /// </summary>
    public bool IsDeallocation => _isDeallocation;

    /// <summary>
    /// Gets the tensor role for the allocated buffer.
    /// </summary>
    public GpuTensorRole Role { get; }

    /// <inheritdoc/>
    public override int EstimatedCost => _isDeallocation ? 1 : (_size / 4096) + 2;

    /// <inheritdoc/>
    public override string Name => _isDeallocation
        ? $"Deallocate_{NodeId}"
        : $"Allocate_{_size}_{Role}_{NodeId}";

    /// <summary>
    /// Creates an allocation node.
    /// </summary>
    /// <param name="size">Number of elements to allocate.</param>
    /// <param name="role">The role of the allocated buffer.</param>
    public static AllocationNode CreateAllocate(int size, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        return new AllocationNode(size, role);
    }

    /// <summary>
    /// Creates a deallocation node.
    /// </summary>
    /// <param name="buffer">The buffer to deallocate.</param>
    public static AllocationNode CreateDeallocate(IGpuBuffer buffer)
    {
        return new AllocationNode(buffer);
    }

    private AllocationNode(int size, GpuTensorRole role)
    {
        _size = size;
        Role = role;
        _isDeallocation = false;
    }

    private AllocationNode(IGpuBuffer buffer)
    {
        _bufferToDeallocate = buffer ?? throw new ArgumentNullException(nameof(buffer));
        _size = buffer.Size;
        _isDeallocation = true;
    }

    /// <inheritdoc/>
    public override void Execute(IDirectGpuBackend backend)
    {
        if (_isDeallocation)
        {
            _bufferToDeallocate?.Dispose();
        }
        else
        {
            _allocatedBuffer = backend.AllocateBuffer(_size);
        }
    }

    /// <inheritdoc/>
    public override void ExecuteAsync(IDirectGpuBackend backend, IGpuStream stream)
    {
        // Allocations are typically synchronous, but we still track stream assignment
        AssignedStream = stream;
        Execute(backend);
    }

    /// <inheritdoc/>
    public override ExecutionNode Clone()
    {
        if (_isDeallocation && _bufferToDeallocate != null)
        {
            return CreateDeallocate(_bufferToDeallocate);
        }
        return CreateAllocate(_size, Role);
    }
}

/// <summary>
/// Represents a barrier/synchronization node in the execution graph.
/// Used to synchronize multiple streams or create synchronization points.
/// </summary>
public sealed class BarrierNode : ExecutionNode
{
    private readonly List<IGpuStream> _streamsToSync;

    /// <inheritdoc/>
    public override ExecutionNodeType NodeType => ExecutionNodeType.Barrier;

    /// <summary>
    /// Gets the streams that this barrier synchronizes.
    /// </summary>
    public IReadOnlyList<IGpuStream> StreamsToSynchronize => _streamsToSync;

    /// <summary>
    /// Gets whether this is a full device synchronization.
    /// </summary>
    public bool IsFullSync { get; }

    /// <inheritdoc/>
    public override int EstimatedCost => IsFullSync ? 50 : _streamsToSync.Count * 5;

    /// <inheritdoc/>
    public override string Name => IsFullSync
        ? $"FullSync_{NodeId}"
        : $"Barrier_{_streamsToSync.Count}_{NodeId}";

    /// <summary>
    /// Creates a barrier that synchronizes specific streams.
    /// </summary>
    /// <param name="streams">Streams to synchronize.</param>
    public BarrierNode(IEnumerable<IGpuStream> streams)
    {
        _streamsToSync = streams?.ToList() ?? throw new ArgumentNullException(nameof(streams));
        IsFullSync = false;
    }

    /// <summary>
    /// Creates a full device synchronization barrier.
    /// </summary>
    public BarrierNode()
    {
        _streamsToSync = new List<IGpuStream>();
        IsFullSync = true;
    }

    /// <inheritdoc/>
    public override void Execute(IDirectGpuBackend backend)
    {
        if (IsFullSync)
        {
            backend.Synchronize();
        }
        else
        {
            foreach (var stream in _streamsToSync)
            {
                stream.Synchronize();
            }
        }
    }

    /// <inheritdoc/>
    public override void ExecuteAsync(IDirectGpuBackend backend, IGpuStream stream)
    {
        AssignedStream = stream;

        if (IsFullSync)
        {
            backend.Synchronize();
        }
        else
        {
            // Record events on all streams and make target stream wait
            foreach (var srcStream in _streamsToSync)
            {
                if (srcStream != stream)
                {
                    var evt = srcStream.RecordEvent();
                    stream.WaitEvent(evt);
                }
            }
        }
    }

    /// <inheritdoc/>
    public override ExecutionNode Clone()
    {
        if (IsFullSync)
        {
            return new BarrierNode();
        }
        return new BarrierNode(_streamsToSync);
    }
}

/// <summary>
/// Represents an event recording node in the execution graph.
/// </summary>
public sealed class EventNode : ExecutionNode
{
    private IGpuEvent? _recordedEvent;
    private readonly IGpuEvent? _eventToWait;
    private readonly bool _isWait;

    /// <inheritdoc/>
    public override ExecutionNodeType NodeType =>
        _isWait ? ExecutionNodeType.WaitEvent : ExecutionNodeType.RecordEvent;

    /// <summary>
    /// Gets the recorded event (for record nodes).
    /// </summary>
    public IGpuEvent? RecordedEvent => _recordedEvent;

    /// <summary>
    /// Gets the event to wait for (for wait nodes).
    /// </summary>
    public IGpuEvent? EventToWait => _eventToWait;

    /// <summary>
    /// Gets whether this is a wait node.
    /// </summary>
    public bool IsWait => _isWait;

    /// <inheritdoc/>
    public override int EstimatedCost => _isWait ? 2 : 1;

    /// <inheritdoc/>
    public override string Name => _isWait
        ? $"WaitEvent_{NodeId}"
        : $"RecordEvent_{NodeId}";

    /// <summary>
    /// Creates an event recording node.
    /// </summary>
    public static EventNode CreateRecord()
    {
        return new EventNode(isWait: false);
    }

    /// <summary>
    /// Creates an event wait node.
    /// </summary>
    /// <param name="eventToWait">The event to wait for.</param>
    public static EventNode CreateWait(IGpuEvent eventToWait)
    {
        return new EventNode(eventToWait);
    }

    private EventNode(bool isWait)
    {
        _isWait = isWait;
    }

    private EventNode(IGpuEvent eventToWait)
    {
        _eventToWait = eventToWait ?? throw new ArgumentNullException(nameof(eventToWait));
        _isWait = true;
    }

    /// <inheritdoc/>
    public override void Execute(IDirectGpuBackend backend)
    {
        if (_isWait && _eventToWait != null)
        {
            _eventToWait.Synchronize();
        }
        // For record nodes, we need a stream
    }

    /// <inheritdoc/>
    public override void ExecuteAsync(IDirectGpuBackend backend, IGpuStream stream)
    {
        AssignedStream = stream;

        if (_isWait && _eventToWait != null)
        {
            stream.WaitEvent(_eventToWait);
        }
        else if (!_isWait)
        {
            _recordedEvent = stream.RecordEvent();
        }
    }

    /// <inheritdoc/>
    public override ExecutionNode Clone()
    {
        if (_isWait && _eventToWait != null)
        {
            return CreateWait(_eventToWait);
        }
        return CreateRecord();
    }
}
