using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu.Graph;

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
