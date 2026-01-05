using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu.Graph;

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
                    using var evt = srcStream.RecordEvent();
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
