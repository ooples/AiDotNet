namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Abstract base class for deferred GPU synchronization.
/// A sync point represents a point in the GPU execution timeline that can be waited on.
/// Concrete implementations use backend-specific events (CUDA events, OpenCL events, HIP events).
/// </summary>
public abstract class GpuSyncPoint : IDisposable
{
    private bool _disposed;

    /// <summary>
    /// Gets whether the GPU operations up to this sync point have completed.
    /// </summary>
    public abstract bool IsComplete { get; }

    /// <summary>
    /// Gets the stream on which this sync point was created.
    /// </summary>
    public abstract IGpuStream? Stream { get; }

    /// <summary>
    /// Gets the underlying event handle for this sync point.
    /// </summary>
    public abstract IGpuEvent? Event { get; }

    /// <summary>
    /// Blocks the calling CPU thread until all GPU operations up to this sync point complete.
    /// </summary>
    public abstract void Wait();

    /// <summary>
    /// Non-blocking check whether the sync point has been reached.
    /// </summary>
    /// <returns>True if complete, false if still pending.</returns>
    public abstract bool Poll();

    /// <summary>
    /// Makes the specified stream wait for this sync point.
    /// Operations enqueued on the stream after this call will not start until the sync point is reached.
    /// </summary>
    /// <param name="stream">The stream that should wait.</param>
    public virtual void MakeStreamWait(IGpuStream stream)
    {
        if (Event != null)
        {
            stream.WaitEvent(Event);
        }
    }

    /// <summary>
    /// Gets the elapsed time since a previous sync point.
    /// Both sync points must be complete.
    /// </summary>
    /// <param name="startPoint">The starting sync point.</param>
    /// <returns>Elapsed time in milliseconds.</returns>
    public virtual float GetElapsedTime(GpuSyncPoint startPoint)
    {
        if (Event == null || startPoint.Event == null)
        {
            return 0f;
        }

        return Event.GetElapsedTime(startPoint.Event);
    }

    /// <summary>
    /// Creates a sync point that is already complete (no waiting required).
    /// </summary>
    /// <returns>An immediately complete sync point.</returns>
    public static GpuSyncPoint CreateComplete()
    {
        return new CompleteSyncPoint();
    }

    /// <summary>
    /// Creates a sync point that combines multiple sync points.
    /// The combined point is complete when all constituent points are complete.
    /// </summary>
    /// <param name="syncPoints">The sync points to combine.</param>
    /// <returns>A combined sync point.</returns>
    public static GpuSyncPoint CreateCombined(params GpuSyncPoint[] syncPoints)
    {
        if (syncPoints == null || syncPoints.Length == 0)
        {
            return CreateComplete();
        }

        if (syncPoints.Length == 1)
        {
            return syncPoints[0];
        }

        return new CombinedSyncPoint(syncPoints);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                // Dispose managed resources
            }
            _disposed = true;
        }
    }

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// A sync point that is immediately complete.
    /// </summary>
    private sealed class CompleteSyncPoint : GpuSyncPoint
    {
        public override bool IsComplete => true;
        public override IGpuStream? Stream => null;
        public override IGpuEvent? Event => null;

        public override void Wait()
        {
            // Already complete, nothing to wait for
        }

        public override bool Poll() => true;
    }

    /// <summary>
    /// A sync point that combines multiple sync points.
    /// </summary>
    private sealed class CombinedSyncPoint : GpuSyncPoint
    {
        private readonly GpuSyncPoint[] _syncPoints;

        public CombinedSyncPoint(GpuSyncPoint[] syncPoints)
        {
            _syncPoints = syncPoints;
        }

        public override bool IsComplete
        {
            get
            {
                foreach (var sp in _syncPoints)
                {
                    if (!sp.IsComplete)
                    {
                        return false;
                    }
                }
                return true;
            }
        }

        public override IGpuStream? Stream => null; // Multiple streams

        public override IGpuEvent? Event => null; // Multiple events

        public override void Wait()
        {
            foreach (var sp in _syncPoints)
            {
                sp.Wait();
            }
        }

        public override bool Poll()
        {
            foreach (var sp in _syncPoints)
            {
                if (!sp.Poll())
                {
                    return false;
                }
            }
            return true;
        }

        public override void MakeStreamWait(IGpuStream stream)
        {
            foreach (var sp in _syncPoints)
            {
                sp.MakeStreamWait(stream);
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var sp in _syncPoints)
                {
                    sp.Dispose();
                }
            }
            base.Dispose(disposing);
        }
    }
}
