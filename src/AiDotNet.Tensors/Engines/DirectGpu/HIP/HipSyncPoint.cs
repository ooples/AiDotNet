using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// HIP implementation of a GPU sync point.
/// Uses HIP events for deferred synchronization.
/// </summary>
public sealed class HipSyncPoint : GpuSyncPoint
{
    private readonly HipEvent _event;
    private readonly HipStream? _stream;
    private bool _disposed;

    /// <inheritdoc/>
    public override bool IsComplete => !_disposed && _event.IsComplete;

    /// <inheritdoc/>
    public override IGpuStream? Stream => _stream;

    /// <inheritdoc/>
    public override IGpuEvent? Event => _event;

    /// <summary>
    /// Creates a new HIP sync point by recording an event on the specified stream.
    /// </summary>
    /// <param name="backend">The HIP backend.</param>
    /// <param name="stream">The stream to create the sync point on.</param>
    /// <param name="enableTiming">Whether to enable timing data collection.</param>
    public HipSyncPoint(HipBackend backend, HipStream stream, bool enableTiming = false)
    {
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
        _event = new HipEvent(backend, stream, enableTiming);
    }

    /// <summary>
    /// Creates a new HIP sync point from an existing event.
    /// </summary>
    /// <param name="gpuEvent">The existing HIP event.</param>
    /// <param name="stream">The stream the event was recorded on.</param>
    internal HipSyncPoint(HipEvent gpuEvent, HipStream? stream)
    {
        _event = gpuEvent ?? throw new ArgumentNullException(nameof(gpuEvent));
        _stream = stream;
    }

    /// <inheritdoc/>
    public override void Wait()
    {
        ThrowIfDisposed();
        _event.Synchronize();
    }

    /// <inheritdoc/>
    public override bool Poll()
    {
        ThrowIfDisposed();
        return _event.Query();
    }

    /// <inheritdoc/>
    public override void MakeStreamWait(IGpuStream stream)
    {
        ThrowIfDisposed();

        if (stream is not HipStream hipStream)
        {
            throw new ArgumentException("Stream must be a HipStream", nameof(stream));
        }

        hipStream.WaitEvent(_event);
    }

    /// <inheritdoc/>
    public override float GetElapsedTime(GpuSyncPoint startPoint)
    {
        ThrowIfDisposed();

        if (startPoint is not HipSyncPoint hipStartPoint)
        {
            throw new ArgumentException("Start point must be a HipSyncPoint", nameof(startPoint));
        }

        return _event.GetElapsedTime(hipStartPoint._event);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(HipSyncPoint));
        }
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _event.Dispose();
            }
            _disposed = true;
        }

        base.Dispose(disposing);
    }
}
