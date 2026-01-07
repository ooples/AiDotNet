using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// CUDA implementation of a GPU sync point.
/// Uses CUDA events for deferred synchronization.
/// </summary>
public sealed class CudaSyncPoint : GpuSyncPoint
{
    private readonly CudaEvent _event;
    private readonly CudaStream? _stream;
    private bool _disposed;

    /// <inheritdoc/>
    public override bool IsComplete => !_disposed && _event.IsComplete;

    /// <inheritdoc/>
    public override IGpuStream? Stream => _stream;

    /// <inheritdoc/>
    public override IGpuEvent? Event => _event;

    /// <summary>
    /// Creates a new CUDA sync point by recording an event on the specified stream.
    /// </summary>
    /// <param name="backend">The CUDA backend.</param>
    /// <param name="stream">The stream to create the sync point on.</param>
    /// <param name="enableTiming">Whether to enable timing data collection.</param>
    public CudaSyncPoint(CudaBackend backend, CudaStream stream, bool enableTiming = false)
    {
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
        _event = new CudaEvent(backend, stream, enableTiming);
    }

    /// <summary>
    /// Creates a new CUDA sync point from an existing event.
    /// </summary>
    /// <param name="gpuEvent">The existing CUDA event.</param>
    /// <param name="stream">The stream the event was recorded on.</param>
    internal CudaSyncPoint(CudaEvent gpuEvent, CudaStream? stream)
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

        if (stream is not CudaStream cudaStream)
        {
            throw new ArgumentException("Stream must be a CudaStream", nameof(stream));
        }

        cudaStream.WaitEvent(_event);
    }

    /// <inheritdoc/>
    public override float GetElapsedTime(GpuSyncPoint startPoint)
    {
        ThrowIfDisposed();

        if (startPoint is not CudaSyncPoint cudaStartPoint)
        {
            throw new ArgumentException("Start point must be a CudaSyncPoint", nameof(startPoint));
        }

        return _event.GetElapsedTime(cudaStartPoint._event);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CudaSyncPoint));
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
