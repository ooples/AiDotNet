using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// CUDA implementation of a GPU event.
/// Wraps a CUevent handle for synchronization between streams.
/// </summary>
public sealed class CudaEvent : IGpuEvent
{
    private readonly CudaBackend _backend;
    private IntPtr _handle;
    private bool _disposed;
    private bool _isRecorded;
    private readonly bool _enableTiming;

    /// <inheritdoc/>
    public IntPtr Handle => _handle;

    /// <inheritdoc/>
    public bool IsRecorded => _isRecorded;

    /// <inheritdoc/>
    public bool IsComplete
    {
        get
        {
            if (_disposed || !_isRecorded)
            {
                return false;
            }

            return Query();
        }
    }

    /// <summary>
    /// Creates a new CUDA event and records it on the specified stream.
    /// </summary>
    /// <param name="backend">The CUDA backend.</param>
    /// <param name="stream">The stream to record the event on.</param>
    public CudaEvent(CudaBackend backend, IGpuStream stream)
        : this(backend, stream, true)
    {
    }

    /// <summary>
    /// Creates a new CUDA event with optional timing capability.
    /// </summary>
    /// <param name="backend">The CUDA backend.</param>
    /// <param name="stream">The stream to record the event on (can be null for unrecorded event).</param>
    /// <param name="enableTiming">Whether to enable timing data collection.</param>
    public CudaEvent(CudaBackend backend, IGpuStream? stream, bool enableTiming)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _enableTiming = enableTiming;

        IntPtr eventHandle;
        CudaResult result;

        if (enableTiming)
        {
            result = CudaNativeBindings.cuEventCreate(out eventHandle, CudaNativeBindings.CU_EVENT_DEFAULT);
        }
        else
        {
            // Disable timing for better performance when not needed
            result = CudaNativeBindings.cuEventCreate(out eventHandle, CudaNativeBindings.CU_EVENT_DISABLE_TIMING);
        }

        CuBlasNative.CheckCudaResult(result, "cuEventCreate");
        _handle = eventHandle;

        // Record on the stream if provided
        if (stream != null)
        {
            Record(stream);
        }
    }

    /// <summary>
    /// Records this event on a stream.
    /// </summary>
    /// <param name="stream">The stream to record on.</param>
    public void Record(IGpuStream stream)
    {
        ThrowIfDisposed();

        if (stream is not CudaStream)
        {
            throw new ArgumentException("Stream must be a CudaStream", nameof(stream));
        }

        var result = CudaNativeBindings.cuEventRecord(_handle, stream.Handle);
        CuBlasNative.CheckCudaResult(result, "cuEventRecord");
        _isRecorded = true;
    }

    /// <inheritdoc/>
    public void Synchronize()
    {
        ThrowIfDisposed();

        if (!_isRecorded)
        {
            return; // Nothing to synchronize
        }

        var result = CudaNativeBindings.cuEventSynchronize(_handle);
        CuBlasNative.CheckCudaResult(result, "cuEventSynchronize");
    }

    /// <inheritdoc/>
    public bool Query()
    {
        ThrowIfDisposed();

        if (!_isRecorded)
        {
            return false;
        }

        var result = CudaNativeBindings.cuEventQuery(_handle);

        if (result == CudaResult.Success)
        {
            return true;
        }

        if (result == CudaResult.NotReady)
        {
            return false;
        }

        CuBlasNative.CheckCudaResult(result, "cuEventQuery");
        return false;
    }

    /// <inheritdoc/>
    public float GetElapsedTime(IGpuEvent startEvent)
    {
        ThrowIfDisposed();

        if (startEvent is not CudaEvent cudaStartEvent)
        {
            throw new ArgumentException("Event must be a CudaEvent", nameof(startEvent));
        }

        if (!_enableTiming)
        {
            throw new InvalidOperationException("Event was created without timing capability");
        }

        if (!_isRecorded || !cudaStartEvent.IsRecorded)
        {
            throw new InvalidOperationException("Both events must be recorded");
        }

        // Ensure both events are complete
        Synchronize();
        cudaStartEvent.Synchronize();

        float elapsedMs;
        var result = CudaNativeBindings.cuEventElapsedTime(
            out elapsedMs,
            cudaStartEvent.Handle,
            _handle);

        CuBlasNative.CheckCudaResult(result, "cuEventElapsedTime");
        return elapsedMs;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CudaEvent));
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

        if (_handle != IntPtr.Zero)
        {
            try
            {
                CudaNativeBindings.cuEventDestroy(_handle);
            }
            catch
            {
                // Ignore destruction errors during disposal
            }
        }

        _handle = IntPtr.Zero;
    }
}
