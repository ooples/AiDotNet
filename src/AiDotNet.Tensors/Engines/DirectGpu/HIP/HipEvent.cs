using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// HIP implementation of a GPU event.
/// Wraps a hipEvent_t handle for synchronization between streams.
/// </summary>
public sealed class HipEvent : IGpuEvent
{
    private readonly HipBackend _backend;
    private IntPtr _handle;
    private bool _disposed;
    private bool _isRecorded;
    private readonly bool _enableTiming;

    /// <summary>
    /// HIP event flags.
    /// </summary>
    [Flags]
    public enum HipEventFlags : uint
    {
        /// <summary>Default event behavior.</summary>
        Default = 0,

        /// <summary>Event uses blocking synchronization.</summary>
        BlockingSync = 1,

        /// <summary>Event will not record timing data.</summary>
        DisableTiming = 2,

        /// <summary>Event is suitable for interprocess use.</summary>
        Interprocess = 4
    }

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
    /// Creates a new HIP event and records it on the specified stream.
    /// </summary>
    /// <param name="backend">The HIP backend.</param>
    /// <param name="stream">The stream to record the event on.</param>
    public HipEvent(HipBackend backend, IGpuStream stream)
        : this(backend, stream, true)
    {
    }

    /// <summary>
    /// Creates a new HIP event with optional timing capability.
    /// </summary>
    /// <param name="backend">The HIP backend.</param>
    /// <param name="stream">The stream to record the event on (can be null for unrecorded event).</param>
    /// <param name="enableTiming">Whether to enable timing data collection.</param>
    public HipEvent(HipBackend backend, IGpuStream? stream, bool enableTiming)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _enableTiming = enableTiming;

        IntPtr eventHandle = IntPtr.Zero;
        HipError result;

        if (enableTiming)
        {
            result = HipNativeBindings.hipEventCreate(ref eventHandle);
        }
        else
        {
            // Disable timing for better performance when not needed
            result = HipNativeBindings.hipEventCreateWithFlags(
                ref eventHandle,
                (uint)HipEventFlags.DisableTiming);
        }

        HipNativeBindings.CheckError(result, "hipEventCreate");
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

        if (stream is not HipStream)
        {
            throw new ArgumentException("Stream must be a HipStream", nameof(stream));
        }

        var result = HipNativeBindings.hipEventRecord(_handle, stream.Handle);
        HipNativeBindings.CheckError(result, "hipEventRecord");
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

        var result = HipNativeBindings.hipEventSynchronize(_handle);
        HipNativeBindings.CheckError(result, "hipEventSynchronize");
    }

    /// <inheritdoc/>
    public bool Query()
    {
        ThrowIfDisposed();

        if (!_isRecorded)
        {
            return false;
        }

        var result = HipNativeBindings.hipEventQuery(_handle);

        if (result == HipError.Success)
        {
            return true;
        }

        if (result == HipError.ErrorNotReady)
        {
            return false;
        }

        HipNativeBindings.CheckError(result, "hipEventQuery");
        return false;
    }

    /// <inheritdoc/>
    public float GetElapsedTime(IGpuEvent startEvent)
    {
        ThrowIfDisposed();

        if (startEvent is not HipEvent hipStartEvent)
        {
            throw new ArgumentException("Event must be a HipEvent", nameof(startEvent));
        }

        if (!_enableTiming)
        {
            throw new InvalidOperationException("Event was created without timing capability");
        }

        if (!_isRecorded || !hipStartEvent.IsRecorded)
        {
            throw new InvalidOperationException("Both events must be recorded");
        }

        // Ensure both events are complete
        Synchronize();
        hipStartEvent.Synchronize();

        float elapsedMs = 0f;
        var result = HipNativeBindings.hipEventElapsedTime(
            ref elapsedMs,
            hipStartEvent.Handle,
            _handle);

        HipNativeBindings.CheckError(result, "hipEventElapsedTime");
        return elapsedMs;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(HipEvent));
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
                HipNativeBindings.hipEventDestroy(_handle);
            }
            catch
            {
                // Ignore destruction errors during disposal
            }
        }

        _handle = IntPtr.Zero;
    }
}
