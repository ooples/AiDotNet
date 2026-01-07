using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// HIP implementation of a GPU stream.
/// Wraps a hipStream_t handle for concurrent execution and compute/transfer overlap.
/// </summary>
public sealed class HipStream : IGpuStream
{
    private readonly HipBackend _backend;
    private IntPtr _handle;
    private bool _disposed;
    private readonly bool _ownsHandle;

    /// <inheritdoc/>
    public IntPtr Handle => _handle;

    /// <inheritdoc/>
    public GpuStreamType StreamType { get; }

    /// <inheritdoc/>
    public bool IsDefault { get; }

    /// <inheritdoc/>
    public int Priority { get; }

    /// <summary>
    /// Creates a new HIP stream.
    /// </summary>
    /// <param name="backend">The HIP backend.</param>
    /// <param name="streamType">The type of stream.</param>
    /// <param name="priority">Stream priority (lower = higher priority).</param>
    public HipStream(HipBackend backend, GpuStreamType streamType, int priority = 0)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        StreamType = streamType;
        Priority = priority;
        IsDefault = false;
        _ownsHandle = true;

        IntPtr stream = IntPtr.Zero;
        HipError result;

        if (priority != 0)
        {
            // Create stream with priority using hipStreamCreateWithPriority
            result = HipNativeBindings.hipStreamCreateWithPriority(ref stream, 0, priority);
        }
        else
        {
            result = HipNativeBindings.hipStreamCreate(ref stream);
        }

        HipNativeBindings.CheckError(result, "hipStreamCreate");
        _handle = stream;
    }

    /// <summary>
    /// Creates a HIP stream wrapper for an existing handle (e.g., the default stream).
    /// </summary>
    /// <param name="backend">The HIP backend.</param>
    /// <param name="handle">The existing stream handle.</param>
    /// <param name="streamType">The type of stream.</param>
    /// <param name="isDefault">Whether this is the default stream.</param>
    internal HipStream(HipBackend backend, IntPtr handle, GpuStreamType streamType, bool isDefault)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _handle = handle;
        StreamType = streamType;
        IsDefault = isDefault;
        Priority = 0;
        _ownsHandle = false; // Don't dispose external handles
    }

    /// <inheritdoc/>
    public void Synchronize()
    {
        ThrowIfDisposed();
        var result = HipNativeBindings.hipStreamSynchronize(_handle);
        HipNativeBindings.CheckError(result, "hipStreamSynchronize");
    }

    /// <inheritdoc/>
    public IGpuEvent RecordEvent()
    {
        ThrowIfDisposed();
        return new HipEvent(_backend, this);
    }

    /// <inheritdoc/>
    public void WaitEvent(IGpuEvent gpuEvent)
    {
        ThrowIfDisposed();

        if (gpuEvent is not HipEvent hipEvent)
        {
            throw new ArgumentException("Event must be a HipEvent", nameof(gpuEvent));
        }

        var result = HipNativeBindings.hipStreamWaitEvent(_handle, hipEvent.Handle, 0);
        HipNativeBindings.CheckError(result, "hipStreamWaitEvent");
    }

    /// <inheritdoc/>
    public bool Query()
    {
        ThrowIfDisposed();
        var result = HipNativeBindings.hipStreamQuery(_handle);

        if (result == HipError.Success)
        {
            return true;
        }

        if (result == HipError.ErrorNotReady)
        {
            return false;
        }

        HipNativeBindings.CheckError(result, "hipStreamQuery");
        return false;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(HipStream));
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

        if (_ownsHandle && _handle != IntPtr.Zero)
        {
            try
            {
                HipNativeBindings.hipStreamDestroy(_handle);
            }
            catch
            {
                // Ignore destruction errors during disposal
            }
        }

        _handle = IntPtr.Zero;
    }
}
