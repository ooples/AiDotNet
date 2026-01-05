using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// CUDA implementation of a GPU stream.
/// Wraps a CUstream handle for concurrent kernel execution.
/// </summary>
public sealed class CudaStream : IGpuStream
{
    private readonly CudaBackend _backend;
    private IntPtr _handle;
    private bool _disposed;
    private readonly bool _ownsHandle;

    /// <inheritdoc/>
    public IntPtr Handle => _handle;

    /// <inheritdoc/>
    public GpuStreamType StreamType { get; }

    /// <summary>
    /// Gets whether this is the default stream (null stream).
    /// </summary>
    public bool IsDefault { get; }

    /// <summary>
    /// Gets the stream priority.
    /// Lower values indicate higher priority.
    /// </summary>
    public int Priority { get; }

    /// <summary>
    /// Creates the default CUDA stream (null stream).
    /// </summary>
    /// <param name="backend">The CUDA backend.</param>
    internal CudaStream(CudaBackend backend)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _handle = IntPtr.Zero;
        IsDefault = true;
        StreamType = GpuStreamType.Default;
        Priority = 0;
        _ownsHandle = false;
    }

    /// <summary>
    /// Creates a new CUDA stream with the specified type and priority.
    /// </summary>
    /// <param name="backend">The CUDA backend.</param>
    /// <param name="streamType">The type of stream to create.</param>
    /// <param name="priority">The stream priority (lower is higher priority).</param>
    public CudaStream(CudaBackend backend, GpuStreamType streamType, int priority = 0)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        StreamType = streamType;
        Priority = priority;
        IsDefault = false;
        _ownsHandle = true;

        IntPtr streamHandle;
        CudaResult result;

        if (priority != 0)
        {
            result = CudaNativeBindings.cuStreamCreateWithPriority(
                out streamHandle,
                CudaNativeBindings.CU_STREAM_NON_BLOCKING,
                priority);
        }
        else
        {
            result = CudaNativeBindings.cuStreamCreate(
                out streamHandle,
                CudaNativeBindings.CU_STREAM_NON_BLOCKING);
        }

        CuBlasNative.CheckCudaResult(result, "cuStreamCreate");
        _handle = streamHandle;
    }

    /// <summary>
    /// Wraps an existing CUDA stream handle.
    /// </summary>
    /// <param name="backend">The CUDA backend.</param>
    /// <param name="handle">The existing stream handle.</param>
    /// <param name="streamType">The type of stream.</param>
    /// <param name="ownsHandle">Whether this wrapper owns the handle and should destroy it.</param>
    internal CudaStream(CudaBackend backend, IntPtr handle, GpuStreamType streamType, bool ownsHandle)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _handle = handle;
        StreamType = streamType;
        IsDefault = handle == IntPtr.Zero;
        Priority = 0;
        _ownsHandle = ownsHandle;
    }

    /// <inheritdoc/>
    public void Synchronize()
    {
        ThrowIfDisposed();

        var result = CudaNativeBindings.cuStreamSynchronize(_handle);
        CuBlasNative.CheckCudaResult(result, "cuStreamSynchronize");
    }

    /// <inheritdoc/>
    public IGpuEvent RecordEvent()
    {
        ThrowIfDisposed();
        return new CudaEvent(_backend, this);
    }

    /// <inheritdoc/>
    public void WaitEvent(IGpuEvent gpuEvent)
    {
        ThrowIfDisposed();

        if (gpuEvent is not CudaEvent cudaEvent)
        {
            throw new ArgumentException("Event must be a CudaEvent", nameof(gpuEvent));
        }

        var result = CudaNativeBindings.cuStreamWaitEvent(_handle, cudaEvent.Handle, 0);
        CuBlasNative.CheckCudaResult(result, "cuStreamWaitEvent");
    }

    /// <summary>
    /// Queries whether all operations submitted to the stream have completed.
    /// </summary>
    /// <returns>True if all operations are complete, false otherwise.</returns>
    public bool Query()
    {
        ThrowIfDisposed();

        var result = CudaNativeBindings.cuStreamQuery(_handle);

        if (result == CudaResult.Success)
        {
            return true;
        }

        if (result == CudaResult.NotReady)
        {
            return false;
        }

        CuBlasNative.CheckCudaResult(result, "cuStreamQuery");
        return false;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CudaStream));
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
                CudaNativeBindings.cuStreamDestroy(_handle);
            }
            catch
            {
                // Ignore destruction errors during disposal
            }
        }

        _handle = IntPtr.Zero;
    }
}
