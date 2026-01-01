#if !NET462
using ILGPU;
using ILGPU.Runtime;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Low-level GPU memory buffer handle with explicit lifecycle management.
/// Wraps ILGPU's MemoryBuffer1D for persistent GPU tensor storage.
/// </summary>
/// <typeparam name="T">The unmanaged element type (float, double).</typeparam>
/// <remarks>
/// <para><b>Phase B: Persistent GPU Tensors (US-GPU-030)</b></para>
/// <para>
/// GpuTensorHandle provides direct GPU memory access without automatic pooling.
/// Unlike pooled buffers, these handles maintain persistent GPU memory for
/// weights and biases that stay on GPU across multiple forward/backward passes.
/// </para>
/// <para><b>Key Design:</b>
/// - Explicit disposal (not pooled) - caller owns the lifecycle
/// - Thread-safe via external lock (GpuEngine._gpuLock)
/// - Tracks dirty state for lazy CPU sync
/// - Direct ArrayView access for kernel operations
/// </para>
/// <para><b>Usage Pattern:</b>
/// <code>
/// // Allocate persistent GPU buffer for weights
/// var handle = gpuEngine.AllocateGpuTensor&lt;float&gt;(weightsData);
///
/// // Use in kernels via ArrayView
/// kernel(handle.View, ...);
///
/// // Sync back to CPU when needed
/// var cpuData = handle.ToCpuArray();
///
/// // Explicit dispose when done
/// handle.Dispose();
/// </code>
/// </para>
/// </remarks>
public sealed class GpuTensorHandle<T> : IDisposable where T : unmanaged
{
    private readonly MemoryBuffer1D<T, Stride1D.Dense> _buffer;
    private readonly int[] _shape;
    private bool _disposed;
    private bool _gpuDirty;

    /// <summary>
    /// Gets the underlying ILGPU memory buffer.
    /// </summary>
    internal MemoryBuffer1D<T, Stride1D.Dense> Buffer => _buffer;

    /// <summary>
    /// Gets the 1D ArrayView for use in GPU kernels that expect 1D views.
    /// </summary>
    public ArrayView1D<T, Stride1D.Dense> View1D => _buffer.View;

    /// <summary>
    /// Gets the base ArrayView for use in simple GPU kernels.
    /// </summary>
    public ArrayView<T> View => _buffer.View.BaseView;

    /// <summary>
    /// Gets the total number of elements in the buffer.
    /// </summary>
    public long Length => _buffer.Length;

    /// <summary>
    /// Gets the shape of the tensor (dimensions).
    /// </summary>
    public int[] Shape => _shape;

    /// <summary>
    /// Gets whether the GPU buffer has been modified since last CPU sync.
    /// </summary>
    public bool IsGpuDirty => _gpuDirty;

    /// <summary>
    /// Creates a new GPU tensor handle with the specified buffer and shape.
    /// </summary>
    /// <param name="buffer">The ILGPU memory buffer.</param>
    /// <param name="shape">The tensor shape (dimensions).</param>
    internal GpuTensorHandle(MemoryBuffer1D<T, Stride1D.Dense> buffer, int[] shape)
    {
        _buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
        _shape = shape ?? throw new ArgumentNullException(nameof(shape));
        _gpuDirty = false;
        _disposed = false;
    }

    /// <summary>
    /// Marks the GPU buffer as modified (dirty), indicating CPU sync is needed.
    /// Call this after any GPU kernel modifies the buffer contents.
    /// </summary>
    public void MarkDirty()
    {
        ThrowIfDisposed();
        _gpuDirty = true;
    }

    /// <summary>
    /// Clears the dirty flag after CPU synchronization.
    /// </summary>
    internal void ClearDirty()
    {
        _gpuDirty = false;
    }

    /// <summary>
    /// Copies the GPU buffer contents to a CPU array.
    /// </summary>
    /// <returns>A new array containing the GPU buffer data.</returns>
    public T[] ToCpuArray()
    {
        ThrowIfDisposed();
        var result = new T[_buffer.Length];
        _buffer.CopyToCPU(result);
        _gpuDirty = false;
        return result;
    }

    /// <summary>
    /// Copies the GPU buffer contents to an existing CPU array.
    /// </summary>
    /// <param name="destination">The destination array (must be at least Length elements).</param>
    public void CopyToCpu(T[] destination)
    {
        ThrowIfDisposed();
        if (destination == null)
            throw new ArgumentNullException(nameof(destination));
        if (destination.Length < _buffer.Length)
            throw new ArgumentException($"Destination array too small. Required: {_buffer.Length}, provided: {destination.Length}", nameof(destination));

        _buffer.CopyToCPU(destination);
        _gpuDirty = false;
    }

    /// <summary>
    /// Updates the GPU buffer from CPU data.
    /// </summary>
    /// <param name="source">The source CPU array.</param>
    public void CopyFromCpu(T[] source)
    {
        ThrowIfDisposed();
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (source.Length != _buffer.Length)
            throw new ArgumentException($"Source array size mismatch. Expected: {_buffer.Length}, provided: {source.Length}", nameof(source));

        _buffer.CopyFromCPU(source);
        _gpuDirty = false;
    }

    /// <summary>
    /// Sets all elements in the GPU buffer to zero.
    /// </summary>
    /// <remarks>
    /// MemSetToZero is asynchronous on the default stream. We explicitly synchronize
    /// to ensure the buffer is zeroed before setting the dirty flag.
    /// </remarks>
    public void Clear()
    {
        ThrowIfDisposed();
        _buffer.MemSetToZero();
        // MemSetToZero is asynchronous by default - explicitly sync to ensure completion
        _buffer.Accelerator.Synchronize();
        _gpuDirty = true;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GpuTensorHandle<T>));
    }

    /// <summary>
    /// Disposes the GPU memory buffer, releasing GPU resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;

        _buffer.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Finalizer to ensure GPU resources are released.
    /// </summary>
    ~GpuTensorHandle()
    {
        // Attempt cleanup if Dispose wasn't called
        // Note: This runs on finalizer thread, GPU operations may not be safe
        // This is a safety net, not the recommended disposal pattern
        if (!_disposed)
        {
            try
            {
                _buffer.Dispose();
            }
            catch (Exception ex) when (ex is ObjectDisposedException
                or InvalidOperationException
                or NotSupportedException
                or DllNotFoundException
                or PlatformNotSupportedException)
            {
                // Swallow exceptions in finalizer - throwing from finalizers is dangerous
            }
            _disposed = true;
        }
    }
}
#endif
