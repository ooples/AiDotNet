using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// GPU-resident tensor implementation with lazy CPU synchronization.
/// Data remains on GPU until explicitly downloaded via GetCpuData() or ToTensor().
/// </summary>
/// <typeparam name="T">The element type of the tensor.</typeparam>
/// <remarks>
/// <para><b>Usage Pattern:</b></para>
/// <code>
/// // Create GPU tensor from CPU data
/// var gpuTensor = new GpuTensor&lt;float&gt;(backend, cpuData, shape, GpuTensorRole.Activation);
///
/// // Operations keep data on GPU
/// var result = engine.MatMul(gpuTensor, weights);
///
/// // Data only downloaded when explicitly requested
/// var cpuResult = result.ToTensor();  // &lt;-- Synchronizes here
/// </code>
/// </remarks>
public sealed class GpuTensor<T> : IGpuTensor<T>, IGpuTensor
{
    private readonly IDirectGpuBackend _backend;
    private readonly INumericOperations<T> _numOps;
    private readonly bool _ownsBuffer;
    private T[]? _cpuCache;
    private bool _disposed;

    /// <inheritdoc/>
    public IGpuBuffer Buffer { get; }

    /// <inheritdoc/>
    public int[] Shape { get; }

    /// <inheritdoc/>
    public int ElementCount { get; }

    /// <inheritdoc/>
    public GpuTensorRole Role { get; }

    /// <inheritdoc/>
    public GpuSyncPoint? LastWriteSync { get; private set; }

    /// <inheritdoc/>
    public bool IsDirty { get; private set; }

    /// <inheritdoc/>
    public Type ElementType => typeof(T);

    /// <summary>
    /// Creates a GPU tensor from an existing buffer.
    /// </summary>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="buffer">The existing GPU buffer.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <param name="ownsBuffer">If true, the tensor owns the buffer and will dispose it.</param>
    public GpuTensor(IDirectGpuBackend backend, IGpuBuffer buffer, int[] shape, GpuTensorRole role, bool ownsBuffer = true)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        Buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
        Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        Role = role;
        _ownsBuffer = ownsBuffer;
        _numOps = MathHelper.GetNumericOperations<T>();

        ElementCount = 1;
        foreach (var dim in shape)
        {
            ElementCount *= dim;
        }

        // Validate buffer size matches shape
        if (buffer.Size < ElementCount)
        {
            throw new ArgumentException(
                $"Buffer size ({buffer.Size}) is smaller than tensor element count ({ElementCount}).");
        }

        IsDirty = true; // Assume buffer has data that differs from any CPU cache
    }

    /// <summary>
    /// Creates a GPU tensor by uploading CPU data.
    /// </summary>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="data">The CPU data to upload.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    public GpuTensor(IDirectGpuBackend backend, T[] data, int[] shape, GpuTensorRole role)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        Role = role;
        _ownsBuffer = true;
        _numOps = MathHelper.GetNumericOperations<T>();

        ElementCount = 1;
        foreach (var dim in shape)
        {
            ElementCount *= dim;
        }

        if (data == null || data.Length < ElementCount)
        {
            throw new ArgumentException(
                $"Data length ({data?.Length ?? 0}) is smaller than tensor element count ({ElementCount}).");
        }

        // Convert to float and upload
        float[] floatData = ConvertToFloat(data);
        Buffer = backend.AllocateBuffer(floatData);

        // Cache the CPU data
        _cpuCache = data;
        IsDirty = false;
    }

    /// <summary>
    /// Creates a GPU tensor from a CPU Tensor.
    /// </summary>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="tensor">The CPU tensor to upload.</param>
    /// <param name="role">The role of this tensor.</param>
    public GpuTensor(IDirectGpuBackend backend, Tensor<T> tensor, GpuTensorRole role)
        : this(backend, tensor.Data, tensor.Shape, role)
    {
    }

    /// <inheritdoc/>
    public T[] GetCpuData()
    {
        ThrowIfDisposed();
        EnsureSynchronized();

        if (_cpuCache != null && !IsDirty)
        {
            return _cpuCache;
        }

        // Download from GPU
        float[] floatData = _backend.DownloadBuffer(Buffer);
        _cpuCache = ConvertFromFloat(floatData);
        IsDirty = false;

        return _cpuCache;
    }

    /// <inheritdoc/>
    public Tensor<T> ToTensor()
    {
        var data = GetCpuData();
        return new Tensor<T>(data, Shape);
    }

    /// <inheritdoc/>
    public void Synchronize()
    {
        ThrowIfDisposed();
        EnsureSynchronized();
    }

    /// <inheritdoc/>
    public void MarkModified(GpuSyncPoint? syncPoint)
    {
        ThrowIfDisposed();
        IsDirty = true;
        _cpuCache = null;
        LastWriteSync = syncPoint;
    }

    /// <inheritdoc/>
    public IGpuTensor<T> CreateView(int offset, int[] shape)
    {
        ThrowIfDisposed();

        int viewElements = 1;
        foreach (var dim in shape)
        {
            viewElements *= dim;
        }

        if (offset < 0 || offset + viewElements > ElementCount)
        {
            throw new ArgumentOutOfRangeException(nameof(offset),
                $"View with offset {offset} and {viewElements} elements exceeds tensor bounds.");
        }

        // For now, views share the same buffer (read-only semantics assumed)
        // A full implementation would create a sub-buffer view
        return new GpuTensorView<T>(this, offset, shape);
    }

    private void EnsureSynchronized()
    {
        if (LastWriteSync != null && !LastWriteSync.IsComplete)
        {
            LastWriteSync.Wait();
        }
    }

    private float[] ConvertToFloat(T[] data)
    {
        if (typeof(T) == typeof(float))
        {
            return (float[])(object)data;
        }

        float[] result = new float[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            result[i] = (float)_numOps.ToDouble(data[i]);
        }
        return result;
    }

    private T[] ConvertFromFloat(float[] data)
    {
        if (typeof(T) == typeof(float))
        {
            return (T[])(object)data;
        }

        T[] result = new T[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            result[i] = _numOps.FromDouble(data[i]);
        }
        return result;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GpuTensor<T>));
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

        // Wait for any pending operations
        if (LastWriteSync != null && !LastWriteSync.IsComplete)
        {
            try
            {
                LastWriteSync.Wait();
            }
            catch
            {
                // Ignore synchronization errors during disposal
            }
        }

        // Dispose the buffer if we own it
        if (_ownsBuffer)
        {
            Buffer.Dispose();
        }

        // Clear CPU cache
        _cpuCache = null;
    }
}

/// <summary>
/// A view into a portion of a GpuTensor without copying data.
/// </summary>
internal sealed class GpuTensorView<T> : IGpuTensor<T>
{
    private readonly GpuTensor<T> _parent;
    private readonly int _offset;
    private bool _disposed;

    public IGpuBuffer Buffer => _parent.Buffer;
    public int[] Shape { get; }
    public int ElementCount { get; }
    public GpuTensorRole Role => _parent.Role;
    public GpuSyncPoint? LastWriteSync => _parent.LastWriteSync;
    public bool IsDirty => _parent.IsDirty;
    public Type ElementType => typeof(T);

    public GpuTensorView(GpuTensor<T> parent, int offset, int[] shape)
    {
        _parent = parent;
        _offset = offset;
        Shape = shape;

        ElementCount = 1;
        foreach (var dim in shape)
        {
            ElementCount *= dim;
        }
    }

    public T[] GetCpuData()
    {
        ThrowIfDisposed();

        // Get parent's data and extract the view portion
        var parentData = _parent.GetCpuData();
        var viewData = new T[ElementCount];
        Array.Copy(parentData, _offset, viewData, 0, ElementCount);
        return viewData;
    }

    public Tensor<T> ToTensor()
    {
        return new Tensor<T>(GetCpuData(), Shape);
    }

    public void Synchronize()
    {
        _parent.Synchronize();
    }

    public void MarkModified(GpuSyncPoint? syncPoint)
    {
        _parent.MarkModified(syncPoint);
    }

    public IGpuTensor<T> CreateView(int offset, int[] shape)
    {
        return _parent.CreateView(_offset + offset, shape);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GpuTensorView<T>));
        }
    }

    public void Dispose()
    {
        // Views don't own the underlying buffer
        _disposed = true;
    }
}

/// <summary>
/// Factory methods for creating GPU tensors.
/// </summary>
public static class GpuTensorFactory
{
    /// <summary>
    /// Creates an empty GPU tensor with the specified shape.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A new GPU tensor with uninitialized data.</returns>
    public static GpuTensor<T> Empty<T>(IDirectGpuBackend backend, int[] shape, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        int elementCount = 1;
        foreach (var dim in shape)
        {
            elementCount *= dim;
        }

        var buffer = backend.AllocateBuffer(elementCount);
        return new GpuTensor<T>(backend, buffer, shape, role);
    }

    /// <summary>
    /// Creates a GPU tensor filled with zeros.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A new GPU tensor filled with zeros.</returns>
    public static GpuTensor<T> Zeros<T>(IDirectGpuBackend backend, int[] shape, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        int elementCount = 1;
        foreach (var dim in shape)
        {
            elementCount *= dim;
        }

        var buffer = backend.AllocateBuffer(elementCount);
        backend.Fill(buffer, 0f, elementCount);
        return new GpuTensor<T>(backend, buffer, shape, role);
    }

    /// <summary>
    /// Creates a GPU tensor filled with ones.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A new GPU tensor filled with ones.</returns>
    public static GpuTensor<T> Ones<T>(IDirectGpuBackend backend, int[] shape, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        int elementCount = 1;
        foreach (var dim in shape)
        {
            elementCount *= dim;
        }

        var buffer = backend.AllocateBuffer(elementCount);
        backend.Fill(buffer, 1f, elementCount);
        return new GpuTensor<T>(backend, buffer, shape, role);
    }

    /// <summary>
    /// Creates a GPU tensor from a CPU tensor.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="tensor">The CPU tensor to upload.</param>
    /// <param name="role">The role of this tensor.</param>
    /// <returns>A new GPU tensor with the data uploaded.</returns>
    public static GpuTensor<T> FromTensor<T>(IDirectGpuBackend backend, Tensor<T> tensor, GpuTensorRole role = GpuTensorRole.General)
    {
        return new GpuTensor<T>(backend, tensor, role);
    }

    /// <summary>
    /// Creates a GPU tensor for weights (persistent).
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="data">The weight data.</param>
    /// <param name="shape">The shape of the weights.</param>
    /// <returns>A new GPU tensor marked as weights.</returns>
    public static GpuTensor<T> ForWeights<T>(IDirectGpuBackend backend, T[] data, int[] shape)
    {
        return new GpuTensor<T>(backend, data, shape, GpuTensorRole.Weight);
    }

    /// <summary>
    /// Creates a GPU tensor for biases (persistent).
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="backend">The GPU backend.</param>
    /// <param name="data">The bias data.</param>
    /// <returns>A new GPU tensor marked as bias.</returns>
    public static GpuTensor<T> ForBias<T>(IDirectGpuBackend backend, T[] data)
    {
        return new GpuTensor<T>(backend, data, [data.Length], GpuTensorRole.Bias);
    }
}
