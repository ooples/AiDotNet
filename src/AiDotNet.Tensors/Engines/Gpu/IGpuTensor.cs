using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Represents a GPU-resident tensor with lazy CPU synchronization.
/// Data remains on GPU until explicitly downloaded via GetCpuData() or ToTensor().
/// </summary>
/// <typeparam name="T">The element type of the tensor.</typeparam>
public interface IGpuTensor<T> : IDisposable
{
    /// <summary>
    /// Gets the underlying GPU buffer.
    /// </summary>
    IGpuBuffer Buffer { get; }

    /// <summary>
    /// Gets the shape of the tensor.
    /// </summary>
    int[] Shape { get; }

    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// </summary>
    int ElementCount { get; }

    /// <summary>
    /// Gets the role of this tensor for memory management decisions.
    /// </summary>
    GpuTensorRole Role { get; }

    /// <summary>
    /// Gets the synchronization point for the last write operation.
    /// Can be null if no async operations have been performed.
    /// </summary>
    GpuSyncPoint? LastWriteSync { get; }

    /// <summary>
    /// Gets whether the tensor data has been modified on GPU since last CPU sync.
    /// </summary>
    bool IsDirty { get; }

    /// <summary>
    /// Gets the CPU data, synchronizing if necessary.
    /// This is a blocking operation if async operations are pending.
    /// </summary>
    /// <returns>The tensor data as a CPU array.</returns>
    T[] GetCpuData();

    /// <summary>
    /// Converts this GPU tensor to a CPU Tensor.
    /// This is a blocking operation if async operations are pending.
    /// </summary>
    /// <returns>A new CPU Tensor with the data.</returns>
    Tensor<T> ToTensor();

    /// <summary>
    /// Ensures all pending GPU operations on this tensor are complete.
    /// Call this before accessing the buffer from another stream.
    /// </summary>
    void Synchronize();

    /// <summary>
    /// Marks the tensor as modified by a GPU operation.
    /// </summary>
    /// <param name="syncPoint">The sync point for the modifying operation.</param>
    void MarkModified(GpuSyncPoint? syncPoint);

    /// <summary>
    /// Creates a view of a portion of this tensor without copying data.
    /// The view shares the same underlying GPU buffer.
    /// </summary>
    /// <param name="offset">The starting offset in elements.</param>
    /// <param name="shape">The shape of the view.</param>
    /// <returns>A new GPU tensor view.</returns>
    IGpuTensor<T> CreateView(int offset, int[] shape);
}

/// <summary>
/// Non-generic interface for GPU tensors when element type is not known at compile time.
/// </summary>
public interface IGpuTensor : IDisposable
{
    /// <summary>
    /// Gets the underlying GPU buffer.
    /// </summary>
    IGpuBuffer Buffer { get; }

    /// <summary>
    /// Gets the shape of the tensor.
    /// </summary>
    int[] Shape { get; }

    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// </summary>
    int ElementCount { get; }

    /// <summary>
    /// Gets the role of this tensor for memory management decisions.
    /// </summary>
    GpuTensorRole Role { get; }

    /// <summary>
    /// Gets the synchronization point for the last write operation.
    /// </summary>
    GpuSyncPoint? LastWriteSync { get; }

    /// <summary>
    /// Gets whether the tensor data has been modified on GPU since last CPU sync.
    /// </summary>
    bool IsDirty { get; }

    /// <summary>
    /// Gets the element type of the tensor.
    /// </summary>
    Type ElementType { get; }

    /// <summary>
    /// Ensures all pending GPU operations on this tensor are complete.
    /// </summary>
    void Synchronize();
}
