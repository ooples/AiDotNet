using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimized;

/// <summary>
/// Base interface for tensors stored in hardware-optimized memory formats.
/// Implemented by both GPU-resident tensors and CPU tensors in blocked formats.
/// </summary>
/// <remarks>
/// <para>
/// This interface provides a unified abstraction for tensors that may be stored in
/// hardware-specific optimized formats (GPU memory, CPU blocked layouts, etc.).
/// It allows operations to efficiently work with data in its optimal format while
/// providing conversion back to standard format when needed.
/// </para>
/// <para><b>For Beginners:</b> When neural networks run, data needs to be stored in memory.
///
/// Different hardware works best with data organized differently:
/// - GPUs like data in their own memory (GPU-resident)
/// - CPUs work faster with data arranged in special patterns (blocked formats)
///
/// This interface is like a common language that all these different storage methods
/// share, so the rest of the code doesn't need to know the specific details.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type of the tensor elements (e.g., float, double).</typeparam>
public interface IOptimizedTensor<T> : IDisposable
{
    /// <summary>
    /// Gets the logical shape of the tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The shape represents the dimensions of the tensor regardless of how it's stored internally.
    /// For example, a 4D tensor for convolution might have shape [batch, channels, height, width].
    /// </para>
    /// <para><b>For Beginners:</b> The shape tells you how the data is organized conceptually.
    /// Even if the data is stored in a special optimized format internally, the shape stays
    /// the same - like how a photo is still 1920x1080 pixels even when compressed to JPEG.
    /// </para>
    /// </remarks>
    int[] Shape { get; }

    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the product of all dimensions in the shape. For a tensor with shape [2, 3, 4],
    /// the element count would be 2 * 3 * 4 = 24.
    /// </para>
    /// </remarks>
    int ElementCount { get; }

    /// <summary>
    /// Gets the current memory format of the tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This indicates how the tensor data is currently stored in memory. Operations can check
    /// this to determine if data needs to be converted before processing, or if it can be
    /// used directly.
    /// </para>
    /// </remarks>
    OptimizedTensorFormat Format { get; }

    /// <summary>
    /// Converts this optimized tensor to a standard CPU Tensor.
    /// </summary>
    /// <returns>A new CPU Tensor with the data in standard NCHW format.</returns>
    /// <remarks>
    /// <para>
    /// This method may trigger a data transfer (for GPU-resident tensors) or a reorder
    /// operation (for blocked format tensors). The returned tensor is always in standard
    /// row-major format and can be used with any operation.
    /// </para>
    /// <para><b>For Beginners:</b> This converts the optimized data back to the standard format
    /// that all operations can understand. It's like converting a JPEG back to raw pixels -
    /// you can work with it normally, but there may be some cost to do the conversion.
    /// </para>
    /// </remarks>
    Tensor<T> ToTensor();

    /// <summary>
    /// Ensures the tensor is in standard (non-optimized) format.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method converts the tensor to standard format in-place if it isn't already.
    /// For tensors already in standard format, this is a no-op.
    /// </para>
    /// <para>
    /// Note: Not all optimized tensor implementations support in-place conversion.
    /// Use <see cref="ToTensor"/> if you need a guaranteed standard-format tensor.
    /// </para>
    /// </remarks>
    void EnsureStandardFormat();
}

/// <summary>
/// Non-generic base interface for optimized tensors when element type is not known at compile time.
/// </summary>
/// <remarks>
/// <para>
/// This interface provides type-erased access to optimized tensor properties, useful when
/// working with collections of tensors with different element types or in reflection-based
/// scenarios.
/// </para>
/// </remarks>
public interface IOptimizedTensor : IDisposable
{
    /// <summary>
    /// Gets the logical shape of the tensor.
    /// </summary>
    int[] Shape { get; }

    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// </summary>
    int ElementCount { get; }

    /// <summary>
    /// Gets the current memory format of the tensor.
    /// </summary>
    OptimizedTensorFormat Format { get; }

    /// <summary>
    /// Gets the element type of the tensor (e.g., typeof(float), typeof(double)).
    /// </summary>
    Type ElementType { get; }
}
