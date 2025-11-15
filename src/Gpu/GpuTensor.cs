using AiDotNet.Enums;
using ILGPU.Runtime;

namespace AiDotNet.Gpu;

/// <summary>
/// Represents a tensor stored in GPU memory.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <remarks>
/// <para>
/// GpuTensor wraps GPU memory buffers and provides a tensor interface.
/// It tracks the tensor's shape and location, and handles memory lifecycle.
/// </para>
/// <para><b>For Beginners:</b> This is like a regular tensor, but the data lives on the GPU.
///
/// Key differences from CPU tensors:
/// - Data stored in graphics card memory (much faster for parallel operations)
/// - Cannot directly access individual elements from CPU code
/// - Must transfer to CPU to read/modify values directly
/// - Operations execute much faster when data stays on GPU
///
/// Think of it like files on a remote server:
/// - Faster to process them where they are
/// - Slower to download/upload constantly
/// - Keep them there as long as you're working with them
/// </para>
/// </remarks>
public class GpuTensor<T> : IDisposable
    where T : unmanaged
{
    /// <summary>
    /// Gets the GPU memory buffer containing the tensor data.
    /// </summary>
    internal MemoryBuffer1D<T, Stride1D.Dense> Buffer { get; private set; }

    /// <summary>
    /// Gets the shape of the tensor.
    /// </summary>
    public int[] Shape { get; }

    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets the rank (number of dimensions) of the tensor.
    /// </summary>
    public int Rank => Shape.Length;

    /// <summary>
    /// Gets the location of this tensor (always GPU).
    /// </summary>
    public TensorLocation Location => TensorLocation.GPU;

    /// <summary>
    /// Gets the backend that manages this GPU tensor.
    /// </summary>
    internal IGpuBackend<T>? Backend { get; set; }

    /// <summary>
    /// Gets a value indicating whether this tensor has been disposed.
    /// </summary>
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuTensor{T}"/> class.
    /// </summary>
    /// <param name="buffer">The GPU memory buffer.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="backend">Optional backend reference for operations.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a GPU tensor from an existing GPU memory buffer.
    ///
    /// Usually you don't create these directly - instead you use methods like:
    /// - backend.Allocate(shape) - Allocate new GPU memory
    /// - backend.ToGpu(cpuTensor) - Transfer from CPU to GPU
    /// </para>
    /// </remarks>
    public GpuTensor(MemoryBuffer1D<T, Stride1D.Dense> buffer, int[] shape, IGpuBackend<T>? backend = null)
    {
        Buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
        Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        Backend = backend;

        // Calculate total length
        Length = 1;
        foreach (var dim in shape)
        {
            if (dim <= 0)
            {
                throw new ArgumentException($"Invalid shape dimension: {dim}. All dimensions must be positive.");
            }
            Length *= dim;
        }

        // Verify buffer size matches shape
        if (buffer.Length != Length)
        {
            throw new ArgumentException(
                $"Buffer length ({buffer.Length}) does not match shape length ({Length}).");
        }
    }

    /// <summary>
    /// Converts a flat index to multi-dimensional indices.
    /// </summary>
    /// <param name="flatIndex">The flat index to convert.</param>
    /// <param name="indices">An array to store the resulting indices.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Converts a single number into coordinates.
    ///
    /// Example: For a 3x4 tensor (3 rows, 4 columns):
    /// - flatIndex 0 → indices [0, 0] (first element)
    /// - flatIndex 5 → indices [1, 1] (second row, second column)
    /// - flatIndex 11 → indices [2, 3] (last element)
    ///
    /// This is useful for understanding which "cell" an element represents.
    /// </para>
    /// </remarks>
    public void GetIndices(int flatIndex, int[] indices)
    {
        if (indices.Length != Rank)
        {
            throw new ArgumentException($"Indices array must have length {Rank}");
        }

        int remainder = flatIndex;
        for (int i = Rank - 1; i >= 0; i--)
        {
            indices[i] = remainder % Shape[i];
            remainder /= Shape[i];
        }
    }

    /// <summary>
    /// Converts multi-dimensional indices to a flat index.
    /// </summary>
    /// <param name="indices">The multi-dimensional indices.</param>
    /// <returns>The corresponding flat index.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Converts coordinates into a single number.
    ///
    /// This is the reverse of GetIndices:
    /// - indices [0, 0] → flatIndex 0
    /// - indices [1, 1] → flatIndex 5 (for a 3x4 tensor)
    /// - indices [2, 3] → flatIndex 11
    ///
    /// GPUs store data in a flat array, so we need this conversion.
    /// </para>
    /// </remarks>
    public int GetFlatIndex(int[] indices)
    {
        if (indices.Length != Rank)
        {
            throw new ArgumentException($"Indices array must have length {Rank}");
        }

        int flatIndex = 0;
        int multiplier = 1;

        for (int i = Rank - 1; i >= 0; i--)
        {
            if (indices[i] < 0 || indices[i] >= Shape[i])
            {
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Index {i} is out of range: {indices[i]} (shape dimension: {Shape[i]})");
            }

            flatIndex += indices[i] * multiplier;
            multiplier *= Shape[i];
        }

        return flatIndex;
    }

    /// <summary>
    /// Returns a string representation of the GPU tensor.
    /// </summary>
    /// <returns>A string describing the tensor.</returns>
    public override string ToString()
    {
        return $"GpuTensor<{typeof(T).Name}> with shape [{string.Join(", ", Shape)}] on {Location}";
    }

    /// <summary>
    /// Disposes the GPU tensor, freeing its memory.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This releases the GPU memory used by this tensor.
    ///
    /// IMPORTANT: Always dispose GPU tensors when you're done with them!
    /// - GPU memory is limited (usually 4-16 GB)
    /// - Not disposing can lead to out-of-memory errors
    /// - Use 'using' statements to ensure cleanup:
    ///
    /// <code>
    /// using (var gpuTensor = backend.Allocate(shape))
    /// {
    ///     // Use the tensor
    /// } // Automatically disposed here
    /// </code>
    /// </para>
    /// </remarks>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        Buffer?.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Finalizer to ensure GPU memory is freed even if Dispose is not called.
    /// </summary>
    ~GpuTensor()
    {
        Dispose();
    }
}
