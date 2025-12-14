using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.ModelCompression;

/// <summary>
/// Metadata for N-dimensional tensor compression operations that wraps the underlying vector compression metadata.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// TensorCompressionMetadata stores the information needed to decompress an N-dimensional weight tensor that was
/// compressed by first flattening it to a vector. It preserves the original tensor shape (dimensions) and
/// delegates the actual compression metadata to an inner ICompressionMetadata instance.
/// </para>
/// <para><b>For Beginners:</b> Tensors are multi-dimensional arrays used extensively in deep learning:
///
/// - 1D tensor (vector): [100] - like a bias term with 100 values
/// - 2D tensor (matrix): [100, 50] - like fully connected layer weights
/// - 3D tensor: [32, 100, 50] - like a batch of 32 matrices
/// - 4D tensor: [64, 3, 3, 3] - like 64 convolutional filters with 3 channels, 3x3 kernels
///
/// When compressing a tensor, we flatten it to a 1D array, compress it, and need to remember
/// the original shape to reconstruct it. This metadata stores:
///
/// 1. The original shape - the dimensions of the tensor (e.g., [64, 3, 3, 3])
/// 2. The inner compression details - how the flattened data was compressed
///
/// Think of it like packing a Rubik's cube for shipping:
/// - You disassemble it into individual pieces (flatten)
/// - You put the pieces in a compressed bag (apply compression)
/// - You include assembly instructions with the dimensions (this metadata)
/// - Later, you can perfectly reconstruct the original cube
/// </para>
/// </remarks>
public class TensorCompressionMetadata<T> : ICompressionMetadata<T>
{
    /// <summary>
    /// Initializes a new instance of the TensorCompressionMetadata class.
    /// </summary>
    /// <param name="originalShape">The shape (dimensions) of the original tensor.</param>
    /// <param name="innerMetadata">The compression metadata from the underlying vector compression.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When creating tensor compression metadata, you specify:
    ///
    /// - originalShape: The dimensions of the tensor (e.g., [64, 3, 3, 3] for conv filters)
    /// - innerMetadata: The details of how the flattened weights were compressed
    ///
    /// The shape array is copied to prevent external modifications from affecting the metadata.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when originalShape or innerMetadata is null.</exception>
    /// <exception cref="ArgumentException">Thrown when originalShape is empty or contains non-positive dimensions.</exception>
    public TensorCompressionMetadata(int[] originalShape, ICompressionMetadata<T> innerMetadata)
    {
        if (originalShape == null) throw new ArgumentNullException(nameof(originalShape));
        if (innerMetadata == null) throw new ArgumentNullException(nameof(innerMetadata));

        if (originalShape.Length == 0)
        {
            throw new ArgumentException("Tensor shape cannot be empty.", nameof(originalShape));
        }

        for (int i = 0; i < originalShape.Length; i++)
        {
            if (originalShape[i] <= 0)
            {
                throw new ArgumentException(
                    $"All tensor dimensions must be positive. Dimension {i} has value {originalShape[i]}.",
                    nameof(originalShape));
            }
        }

        // Copy the array to prevent external modifications
        OriginalShape = (int[])originalShape.Clone();
        InnerMetadata = innerMetadata;
    }

    /// <summary>
    /// Gets the compression type from the underlying compression algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This returns the actual compression algorithm type (like WeightClustering
    /// or HuffmanEncoding) that was used to compress the flattened tensor data. The tensor metadata itself
    /// is just a shape container - it delegates to the inner metadata for the real compression type.
    /// </para>
    /// </remarks>
    public CompressionType Type => InnerMetadata.Type;

    /// <summary>
    /// Gets the original total number of elements in the flattened tensor.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the total count of weight values in the original tensor,
    /// calculated by multiplying all dimensions together. For example:
    ///
    /// - A [64, 3, 3, 3] tensor has 64 * 3 * 3 * 3 = 1,728 elements
    /// - A [100, 50] tensor has 100 * 50 = 5,000 elements
    ///
    /// This is needed to allocate the right amount of memory when decompressing.
    /// </para>
    /// </remarks>
    public int OriginalLength
    {
        get
        {
            int length = 1;
            foreach (int dim in OriginalShape)
            {
                length *= dim;
            }
            return length;
        }
    }

    /// <summary>
    /// Gets the original shape (dimensions) of the tensor.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The shape array describes the structure of the original tensor:
    ///
    /// - [100] means a 1D tensor with 100 elements
    /// - [100, 50] means a 2D tensor (matrix) with 100 rows and 50 columns
    /// - [64, 3, 3, 3] means a 4D tensor (common for conv filters)
    ///
    /// This shape is essential for reshaping the decompressed data back into the correct tensor format.
    /// </para>
    /// </remarks>
    public int[] OriginalShape { get; }

    /// <summary>
    /// Gets the number of dimensions in the original tensor.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how many dimensions the tensor has:
    ///
    /// - 1 for vectors
    /// - 2 for matrices
    /// - 3+ for higher-dimensional tensors (common in deep learning)
    /// </para>
    /// </remarks>
    public int Rank => OriginalShape.Length;

    /// <summary>
    /// Gets the inner compression metadata from the underlying vector compression algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This contains the actual compression details from whatever
    /// algorithm was used (weight clustering, Huffman encoding, pruning, etc.). The tensor compression
    /// is just a wrapper that remembers the shape; the real compression work is tracked here.
    /// </para>
    /// </remarks>
    public ICompressionMetadata<T> InnerMetadata { get; }

    /// <summary>
    /// Gets the total size in bytes of this metadata structure, including the inner metadata.
    /// </summary>
    /// <returns>The metadata size in bytes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When calculating the total compressed size, you need to include
    /// the metadata overhead. This method calculates:
    ///
    /// - The size of the rank (1 integer = 4 bytes, to know how many dimensions to read)
    /// - The size of the shape array (rank integers, each 4 bytes)
    /// - Plus the size of the inner compression metadata
    ///
    /// For example, a 4D tensor adds 4 + 4*4 = 20 bytes of shape overhead.
    /// </para>
    /// </remarks>
    public long GetMetadataSize()
    {
        // Size = Rank (int, to know array length) + OriginalShape (int[]) + InnerMetadata size
        return sizeof(int) + (OriginalShape.Length * sizeof(int)) + InnerMetadata.GetMetadataSize();
    }
}
