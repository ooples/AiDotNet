using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.ModelCompression;

/// <summary>
/// Metadata for matrix compression operations that wraps the underlying vector compression metadata.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// MatrixCompressionMetadata stores the information needed to decompress a 2D weight matrix that was
/// compressed by first flattening it to a vector. It preserves the original matrix dimensions and
/// delegates the actual compression metadata to an inner ICompressionMetadata instance.
/// </para>
/// <para><b>For Beginners:</b> When compressing a 2D matrix (like weights in a fully connected layer),
/// we need to remember:
///
/// 1. The original shape - how many rows and columns the matrix had
/// 2. How the flattened data was compressed (the inner compression details)
///
/// Think of it like folding a shirt to pack in a suitcase:
/// - You flatten the shirt (2D to 1D)
/// - You compress it in a vacuum bag (apply compression algorithm)
/// - You need to remember the original shirt size to unfold it properly later
///
/// This metadata class keeps track of all that information so we can perfectly restore
/// the original matrix shape after decompression.
/// </para>
/// </remarks>
public class MatrixCompressionMetadata<T> : ICompressionMetadata<T>
{
    private readonly int _elementSize;

    /// <summary>
    /// Initializes a new instance of the MatrixCompressionMetadata class.
    /// </summary>
    /// <param name="originalRows">The number of rows in the original matrix.</param>
    /// <param name="originalColumns">The number of columns in the original matrix.</param>
    /// <param name="innerMetadata">The compression metadata from the underlying vector compression.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When creating matrix compression metadata, you specify:
    ///
    /// - originalRows: How tall the matrix was (e.g., 100 for a 100x50 matrix)
    /// - originalColumns: How wide the matrix was (e.g., 50 for a 100x50 matrix)
    /// - innerMetadata: The details of how the flattened weights were compressed
    ///
    /// This information is essential for restoring the matrix to its exact original shape.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when rows or columns are not positive.</exception>
    /// <exception cref="ArgumentNullException">Thrown when innerMetadata is null.</exception>
    public MatrixCompressionMetadata(int originalRows, int originalColumns, ICompressionMetadata<T> innerMetadata)
    {
        if (originalRows <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(originalRows), "Number of rows must be positive.");
        }

        if (originalColumns <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(originalColumns), "Number of columns must be positive.");
        }

        if (innerMetadata == null) throw new ArgumentNullException(nameof(innerMetadata));

        OriginalRows = originalRows;
        OriginalColumns = originalColumns;
        InnerMetadata = innerMetadata;

        _elementSize = typeof(T) == typeof(float) ? 4 :
                       typeof(T) == typeof(double) ? 8 :
                       System.Runtime.InteropServices.Marshal.SizeOf(typeof(T));
    }

    /// <summary>
    /// Gets the compression type from the underlying compression algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This returns the actual compression algorithm type (like WeightClustering
    /// or HuffmanEncoding) that was used to compress the flattened matrix data. The matrix metadata itself
    /// is just a shape container - it delegates to the inner metadata for the real compression type.
    /// </para>
    /// </remarks>
    public CompressionType Type => InnerMetadata.Type;

    /// <summary>
    /// Gets the original total number of elements in the flattened matrix.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the total count of weight values in the original matrix,
    /// calculated as rows multiplied by columns. For example, a 100x50 matrix has 5,000 elements.
    /// This is needed to allocate the right amount of memory when decompressing.
    /// </para>
    /// </remarks>
    public int OriginalLength => checked(OriginalRows * OriginalColumns);

    /// <summary>
    /// Gets the number of rows in the original matrix.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The row count of the original weight matrix before compression.
    /// In neural networks, this often corresponds to the number of input features or neurons.
    /// </para>
    /// </remarks>
    public int OriginalRows { get; }

    /// <summary>
    /// Gets the number of columns in the original matrix.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The column count of the original weight matrix before compression.
    /// In neural networks, this often corresponds to the number of output neurons in a layer.
    /// </para>
    /// </remarks>
    public int OriginalColumns { get; }

    /// <summary>
    /// Gets the inner compression metadata from the underlying vector compression algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This contains the actual compression details from whatever
    /// algorithm was used (weight clustering, Huffman encoding, etc.). The matrix compression
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
    /// - The size of the shape information (2 integers for rows and columns = 8 bytes)
    /// - Plus the size of the inner compression metadata
    ///
    /// This gives an accurate picture of the total storage needed for decompression.
    /// </para>
    /// </remarks>
    public long GetMetadataSize()
    {
        // Size = OriginalRows (int) + OriginalColumns (int) + InnerMetadata size
        return sizeof(int) + sizeof(int) + InnerMetadata.GetMetadataSize();
    }
}
