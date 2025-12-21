using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ModelCompression;

/// <summary>
/// Provides a base implementation for model compression techniques used to reduce model size while preserving accuracy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// ModelCompressionBase serves as an abstract foundation for implementing various compression strategies.
/// Model compression reduces the storage and computational requirements of machine learning models,
/// making them more suitable for deployment on resource-constrained devices or in bandwidth-limited environments.
/// </para>
/// <para><b>For Beginners:</b> Think of model compression as packing for a trip - you want to fit everything
/// you need into a smaller suitcase.
///
/// When you train an AI model:
/// - It learns millions or billions of parameters (weights)
/// - These weights need to be stored and loaded when making predictions
/// - Larger models are slower and use more memory
///
/// Model compression helps by:
/// - Reducing the file size (easier to download and store)
/// - Speeding up predictions (less data to process)
/// - Enabling deployment on phones, tablets, or embedded devices
/// - Lowering costs in cloud environments
///
/// This base class provides the common structure that all compression techniques share. Different
/// compression approaches (like weight clustering, quantization, or Huffman coding) work in different
/// ways, but they all aim to make your model smaller and faster while keeping it accurate.
/// </para>
/// </remarks>
public abstract class ModelCompressionBase<T> : IModelCompressionStrategy<T>
{
    /// <summary>
    /// Provides numeric operations appropriate for the generic type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds a reference to the appropriate numeric operations implementation for the
    /// generic type T, allowing the compression methods to perform mathematical operations
    /// regardless of whether T is float, double, or another numeric type.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper that allows the code to work with different number types.
    ///
    /// Since this class uses a generic type T (which could be float, double, etc.):
    /// - We need a way to perform math operations (+, -, *, /) on these values
    /// - NumOps provides the right methods for whatever numeric type is being used
    ///
    /// Think of it like having different calculators for different types of numbers,
    /// and NumOps makes sure we're using the right calculator for the job.
    /// </para>
    /// </remarks>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Initializes a new instance of the ModelCompressionBase class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor initializes the base class for a compression implementation,
    /// setting up the numeric operations required for mathematical calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the foundation for any type of compression.
    ///
    /// When creating a compression object, it gets the right calculator for the numeric type being used.
    /// This is like preparing your workspace before starting a project - gathering the tools you'll need.
    /// </para>
    /// </remarks>
    protected ModelCompressionBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Compresses the given model weights.
    /// </summary>
    /// <param name="weights">The original model weights to compress.</param>
    /// <returns>A tuple containing the compressed weights and compression metadata.</returns>
    public abstract (Vector<T> compressedWeights, ICompressionMetadata<T> metadata) Compress(Vector<T> weights);

    /// <summary>
    /// Decompresses the compressed weights back to their original form.
    /// </summary>
    /// <param name="compressedWeights">The compressed weights.</param>
    /// <param name="metadata">The metadata needed for decompression.</param>
    /// <returns>The decompressed weights.</returns>
    public abstract Vector<T> Decompress(Vector<T> compressedWeights, ICompressionMetadata<T> metadata);

    /// <summary>
    /// Calculates the compression ratio achieved.
    /// </summary>
    /// <param name="originalSize">The original size in bytes.</param>
    /// <param name="compressedSize">The compressed size in bytes.</param>
    /// <returns>The compression ratio (original size / compressed size).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates how much smaller the compressed model is.
    ///
    /// The formula is simple: compression ratio = original size ÷ compressed size
    ///
    /// Examples:
    /// - Original: 1000 MB, Compressed: 100 MB → Ratio: 10.0 (90% smaller)
    /// - Original: 500 MB, Compressed: 250 MB → Ratio: 2.0 (50% smaller)
    ///
    /// Higher ratios mean better compression!
    /// </para>
    /// </remarks>
    public virtual double CalculateCompressionRatio(long originalSize, long compressedSize)
    {
        if (compressedSize == 0)
        {
            throw new ArgumentException("Compressed size cannot be zero.", nameof(compressedSize));
        }

        return (double)originalSize / compressedSize;
    }

    /// <summary>
    /// Gets the size in bytes of the compressed representation.
    /// </summary>
    /// <param name="compressedWeights">The compressed weights.</param>
    /// <param name="metadata">The compression metadata.</param>
    /// <returns>The total size in bytes.</returns>
    public abstract long GetCompressedSize(Vector<T> compressedWeights, ICompressionMetadata<T> metadata);

    #region Matrix Operations (2D)

    /// <summary>
    /// Compresses a 2D matrix of weights.
    /// Default implementation flattens to vector, compresses, and reshapes back.
    /// </summary>
    /// <param name="weights">The original weight matrix to compress.</param>
    /// <returns>A tuple containing the compressed weights and compression metadata.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method compresses a 2D weight matrix by:
    ///
    /// 1. Flattening the matrix into a 1D vector (row by row)
    /// 2. Applying the vector compression algorithm
    /// 3. Wrapping the result with shape information
    ///
    /// The compressed matrix maintains the original dimensions for convenience,
    /// but the actual size reduction comes from the underlying vector compression.
    /// </para>
    /// </remarks>
    public virtual (Matrix<T> compressedWeights, ICompressionMetadata<T> metadata) CompressMatrix(Matrix<T> weights)
    {
        if (weights == null) throw new ArgumentNullException(nameof(weights));

        // Flatten matrix to vector
        var flatWeights = MatrixToVector(weights);

        // Compress using vector method
        var (compressedVector, vectorMetadata) = Compress(flatWeights);

        // Create matrix metadata including original shape
        var matrixMetadata = new MatrixCompressionMetadata<T>(
            originalRows: weights.Rows,
            originalColumns: weights.Columns,
            innerMetadata: vectorMetadata);

        // Reshape compressed data back to matrix (may be different size)
        var compressedMatrix = VectorToMatrix(compressedVector, weights.Rows, weights.Columns);

        return (compressedMatrix, matrixMetadata);
    }

    /// <summary>
    /// Decompresses the compressed matrix weights back to their original form.
    /// </summary>
    /// <param name="compressedWeights">The compressed weight matrix.</param>
    /// <param name="metadata">The metadata needed for decompression.</param>
    /// <returns>The decompressed weight matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reverses the compression process:
    ///
    /// 1. Flattens the compressed matrix to a vector
    /// 2. Applies the vector decompression algorithm
    /// 3. Reshapes back to the original matrix dimensions
    ///
    /// The metadata must be the same type returned by CompressMatrix.
    /// </para>
    /// </remarks>
    public virtual Matrix<T> DecompressMatrix(Matrix<T> compressedWeights, ICompressionMetadata<T> metadata)
    {
        if (compressedWeights == null) throw new ArgumentNullException(nameof(compressedWeights));
        if (metadata == null) throw new ArgumentNullException(nameof(metadata));

        if (metadata is not MatrixCompressionMetadata<T> matrixMetadata)
        {
            throw new ArgumentException(
                $"Expected {nameof(MatrixCompressionMetadata<T>)} but received {metadata.GetType().Name}.",
                nameof(metadata));
        }

        // Flatten to vector
        var compressedVector = MatrixToVector(compressedWeights);

        // Decompress using vector method
        var decompressedVector = Decompress(compressedVector, matrixMetadata.InnerMetadata);

        // Reshape back to original matrix dimensions
        return VectorToMatrix(decompressedVector, matrixMetadata.OriginalRows, matrixMetadata.OriginalColumns);
    }

    /// <summary>
    /// Gets the size in bytes of the compressed matrix representation.
    /// </summary>
    /// <param name="compressedWeights">The compressed weight matrix.</param>
    /// <param name="metadata">The compression metadata.</param>
    /// <returns>The total size in bytes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates the total storage needed for the compressed matrix,
    /// including both the compressed data and the metadata overhead (shape information).
    /// </para>
    /// </remarks>
    public virtual long GetCompressedSize(Matrix<T> compressedWeights, ICompressionMetadata<T> metadata)
    {
        if (compressedWeights == null) throw new ArgumentNullException(nameof(compressedWeights));
        if (metadata == null) throw new ArgumentNullException(nameof(metadata));

        if (metadata is not MatrixCompressionMetadata<T> matrixMetadata)
        {
            throw new ArgumentException(
                $"Expected {nameof(MatrixCompressionMetadata<T>)} but received {metadata.GetType().Name}.",
                nameof(metadata));
        }

        var compressedVector = MatrixToVector(compressedWeights);
        return GetCompressedSize(compressedVector, matrixMetadata.InnerMetadata) + matrixMetadata.GetMetadataSize();
    }

    #endregion

    #region Tensor Operations (N-D)

    /// <summary>
    /// Compresses an N-dimensional tensor of weights.
    /// Default implementation flattens to vector, compresses, and reshapes back.
    /// </summary>
    /// <param name="weights">The original weight tensor to compress.</param>
    /// <returns>A tuple containing the compressed weights and compression metadata.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method compresses an N-dimensional tensor by:
    ///
    /// 1. Flattening the tensor into a 1D vector
    /// 2. Applying the vector compression algorithm
    /// 3. Wrapping the result with shape information
    ///
    /// Tensors are essential for convolutional layers (4D: [filters, channels, height, width])
    /// and attention mechanisms. This method preserves the shape for reconstruction.
    /// </para>
    /// </remarks>
    public virtual (Tensor<T> compressedWeights, ICompressionMetadata<T> metadata) CompressTensor(Tensor<T> weights)
    {
        if (weights == null) throw new ArgumentNullException(nameof(weights));

        // Flatten tensor to vector
        var flatWeights = TensorToVector(weights);

        // Compress using vector method
        var (compressedVector, vectorMetadata) = Compress(flatWeights);

        // Create tensor metadata including original shape (clone to avoid external modification)
        var originalShape = (int[])weights.Shape.Clone();
        var tensorMetadata = new TensorCompressionMetadata<T>(
            originalShape: originalShape,
            innerMetadata: vectorMetadata);

        // Reshape compressed data back to tensor (same shape, values may differ)
        var compressedTensor = VectorToTensor(compressedVector, originalShape);

        return (compressedTensor, tensorMetadata);
    }

    /// <summary>
    /// Decompresses the compressed tensor weights back to their original form.
    /// </summary>
    /// <param name="compressedWeights">The compressed weight tensor.</param>
    /// <param name="metadata">The metadata needed for decompression.</param>
    /// <returns>The decompressed weight tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reverses the compression process:
    ///
    /// 1. Flattens the compressed tensor to a vector
    /// 2. Applies the vector decompression algorithm
    /// 3. Reshapes back to the original tensor dimensions
    ///
    /// The metadata must be the same type returned by CompressTensor.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> DecompressTensor(Tensor<T> compressedWeights, ICompressionMetadata<T> metadata)
    {
        if (compressedWeights == null) throw new ArgumentNullException(nameof(compressedWeights));
        if (metadata == null) throw new ArgumentNullException(nameof(metadata));

        if (metadata is not TensorCompressionMetadata<T> tensorMetadata)
        {
            throw new ArgumentException(
                $"Expected {nameof(TensorCompressionMetadata<T>)} but received {metadata.GetType().Name}.",
                nameof(metadata));
        }

        // Flatten to vector
        var compressedVector = TensorToVector(compressedWeights);

        // Decompress using vector method
        var decompressedVector = Decompress(compressedVector, tensorMetadata.InnerMetadata);

        // Reshape back to original tensor dimensions
        return VectorToTensor(decompressedVector, tensorMetadata.OriginalShape);
    }

    /// <summary>
    /// Gets the size in bytes of the compressed tensor representation.
    /// </summary>
    /// <param name="compressedWeights">The compressed weight tensor.</param>
    /// <param name="metadata">The compression metadata.</param>
    /// <returns>The total size in bytes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates the total storage needed for the compressed tensor,
    /// including both the compressed data and the metadata overhead (shape information).
    /// </para>
    /// </remarks>
    public virtual long GetCompressedSize(Tensor<T> compressedWeights, ICompressionMetadata<T> metadata)
    {
        if (compressedWeights == null) throw new ArgumentNullException(nameof(compressedWeights));
        if (metadata == null) throw new ArgumentNullException(nameof(metadata));

        if (metadata is not TensorCompressionMetadata<T> tensorMetadata)
        {
            throw new ArgumentException(
                $"Expected {nameof(TensorCompressionMetadata<T>)} but received {metadata.GetType().Name}.",
                nameof(metadata));
        }

        var compressedVector = TensorToVector(compressedWeights);
        return GetCompressedSize(compressedVector, tensorMetadata.InnerMetadata) + tensorMetadata.GetMetadataSize();
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Gets the size in bytes of a value of type T.
    /// </summary>
    /// <returns>The size in bytes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different number types take different amounts of memory.
    ///
    /// Common sizes:
    /// - float (single precision): 4 bytes
    /// - double (double precision): 8 bytes
    ///
    /// This method figures out the size automatically based on the type being used.
    /// </para>
    /// </remarks>
    protected virtual int GetElementSize()
    {
        return typeof(T) == typeof(float) ? 4 :
               typeof(T) == typeof(double) ? 8 :
               System.Runtime.InteropServices.Marshal.SizeOf(typeof(T));
    }

    /// <summary>
    /// Converts a matrix to a flattened vector.
    /// </summary>
    protected Vector<T> MatrixToVector(Matrix<T> matrix)
    {
        var data = new T[matrix.Rows * matrix.Columns];
        int idx = 0;
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                data[idx++] = matrix[i, j];
            }
        }
        return new Vector<T>(data);
    }

    /// <summary>
    /// Converts a vector to a matrix with specified dimensions.
    /// </summary>
    protected Matrix<T> VectorToMatrix(Vector<T> vector, int rows, int cols)
    {
        var matrix = new Matrix<T>(rows, cols);
        int idx = 0;
        int length = Math.Min(vector.Length, rows * cols);
        for (int i = 0; i < rows && idx < length; i++)
        {
            for (int j = 0; j < cols && idx < length; j++)
            {
                matrix[i, j] = vector[idx++];
            }
        }
        return matrix;
    }

    /// <summary>
    /// Converts a tensor to a flattened vector.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method flattens an N-dimensional tensor into a 1D vector,
    /// preserving all values in row-major order. This is the first step in tensor compression -
    /// convert to 1D, compress, then convert back.
    /// </para>
    /// </remarks>
    protected Vector<T> TensorToVector(Tensor<T> tensor)
    {
        return tensor.ToVector();
    }

    /// <summary>
    /// Converts a vector to a tensor with specified shape.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reshapes a 1D vector into an N-dimensional tensor
    /// with the specified shape. The vector values are placed in row-major order into the tensor.
    /// The total number of elements in the vector must match the product of all shape dimensions.
    /// </para>
    /// </remarks>
    protected Tensor<T> VectorToTensor(Vector<T> vector, int[] shape)
    {
        return Tensor<T>.FromVector(vector, shape);
    }

    #endregion
}
