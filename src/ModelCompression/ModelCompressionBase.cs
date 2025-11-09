using AiDotNet.Interfaces;
using AiDotNet.Helpers;
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
    public abstract (Vector<T> compressedWeights, object metadata) Compress(Vector<T> weights);

    /// <summary>
    /// Decompresses the compressed weights back to their original form.
    /// </summary>
    /// <param name="compressedWeights">The compressed weights.</param>
    /// <param name="metadata">The metadata needed for decompression.</param>
    /// <returns>The decompressed weights.</returns>
    public abstract Vector<T> Decompress(Vector<T> compressedWeights, object metadata);

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
    public abstract long GetCompressedSize(Vector<T> compressedWeights, object metadata);

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
}
