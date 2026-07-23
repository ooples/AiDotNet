using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a type-safe interface for model compression used to reduce model size while preserving accuracy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TMetadata">The strongly-typed metadata class for this compression algorithm.</typeparam>
/// <remarks>
/// <para>
/// This interface provides type-safe model compression by using strongly-typed metadata instead of
/// the object type. Each compression algorithm defines its own metadata class that implements
/// <see cref="ICompressionMetadata{T}"/>, ensuring compile-time type safety.
/// </para>
/// <para><b>For Beginners:</b> Model compression makes AI models smaller and faster without significantly
/// hurting their performance.
///
/// Think of it like compressing a video file - you want to reduce the file size so it's easier to
/// store and share, but you still want the video to look good. Similarly, model compression reduces
/// the memory needed to store an AI model and can make it run faster, which is especially important
/// for deploying models on mobile devices or in the cloud where storage and processing costs matter.
///
/// Different compression strategies work in different ways:
/// - Some group similar values together (clustering)
/// - Some remove less important parts (pruning)
/// - Some use clever encoding schemes to store data more efficiently (quantization, Huffman coding)
///
/// This interface defines the standard methods that all compression implementations must provide.
/// The TMetadata type parameter ensures that each compression algorithm uses its own specific
/// metadata type, preventing errors from using the wrong metadata with the wrong algorithm.
/// </para>
/// </remarks>
public interface IModelCompression<T, TMetadata> where TMetadata : ICompressionMetadata<T>
{
    /// <summary>
    /// Compresses the given model weights.
    /// </summary>
    /// <param name="weights">The original model weights to compress.</param>
    /// <returns>A tuple containing the compressed weights and strongly-typed compression metadata.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes the original weights from your AI model and compresses them
    /// to use less memory. The weights are the learned parameters that make your model work - they're like
    /// the knowledge the model has gained during training.
    ///
    /// The method returns two things:
    /// 1. The compressed weights (smaller in size)
    /// 2. Metadata about the compression (information needed to decompress later)
    ///
    /// For example, if you have 1 million weight values, compression might reduce them to 100,000 values
    /// plus some additional information about how to reconstruct the original values when needed.
    /// </para>
    /// </remarks>
    (Vector<T> compressedWeights, TMetadata metadata) Compress(Vector<T> weights);

    /// <summary>
    /// Decompresses the compressed weights back to their original form.
    /// </summary>
    /// <param name="compressedWeights">The compressed weights.</param>
    /// <param name="metadata">The strongly-typed metadata needed for decompression.</param>
    /// <returns>The decompressed weights.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reverses the compression process, reconstructing the original
    /// (or very close to original) weights from the compressed version.
    ///
    /// Think of it like unzipping a ZIP file - you take the compressed data and the instructions for
    /// how it was compressed (metadata), and produce the usable weights that the model can work with.
    ///
    /// Some compression methods are "lossy" (you don't get exactly the original values back, like JPEG
    /// image compression), while others are "lossless" (you get exact values back, like ZIP compression).
    ///
    /// The metadata parameter contains the information needed to reverse the compression, such as:
    /// - Cluster centers (for weight clustering)
    /// - Huffman trees (for Huffman encoding)
    /// - Scaling factors (for quantization)
    /// </para>
    /// </remarks>
    Vector<T> Decompress(Vector<T> compressedWeights, TMetadata metadata);

    /// <summary>
    /// Calculates the compression ratio achieved.
    /// </summary>
    /// <param name="originalSize">The original size in bytes.</param>
    /// <param name="compressedSize">The compressed size in bytes.</param>
    /// <returns>The compression ratio (original size / compressed size).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The compression ratio tells you how much smaller the compressed model is
    /// compared to the original. It's calculated as: original size ÷ compressed size.
    ///
    /// For example:
    /// - A ratio of 2.0 means the compressed model is half the size (50% reduction)
    /// - A ratio of 10.0 means it's one-tenth the size (90% reduction)
    /// - A ratio of 50.0 means it's one-fiftieth the size (98% reduction)
    ///
    /// Higher compression ratios are better (more compression), but you need to balance this with
    /// accuracy - extreme compression might hurt model performance. The goal is to find the sweet
    /// spot where you get significant size reduction without losing too much accuracy.
    /// </para>
    /// </remarks>
    double CalculateCompressionRatio(long originalSize, long compressedSize);

    /// <summary>
    /// Gets the size in bytes of the compressed representation.
    /// </summary>
    /// <param name="compressedWeights">The compressed weights.</param>
    /// <param name="metadata">The strongly-typed compression metadata.</param>
    /// <returns>The total size in bytes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates the total memory required to store the compressed model,
    /// including both the compressed weights and any metadata needed for decompression.
    ///
    /// It's important to include the metadata size because some compression schemes save space on weights
    /// but require substantial metadata. For example, Huffman encoding needs to store the encoding tree.
    ///
    /// The total size = compressed weights size + metadata size
    ///
    /// This gives you an accurate picture of the actual memory savings you'll achieve.
    /// </para>
    /// </remarks>
    long GetCompressedSize(Vector<T> compressedWeights, TMetadata metadata);
}
