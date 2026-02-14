using AiDotNet.Enums;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for compression metadata that stores information needed to decompress model weights.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// Compression metadata contains the essential information required to reverse the compression process.
/// Different compression algorithms produce different types of metadata - for example, weight clustering
/// stores cluster centers, while Huffman encoding stores the encoding tree.
/// </para>
/// <para><b>For Beginners:</b> When you compress something, you need to remember how you compressed it
/// so you can undo it later. This metadata is like a "recipe" for decompression.
///
/// For example, if you compress weights using clustering:
/// - The compressed data contains which cluster each weight belongs to (just a number like 0, 1, 2...)
/// - The metadata contains the actual cluster center values (like 0.5, 1.2, 3.7...)
/// - To decompress, you look up each cluster number and replace it with the actual value
///
/// Without this metadata, you couldn't restore the original weights from the compressed data.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("CompressionMetadata")]
public interface ICompressionMetadata<T>
{
    /// <summary>
    /// Gets the type of compression algorithm that produced this metadata.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells the system which decompression algorithm to use.
    /// It's like knowing whether a file is a ZIP or RAR - you need the right tool to open it.
    /// </para>
    /// </remarks>
    CompressionType Type { get; }

    /// <summary>
    /// Gets the original length of the uncompressed weight vector.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many weights were in the original model before compression.
    /// It's needed to allocate the right amount of memory when decompressing.
    /// </para>
    /// </remarks>
    int OriginalLength { get; }

    /// <summary>
    /// Gets the size in bytes of this metadata structure.
    /// </summary>
    /// <returns>The metadata size in bytes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The metadata takes up space too! When calculating total compressed size,
    /// you need to include both the compressed weights AND this metadata.
    /// </para>
    /// </remarks>
    long GetMetadataSize();
}
