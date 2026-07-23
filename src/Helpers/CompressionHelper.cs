using System.IO.Compression;
using System.Text;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Enums;

namespace AiDotNet.Helpers;

/// <summary>
/// Provides transparent compression and decompression utilities for model serialization.
/// </summary>
/// <remarks>
/// <para>
/// CompressionHelper handles the application of compression during model serialization and
/// decompression during deserialization. It works transparently with the facade pattern,
/// so users never directly interact with compression algorithms.
/// </para>
/// <para><b>For Beginners:</b> This helper automatically compresses your model when you save it
/// and decompresses it when you load it. You don't need to do anything special - just configure
/// compression when building your model, and the rest happens automatically.
///
/// Benefits:
/// - Smaller model files (50-90% size reduction)
/// - Faster model loading over networks
/// - Lower storage costs
/// - Seamless integration with existing serialize/deserialize methods
/// </para>
/// </remarks>
public static class CompressionHelper
{
    /// <summary>
    /// Magic bytes to identify compressed model data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These magic bytes are like a signature at the start of the file.
    /// They tell us "this data was compressed by AiDotNet" so we know to decompress it.
    /// Without this, we wouldn't know if data is compressed or not.
    /// </para>
    /// </remarks>
    private static readonly byte[] MagicBytes = Encoding.ASCII.GetBytes("AIDNC"); // AiDotNet Compressed

    /// <summary>
    /// Current compression format version for forward compatibility.
    /// </summary>
    private const byte FormatVersion = 1;

    /// <summary>
    /// Compresses the serialized model data based on the compression configuration.
    /// </summary>
    /// <param name="data">The uncompressed serialized model data.</param>
    /// <param name="config">The compression configuration.</param>
    /// <returns>The compressed data with header information.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes your model data and makes it smaller.
    /// It adds a small header so we can decompress it later. The header contains:
    /// - Magic bytes (to identify this as compressed data)
    /// - Version number (for future compatibility)
    /// - Compression type (which algorithm was used)
    /// - Original size (so we know how much memory to allocate)
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when data or config is null.</exception>
    public static byte[] Compress(byte[] data, CompressionConfig config)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (config == null)
        {
            throw new ArgumentNullException(nameof(config));
        }

        // If compression is disabled, return original data
        if (config.Mode == ModelCompressionMode.None)
        {
            return data;
        }

        // For automatic mode, choose the best compression strategy
        var effectiveType = config.Mode == ModelCompressionMode.Automatic
            ? ChooseOptimalCompression(data, config)
            : config.Type;

        // Apply compression
        byte[] compressedData = ApplyCompression(data, effectiveType, config);

        // Create header
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write header
        writer.Write(MagicBytes);
        writer.Write(FormatVersion);
        writer.Write((byte)effectiveType);
        writer.Write((byte)config.Mode);
        writer.Write(data.Length); // Original size for decompression
        writer.Write(compressedData.Length);

        // Write compressed data
        writer.Write(compressedData);

        return ms.ToArray();
    }

    /// <summary>
    /// Decompresses model data if it was compressed, otherwise returns unchanged data.
    /// </summary>
    /// <param name="data">The potentially compressed model data.</param>
    /// <returns>The decompressed data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method checks if the data was compressed by us.
    /// If yes, it decompresses it. If no, it returns the data unchanged.
    /// This allows loading both compressed and uncompressed models seamlessly.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when decompression fails.</exception>
    public static byte[] DecompressIfNeeded(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        // Check if data has our magic bytes
        if (!IsCompressedData(data))
        {
            return data;
        }

        return Decompress(data);
    }

    /// <summary>
    /// Checks if the data was compressed by AiDotNet.
    /// </summary>
    /// <param name="data">The data to check.</param>
    /// <returns>True if the data appears to be compressed AiDotNet model data.</returns>
    public static bool IsCompressedData(byte[] data)
    {
        if (data == null || data.Length < MagicBytes.Length + 10) // Header minimum size
        {
            return false;
        }

        for (int i = 0; i < MagicBytes.Length; i++)
        {
            if (data[i] != MagicBytes[i])
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Decompresses the model data.
    /// </summary>
    /// <param name="data">The compressed model data with header.</param>
    /// <returns>The decompressed data.</returns>
    /// <exception cref="InvalidOperationException">Thrown when decompression fails or format is invalid.</exception>
    private static byte[] Decompress(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read and verify magic bytes
        var magic = reader.ReadBytes(MagicBytes.Length);
        for (int i = 0; i < MagicBytes.Length; i++)
        {
            if (magic[i] != MagicBytes[i])
            {
                throw new InvalidOperationException("Invalid compressed data format: magic bytes mismatch.");
            }
        }

        // Read header
        byte version = reader.ReadByte();
        if (version > FormatVersion)
        {
            throw new InvalidOperationException(
                $"Compressed data format version {version} is not supported. Maximum supported version: {FormatVersion}");
        }

        var compressionType = (CompressionType)reader.ReadByte();
        _ = (ModelCompressionMode)reader.ReadByte(); // Read mode but not used for decompression
        int originalSize = reader.ReadInt32();
        int compressedSize = reader.ReadInt32();

        // Read compressed data
        var compressedData = reader.ReadBytes(compressedSize);

        // Decompress
        return ApplyDecompression(compressedData, compressionType, originalSize);
    }

    /// <summary>
    /// Chooses the optimal compression type based on data characteristics.
    /// </summary>
    private static CompressionType ChooseOptimalCompression(byte[] data, CompressionConfig config)
    {
        // For small data, use simple compression
        if (data.Length < 1024)
        {
            return CompressionType.HuffmanEncoding;
        }

        // For medium data, use weight clustering
        if (data.Length < 1024 * 1024) // < 1MB
        {
            return CompressionType.WeightClustering;
        }

        // For large data, use hybrid compression for maximum reduction
        return CompressionType.HybridHuffmanClustering;
    }

    /// <summary>
    /// Applies the specified compression algorithm to the data.
    /// </summary>
    private static byte[] ApplyCompression(byte[] data, CompressionType type, CompressionConfig config)
    {
        switch (type)
        {
            case CompressionType.None:
                return data;

            case CompressionType.HuffmanEncoding:
                return CompressWithDeflate(data, System.IO.Compression.CompressionLevel.Optimal);

            case CompressionType.WeightClustering:
                // For serialized byte data, use GZip which handles byte streams well
                return CompressWithGZip(data, System.IO.Compression.CompressionLevel.Optimal);

            case CompressionType.ProductQuantization:
            case CompressionType.HybridClusteringQuantization:
            case CompressionType.HybridClusteringPruning:
#if NET6_0_OR_GREATER
                // Use Brotli for best compression ratio on modern .NET
                return CompressWithBrotli(data, System.IO.Compression.CompressionLevel.Optimal);
#else
                // Fall back to GZip on .NET Framework
                return CompressWithGZip(data, System.IO.Compression.CompressionLevel.Optimal);
#endif

            case CompressionType.HybridHuffmanClustering:
#if NET6_0_OR_GREATER
                // Use Brotli with maximum compression for large models on modern .NET
                return CompressWithBrotli(data, System.IO.Compression.CompressionLevel.SmallestSize);
#else
                // Fall back to GZip with optimal compression on .NET Framework
                return CompressWithGZip(data, System.IO.Compression.CompressionLevel.Optimal);
#endif

            default:
                return CompressWithGZip(data, System.IO.Compression.CompressionLevel.Optimal);
        }
    }

    /// <summary>
    /// Applies the corresponding decompression algorithm.
    /// </summary>
    private static byte[] ApplyDecompression(byte[] compressedData, CompressionType type, int originalSize)
    {
        switch (type)
        {
            case CompressionType.None:
                return compressedData;

            case CompressionType.HuffmanEncoding:
                return DecompressWithDeflate(compressedData);

            case CompressionType.WeightClustering:
                return DecompressWithGZip(compressedData);

            case CompressionType.ProductQuantization:
            case CompressionType.HybridClusteringQuantization:
            case CompressionType.HybridClusteringPruning:
            case CompressionType.HybridHuffmanClustering:
#if NET6_0_OR_GREATER
                return DecompressWithBrotli(compressedData);
#else
                return DecompressWithGZip(compressedData);
#endif

            default:
                return DecompressWithGZip(compressedData);
        }
    }

    /// <summary>
    /// Compresses data using the Deflate algorithm.
    /// </summary>
    private static byte[] CompressWithDeflate(byte[] data, System.IO.Compression.CompressionLevel level)
    {
        using var outputStream = new MemoryStream();
        using (var deflateStream = new DeflateStream(outputStream, level, leaveOpen: true))
        {
            deflateStream.Write(data, 0, data.Length);
        }
        return outputStream.ToArray();
    }

    /// <summary>
    /// Decompresses data using the Deflate algorithm.
    /// </summary>
    private static byte[] DecompressWithDeflate(byte[] compressedData)
    {
        using var inputStream = new MemoryStream(compressedData);
        using var deflateStream = new DeflateStream(inputStream, CompressionMode.Decompress);
        using var outputStream = new MemoryStream();
        deflateStream.CopyTo(outputStream);
        return outputStream.ToArray();
    }

    /// <summary>
    /// Compresses data using the GZip algorithm.
    /// </summary>
    private static byte[] CompressWithGZip(byte[] data, System.IO.Compression.CompressionLevel level)
    {
        using var outputStream = new MemoryStream();
        using (var gzipStream = new GZipStream(outputStream, level, leaveOpen: true))
        {
            gzipStream.Write(data, 0, data.Length);
        }
        return outputStream.ToArray();
    }

    /// <summary>
    /// Decompresses data using the GZip algorithm.
    /// </summary>
    private static byte[] DecompressWithGZip(byte[] compressedData)
    {
        using var inputStream = new MemoryStream(compressedData);
        using var gzipStream = new GZipStream(inputStream, CompressionMode.Decompress);
        using var outputStream = new MemoryStream();
        gzipStream.CopyTo(outputStream);
        return outputStream.ToArray();
    }

#if NET6_0_OR_GREATER
    /// <summary>
    /// Compresses data using the Brotli algorithm.
    /// </summary>
    /// <remarks>
    /// <para>Brotli is only available on .NET 6.0 and later. It provides better compression
    /// ratios than GZip, especially for text and serialized data.</para>
    /// </remarks>
    private static byte[] CompressWithBrotli(byte[] data, System.IO.Compression.CompressionLevel level)
    {
        using var outputStream = new MemoryStream();
        using (var brotliStream = new BrotliStream(outputStream, level, leaveOpen: true))
        {
            brotliStream.Write(data, 0, data.Length);
        }
        return outputStream.ToArray();
    }

    /// <summary>
    /// Decompresses data using the Brotli algorithm.
    /// </summary>
    private static byte[] DecompressWithBrotli(byte[] compressedData)
    {
        using var inputStream = new MemoryStream(compressedData);
        using var brotliStream = new BrotliStream(inputStream, CompressionMode.Decompress);
        using var outputStream = new MemoryStream();
        brotliStream.CopyTo(outputStream);
        return outputStream.ToArray();
    }
#endif

    /// <summary>
    /// Gets compression statistics for the last compression operation.
    /// </summary>
    /// <param name="originalData">The original uncompressed data.</param>
    /// <param name="compressedData">The compressed data.</param>
    /// <returns>A tuple containing (original size, compressed size, compression ratio, space saved percentage).</returns>
    public static (long originalSize, long compressedSize, double ratio, double savedPercent) GetCompressionStats(
        byte[] originalData, byte[] compressedData)
    {
        if (originalData == null || compressedData == null)
        {
            return (0, 0, 1.0, 0.0);
        }

        long originalSize = originalData.Length;
        long compressedSize = compressedData.Length;
        double ratio = compressedSize > 0 ? (double)originalSize / compressedSize : 1.0;
        double savedPercent = originalSize > 0 ? (1.0 - (double)compressedSize / originalSize) * 100.0 : 0.0;

        return (originalSize, compressedSize, ratio, savedPercent);
    }
}
