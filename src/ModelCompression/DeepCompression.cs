using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.ModelCompression;

/// <summary>
/// Implements the Deep Compression algorithm from Han et al. (2015).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Deep Compression is a three-stage compression pipeline that achieves 35-49x compression
/// on neural networks with minimal accuracy loss. The technique was introduced in:
///
/// Han, S., Mao, H., &amp; Dally, W. J. (2015). "Deep Compression: Compressing Deep Neural
/// Networks with Pruning, Trained Quantization and Huffman Coding." arXiv:1510.00149.
///
/// The three stages are applied sequentially:
/// 1. **Pruning**: Remove weights below a magnitude threshold
/// 2. **Quantization**: Cluster remaining weights using k-means (weight sharing)
/// 3. **Huffman Coding**: Apply entropy coding to the sparse, quantized representation
/// </para>
/// <para><b>For Beginners:</b> Deep Compression is like a three-step recipe for making
/// neural networks much smaller:
///
/// **Step 1 - Pruning (Remove the unimportant)**
/// Think of it like cleaning out your closet. Many neural network weights are tiny
/// and don't really matter. We set these to zero and don't store them at all.
/// This alone can give ~9x compression!
///
/// **Step 2 - Quantization (Group similar values)**
/// After pruning, we group similar weight values together. Instead of storing the
/// exact value 0.4523, 0.4518, 0.4531, we store them all as "cluster #7 = 0.4524".
/// We only need to store which cluster each weight belongs to.
/// This gives another ~4x compression!
///
/// **Step 3 - Huffman Coding (Efficient storage)**
/// Some cluster numbers appear more often than others. We use shorter codes for
/// common values and longer codes for rare values (like Morse code).
/// This gives another ~1.5x compression!
///
/// Combined: 9 × 4 × 1.5 ≈ 35-50x compression!
///
/// Example usage:
/// <code>
/// var deepCompression = new DeepCompression&lt;double&gt;(
///     pruningSparsity: 0.9,    // Remove 90% of weights
///     numClusters: 32,          // 5-bit quantization
///     huffmanPrecision: 4);     // 4 decimal places
///
/// var (compressed, metadata) = deepCompression.Compress(weights);
/// var restored = deepCompression.Decompress(compressed, metadata);
/// </code>
/// </para>
/// </remarks>
public class DeepCompression<T> : ModelCompressionBase<T>
{
    private readonly double _pruningSparsity;
    private readonly double _pruningThreshold;
    private readonly int _numClusters;
    private readonly int _maxClusteringIterations;
    private readonly double _clusteringTolerance;
    private readonly int _huffmanPrecision;
    private readonly int? _randomSeed;
    private readonly bool _enableRetraining;

    // Internal compressors for each stage
    private readonly SparsePruningCompression<T> _pruningCompressor;
    private readonly WeightClusteringCompression<T> _clusteringCompressor;
    private readonly HuffmanEncodingCompression<T> _huffmanCompressor;

    /// <summary>
    /// Initializes a new instance of the DeepCompression class.
    /// </summary>
    /// <param name="pruningSparsity">Target sparsity for pruning stage (default: 0.9 = 90% zeros).</param>
    /// <param name="pruningThreshold">Explicit magnitude threshold (default: 0 = use sparsity target).</param>
    /// <param name="numClusters">Number of clusters for quantization (default: 32 for 5-bit).</param>
    /// <param name="maxClusteringIterations">Maximum k-means iterations (default: 100).</param>
    /// <param name="clusteringTolerance">K-means convergence tolerance (default: 1e-6).</param>
    /// <param name="huffmanPrecision">Decimal precision for Huffman encoding (default: 4).</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    /// <param name="enableRetraining">Whether to enable fine-tuning hints (default: false).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> These parameters let you tune the compression:
    ///
    /// **Pruning parameters:**
    /// - pruningSparsity: What fraction of weights to remove (0.9 = remove 90%)
    /// - pruningThreshold: Alternative way to set pruning (by magnitude instead of percentage)
    ///
    /// **Quantization parameters:**
    /// - numClusters: How many unique weight values to allow
    ///   * 16 clusters = 4-bit (more compression, less accuracy)
    ///   * 32 clusters = 5-bit (Han et al. recommended for conv layers)
    ///   * 256 clusters = 8-bit (less compression, higher accuracy)
    ///
    /// **Huffman parameters:**
    /// - huffmanPrecision: How precisely to encode cluster indices
    ///
    /// **Han et al. recommended settings:**
    /// - Convolutional layers: 8-bit (256 clusters), 65-70% sparsity
    /// - Fully-connected layers: 5-bit (32 clusters), 90-95% sparsity
    /// </para>
    /// </remarks>
    public DeepCompression(
        double pruningSparsity = 0.9,
        double pruningThreshold = 0,
        int numClusters = 32,
        int maxClusteringIterations = 100,
        double clusteringTolerance = 1e-6,
        int huffmanPrecision = 4,
        int? randomSeed = null,
        bool enableRetraining = false)
    {
        if (pruningSparsity < 0 || pruningSparsity > 1)
        {
            throw new ArgumentException("Pruning sparsity must be between 0 and 1.", nameof(pruningSparsity));
        }

        if (pruningThreshold < 0)
        {
            throw new ArgumentException("Pruning threshold cannot be negative.", nameof(pruningThreshold));
        }

        if (numClusters <= 0)
        {
            throw new ArgumentException("Number of clusters must be positive.", nameof(numClusters));
        }

        if (maxClusteringIterations <= 0)
        {
            throw new ArgumentException("Max clustering iterations must be positive.", nameof(maxClusteringIterations));
        }

        if (huffmanPrecision <= 0)
        {
            throw new ArgumentException("Huffman precision must be positive.", nameof(huffmanPrecision));
        }

        _pruningSparsity = pruningSparsity;
        _pruningThreshold = pruningThreshold;
        _numClusters = numClusters;
        _maxClusteringIterations = maxClusteringIterations;
        _clusteringTolerance = clusteringTolerance;
        _huffmanPrecision = huffmanPrecision;
        _randomSeed = randomSeed;
        _enableRetraining = enableRetraining;

        // Initialize the three-stage pipeline
        _pruningCompressor = new SparsePruningCompression<T>(
            sparsityTarget: pruningSparsity,
            minMagnitudeThreshold: pruningThreshold,
            useGlobalThreshold: true);

        _clusteringCompressor = new WeightClusteringCompression<T>(
            numClusters: numClusters,
            maxIterations: maxClusteringIterations,
            tolerance: clusteringTolerance,
            randomSeed: randomSeed);

        _huffmanCompressor = new HuffmanEncodingCompression<T>(
            precision: huffmanPrecision);
    }

    /// <summary>
    /// Creates a DeepCompression instance optimized for convolutional layers.
    /// </summary>
    /// <param name="randomSeed">Optional random seed for reproducibility.</param>
    /// <returns>A DeepCompression instance with Han et al. recommended settings for conv layers.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Convolutional layers are typically more sensitive to compression,
    /// so we use more conservative settings: 8-bit quantization and lower sparsity.
    /// </para>
    /// </remarks>
    public static DeepCompression<T> ForConvolutionalLayers(int? randomSeed = null)
    {
        return new DeepCompression<T>(
            pruningSparsity: 0.65,    // 65% sparsity for conv layers
            numClusters: 256,          // 8-bit quantization
            huffmanPrecision: 4,
            randomSeed: randomSeed);
    }

    /// <summary>
    /// Creates a DeepCompression instance optimized for fully-connected layers.
    /// </summary>
    /// <param name="randomSeed">Optional random seed for reproducibility.</param>
    /// <returns>A DeepCompression instance with Han et al. recommended settings for FC layers.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Fully-connected layers have many redundant weights and can
    /// tolerate more aggressive compression: 5-bit quantization and higher sparsity.
    /// </para>
    /// </remarks>
    public static DeepCompression<T> ForFullyConnectedLayers(int? randomSeed = null)
    {
        return new DeepCompression<T>(
            pruningSparsity: 0.92,    // 92% sparsity for FC layers
            numClusters: 32,          // 5-bit quantization
            huffmanPrecision: 4,
            randomSeed: randomSeed);
    }

    /// <summary>
    /// Compresses weights using the three-stage Deep Compression pipeline.
    /// </summary>
    /// <param name="weights">The original model weights.</param>
    /// <returns>Compressed weights and metadata for all three stages.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies all three compression stages in order:
    /// 1. First, it prunes (removes) small weights
    /// 2. Then, it clusters the remaining weights into groups
    /// 3. Finally, it applies Huffman coding for efficient storage
    ///
    /// The metadata contains everything needed to reverse this process.
    /// </para>
    /// </remarks>
    public override (Vector<T> compressedWeights, ICompressionMetadata<T> metadata) Compress(Vector<T> weights)
    {
        if (weights == null) throw new ArgumentNullException(nameof(weights));

        if (weights.Length == 0)
        {
            throw new ArgumentException("Weights cannot be empty.", nameof(weights));
        }

        // Stage 1: Pruning
        // Remove weights below the magnitude threshold
        var (prunedWeights, pruningMetadata) = _pruningCompressor.Compress(weights);
        var sparseMetadata = (SparsePruningMetadata<T>)pruningMetadata;

        // Stage 2: Quantization (Weight Clustering)
        // Cluster the non-zero weights using k-means
        Vector<T> quantizedWeights;
        ICompressionMetadata<T> clusteringMetadata;

        if (prunedWeights.Length > 0)
        {
            (quantizedWeights, clusteringMetadata) = _clusteringCompressor.Compress(prunedWeights);
        }
        else
        {
            // Handle edge case where all weights are pruned
            quantizedWeights = prunedWeights;
            // Use 1 cluster with a dummy center since constructor requires numClusters > 0
            clusteringMetadata = new WeightClusteringMetadata<T>(
                new T[] { NumOps.Zero }, 1, 0);
        }

        var weightClusteringMetadata = (WeightClusteringMetadata<T>)clusteringMetadata;

        // Stage 3: Huffman Coding
        // Apply entropy coding to the quantized, sparse representation
        Vector<T> huffmanWeights;
        ICompressionMetadata<T> huffmanMetadata;

        if (quantizedWeights.Length > 0)
        {
            (huffmanWeights, huffmanMetadata) = _huffmanCompressor.Compress(quantizedWeights);
        }
        else
        {
            huffmanWeights = quantizedWeights;
            huffmanMetadata = new HuffmanEncodingMetadata<T>(
                new HuffmanNode<T>(default, 0, true, 0, null, null),
                new NumericDictionary<T, string>(),
                1, 0);
        }

        var huffmanEncodingMetadata = (HuffmanEncodingMetadata<T>)huffmanMetadata;

        // Create combined metadata
        var deepCompressionMetadata = new DeepCompressionMetadata<T>(
            pruningMetadata: sparseMetadata,
            clusteringMetadata: weightClusteringMetadata,
            huffmanMetadata: huffmanEncodingMetadata,
            originalLength: weights.Length,
            compressionStats: CalculateCompressionStats(
                weights, huffmanWeights, sparseMetadata, weightClusteringMetadata, huffmanEncodingMetadata));

        return (huffmanWeights, deepCompressionMetadata);
    }

    /// <summary>
    /// Decompresses weights by reversing all three stages.
    /// </summary>
    /// <param name="compressedWeights">The compressed weights.</param>
    /// <param name="metadata">The Deep Compression metadata.</param>
    /// <returns>The decompressed weights.</returns>
    public override Vector<T> Decompress(Vector<T> compressedWeights, ICompressionMetadata<T> metadata)
    {
        if (compressedWeights == null) throw new ArgumentNullException(nameof(compressedWeights));
        if (metadata == null) throw new ArgumentNullException(nameof(metadata));

        if (metadata is not DeepCompressionMetadata<T> deepMetadata)
        {
            throw new ArgumentException(
                $"Expected {nameof(DeepCompressionMetadata<T>)} but received {metadata.GetType().Name}.",
                nameof(metadata));
        }

        // Stage 3 (reverse): Huffman Decoding
        var huffmanDecoded = compressedWeights.Length > 0 && deepMetadata.HuffmanMetadata.OriginalLength > 0
            ? _huffmanCompressor.Decompress(compressedWeights, deepMetadata.HuffmanMetadata)
            : compressedWeights;

        // Stage 2 (reverse): De-quantization
        var dequantized = huffmanDecoded.Length > 0 && deepMetadata.ClusteringMetadata.OriginalLength > 0
            ? _clusteringCompressor.Decompress(huffmanDecoded, deepMetadata.ClusteringMetadata)
            : huffmanDecoded;

        // Stage 1 (reverse): Un-pruning (restore zeros)
        var restored = _pruningCompressor.Decompress(
            dequantized, deepMetadata.PruningMetadata);

        return restored;
    }

    /// <summary>
    /// Gets the total compressed size from all three stages.
    /// </summary>
    public override long GetCompressedSize(Vector<T> compressedWeights, ICompressionMetadata<T> metadata)
    {
        if (compressedWeights == null) throw new ArgumentNullException(nameof(compressedWeights));
        if (metadata == null) throw new ArgumentNullException(nameof(metadata));

        if (metadata is not DeepCompressionMetadata<T> deepMetadata)
        {
            throw new ArgumentException(
                $"Expected {nameof(DeepCompressionMetadata<T>)} but received {metadata.GetType().Name}.",
                nameof(metadata));
        }

        // The final compressed representation size
        long huffmanCompressedSize = compressedWeights.Length > 0
            ? _huffmanCompressor.GetCompressedSize(compressedWeights, deepMetadata.HuffmanMetadata)
            : 0;

        // Add metadata sizes from all stages and the composite metadata overhead
        long pruningMetadataSize = deepMetadata.PruningMetadata.GetMetadataSize();
        long clusteringMetadataSize = deepMetadata.ClusteringMetadata.GetMetadataSize();
        long huffmanMetadataSize = deepMetadata.HuffmanMetadata.GetMetadataSize();
        long deepMetadataOverhead = deepMetadata.GetMetadataSize();

        return huffmanCompressedSize + pruningMetadataSize + clusteringMetadataSize +
               huffmanMetadataSize + deepMetadataOverhead;
    }

    /// <summary>
    /// Calculates compression statistics for the Deep Compression pipeline.
    /// </summary>
    private DeepCompressionStats CalculateCompressionStats(
        Vector<T> original,
        Vector<T> compressed,
        SparsePruningMetadata<T> pruningMetadata,
        WeightClusteringMetadata<T> clusteringMetadata,
        HuffmanEncodingMetadata<T> huffmanMetadata)
    {
        long originalSize = original.Length * GetElementSize();

        // Create a temporary DeepCompressionMetadata with the actual stage metadata
        // to get accurate compressed size calculation
        var tempMetadata = new DeepCompressionMetadata<T>(
            pruningMetadata,
            clusteringMetadata,
            huffmanMetadata,
            original.Length,
            new DeepCompressionStats());

        long compressedSize = GetCompressedSize(compressed, tempMetadata);

        return new DeepCompressionStats
        {
            OriginalSizeBytes = originalSize,
            CompressedSizeBytes = compressedSize,
            CompressionRatio = compressedSize > 0 ? (double)originalSize / compressedSize : 1.0,
            Sparsity = pruningMetadata.ActualSparsity,
            NumClusters = clusteringMetadata.NumClusters,
            BitsPerWeight = clusteringMetadata.NumClusters > 0 ? Math.Log(clusteringMetadata.NumClusters) / Math.Log(2) : 0
        };
    }
}

/// <summary>
/// Metadata for Deep Compression containing information from all three stages.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class DeepCompressionMetadata<T> : ICompressionMetadata<T>
{
    /// <summary>
    /// Initializes a new instance of the DeepCompressionMetadata class.
    /// </summary>
    public DeepCompressionMetadata(
        SparsePruningMetadata<T> pruningMetadata,
        WeightClusteringMetadata<T> clusteringMetadata,
        HuffmanEncodingMetadata<T> huffmanMetadata,
        int originalLength,
        DeepCompressionStats compressionStats)
    {
        Guard.NotNull(pruningMetadata);
        PruningMetadata = pruningMetadata;
        Guard.NotNull(clusteringMetadata);
        ClusteringMetadata = clusteringMetadata;
        Guard.NotNull(huffmanMetadata);
        HuffmanMetadata = huffmanMetadata;
        OriginalLength = originalLength;
        CompressionStats = compressionStats ?? new DeepCompressionStats();
    }

    /// <summary>
    /// Gets the compression type.
    /// </summary>
    public CompressionType Type => CompressionType.DeepCompression;

    /// <summary>
    /// Gets the metadata from Stage 1 (Pruning).
    /// </summary>
    public SparsePruningMetadata<T> PruningMetadata { get; }

    /// <summary>
    /// Gets the metadata from Stage 2 (Weight Clustering/Quantization).
    /// </summary>
    public WeightClusteringMetadata<T> ClusteringMetadata { get; }

    /// <summary>
    /// Gets the metadata from Stage 3 (Huffman Encoding).
    /// </summary>
    public HuffmanEncodingMetadata<T> HuffmanMetadata { get; }

    /// <summary>
    /// Gets the original length of the weights array.
    /// </summary>
    public int OriginalLength { get; }

    /// <summary>
    /// Gets the compression statistics.
    /// </summary>
    public DeepCompressionStats CompressionStats { get; }

    /// <summary>
    /// Gets the size in bytes of this metadata structure.
    /// </summary>
    public long GetMetadataSize()
    {
        return PruningMetadata.GetMetadataSize() +
               ClusteringMetadata.GetMetadataSize() +
               HuffmanMetadata.GetMetadataSize() +
               sizeof(int); // originalLength
    }
}

/// <summary>
/// Statistics about Deep Compression performance.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These statistics help you understand how well compression worked:
/// - CompressionRatio: How much smaller the model is (e.g., 35x means 35 times smaller)
/// - Sparsity: What fraction of weights are zero (e.g., 0.9 = 90% zeros)
/// - BitsPerWeight: How many bits are used per non-zero weight
/// </para>
/// </remarks>
public class DeepCompressionStats
{
    /// <summary>
    /// Original size in bytes before compression.
    /// </summary>
    public long OriginalSizeBytes { get; set; }

    /// <summary>
    /// Compressed size in bytes after all three stages.
    /// </summary>
    public long CompressedSizeBytes { get; set; }

    /// <summary>
    /// Overall compression ratio (original / compressed).
    /// </summary>
    public double CompressionRatio { get; set; } = 1.0;

    /// <summary>
    /// Sparsity achieved by pruning (fraction of zeros).
    /// </summary>
    public double Sparsity { get; set; }

    /// <summary>
    /// Number of clusters used for quantization.
    /// </summary>
    public int NumClusters { get; set; }

    /// <summary>
    /// Effective bits per weight after quantization.
    /// </summary>
    public double BitsPerWeight { get; set; }

    /// <summary>
    /// Compression ratio from pruning stage alone.
    /// </summary>
    public double PruningRatio => Sparsity > 0 ? 1.0 / (1.0 - Sparsity) : 1.0;

    /// <summary>
    /// Compression ratio from quantization stage alone.
    /// </summary>
    public double QuantizationRatio => BitsPerWeight > 0 ? 32.0 / BitsPerWeight : 1.0;
}
