using AiDotNet.LinearAlgebra;

namespace AiDotNet.ModelCompression;

/// <summary>
/// Implements hybrid compression that combines weight clustering with Huffman encoding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Hybrid compression first applies weight clustering to reduce the number of unique values,
/// then uses Huffman encoding on the cluster indices to further compress the data. This two-stage
/// approach can achieve compression ratios of 20-50x or higher while maintaining good accuracy.
/// </para>
/// <para><b>For Beginners:</b> Hybrid compression is like using multiple packing strategies together.
///
/// Imagine packing for a trip:
/// 1. First, you organize similar items together (clustering) - all shirts in one pile, pants in another
/// 2. Then, you compress each pile differently based on how much you have (Huffman) - frequently used items
///    get special treatment for quick access
///
/// For neural networks, hybrid compression combines two powerful techniques:
///
/// Stage 1 - Weight Clustering:
/// - Groups similar weights together
/// - Replaces millions of unique weights with 256 cluster IDs (much smaller!)
/// - Lossy but controlled - you choose the quality/size tradeoff
///
/// Stage 2 - Huffman Encoding:
/// - Notices that some cluster IDs appear more often than others
/// - Gives frequent cluster IDs short codes (like "01")
/// - Gives rare cluster IDs long codes (like "110101")
/// - Lossless - perfectly reversible
///
/// Why combine them?
/// - Clustering alone: Gets you ~4x compression (32-bit float → 8-bit cluster ID)
/// - Huffman alone: Doesn't help much if all values are unique
/// - Together: Clustering creates repeated patterns that Huffman can exploit
///
/// Real-world example:
/// - Original model: 100 million weights × 32 bits = 3.2 GB
/// - After clustering: 100 million cluster IDs × 8 bits = 100 MB (32x compression)
/// - After Huffman: 100 million IDs with variable length ≈ 60 MB (53x total compression!)
/// - Result: 3.2 GB → 60 MB with <2% accuracy loss
/// </para>
/// </remarks>
public class HybridHuffmanClusteringCompression<T> : ModelCompressionBase<T>
{
    private readonly WeightClusteringCompression<T> _clusteringCompression;
    private readonly HuffmanEncodingCompression<T> _huffmanCompression;
    private readonly object _lockObject = new object();

    /// <summary>
    /// Initializes a new instance of the HybridHuffmanClusteringCompression class.
    /// </summary>
    /// <param name="numClusters">Number of clusters for weight clustering (default: 256).</param>
    /// <param name="maxIterations">Maximum K-means iterations (default: 100).</param>
    /// <param name="tolerance">K-means convergence tolerance (default: 1e-6).</param>
    /// <param name="huffmanPrecision">Precision for Huffman encoding (default: 0 for cluster indices).</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> These parameters control both compression stages.
    ///
    /// Clustering parameters:
    /// - numClusters: How many groups to create (256 = 8-bit, very common)
    ///   * More clusters = better quality, less compression
    ///   * Fewer clusters = more compression, lower quality
    ///
    /// - maxIterations/tolerance: How hard to work on finding optimal clusters
    ///   * Defaults are usually fine
    ///
    /// Huffman parameters:
    /// - huffmanPrecision: Set to 0 for cluster indices (they're already integers)
    ///   * Higher values only matter for floating-point data
    ///
    /// The magic happens when these work together:
    /// - Clustering creates patterns
    /// - Huffman exploits those patterns
    /// - Total compression is better than either alone
    /// </para>
    /// </remarks>
    public HybridHuffmanClusteringCompression(
        int numClusters = 256,
        int maxIterations = 100,
        double tolerance = 1e-6,
        int huffmanPrecision = 0,
        int? randomSeed = null)
    {
        _clusteringCompression = new WeightClusteringCompression<T>(
            numClusters, maxIterations, tolerance, randomSeed);
        _huffmanCompression = new HuffmanEncodingCompression<T>(huffmanPrecision);
    }

    /// <summary>
    /// Compresses weights using clustering followed by Huffman encoding.
    /// </summary>
    /// <param name="weights">The original model weights.</param>
    /// <returns>Compressed weights and hybrid metadata.</returns>
    public override (Vector<T> compressedWeights, object metadata) Compress(Vector<T> weights)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        if (weights.Length == 0)
        {
            throw new ArgumentException("Weights cannot be empty.", nameof(weights));
        }

        lock (_lockObject)
        {
            // Stage 1: Apply weight clustering
            var (clusteredWeights, clusteringMetadata) = _clusteringCompression.Compress(weights);

            // Stage 2: Apply Huffman encoding to cluster indices
            var (huffmanWeights, huffmanMetadata) = _huffmanCompression.Compress(clusteredWeights);

            // Combine metadata
            var hybridMetadata = new HybridCompressionMetadata(clusteringMetadata, huffmanMetadata);

            return (huffmanWeights, hybridMetadata);
        }
    }

    /// <summary>
    /// Decompresses weights by reversing Huffman encoding then clustering.
    /// </summary>
    /// <param name="compressedWeights">The compressed weights.</param>
    /// <param name="metadata">The hybrid compression metadata.</param>
    /// <returns>The decompressed weights.</returns>
    public override Vector<T> Decompress(Vector<T> compressedWeights, object metadata)
    {
        if (compressedWeights == null)
        {
            throw new ArgumentNullException(nameof(compressedWeights));
        }

        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        var hybridMetadata = metadata as HybridCompressionMetadata;
        if (hybridMetadata == null)
        {
            throw new ArgumentException("Invalid metadata type for hybrid compression.", nameof(metadata));
        }

        lock (_lockObject)
        {
            // Stage 1: Decompress Huffman encoding to get cluster indices
            var clusterIndices = _huffmanCompression.Decompress(
                compressedWeights, hybridMetadata.HuffmanMetadata);

            // Stage 2: Decompress clustering to get original weights
            var originalWeights = _clusteringCompression.Decompress(
                clusterIndices, hybridMetadata.ClusteringMetadata);

            return originalWeights;
        }
    }

    /// <summary>
    /// Gets the total compressed size from both compression stages.
    /// </summary>
    public override long GetCompressedSize(Vector<T> compressedWeights, object metadata)
    {
        if (compressedWeights == null)
        {
            throw new ArgumentNullException(nameof(compressedWeights));
        }

        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        var hybridMetadata = metadata as HybridCompressionMetadata;
        if (hybridMetadata == null)
        {
            throw new ArgumentException("Invalid metadata type.", nameof(metadata));
        }

        // Size from Huffman encoding (the final compressed representation)
        long huffmanSize = _huffmanCompression.GetCompressedSize(
            compressedWeights, hybridMetadata.HuffmanMetadata);

        // Size from clustering metadata (cluster centers)
        var clusterMetadata = hybridMetadata.ClusteringMetadata as WeightClusteringMetadata<T>;
        if (clusterMetadata != null)
        {
            long clusterCentersSize = clusterMetadata.NumClusters * GetElementSize();
            return huffmanSize + clusterCentersSize;
        }

        return huffmanSize;
    }
}

/// <summary>
/// Metadata for hybrid compression combining clustering and Huffman encoding.
/// </summary>
public class HybridCompressionMetadata
{
    /// <summary>
    /// Initializes a new instance of the HybridCompressionMetadata class.
    /// </summary>
    /// <param name="clusteringMetadata">Metadata from the clustering stage.</param>
    /// <param name="huffmanMetadata">Metadata from the Huffman encoding stage.</param>
    public HybridCompressionMetadata(object clusteringMetadata, object huffmanMetadata)
    {
        if (clusteringMetadata == null)
        {
            throw new ArgumentNullException(nameof(clusteringMetadata));
        }

        if (huffmanMetadata == null)
        {
            throw new ArgumentNullException(nameof(huffmanMetadata));
        }

        ClusteringMetadata = clusteringMetadata;
        HuffmanMetadata = huffmanMetadata;
    }

    /// <summary>
    /// Gets the metadata from the clustering stage.
    /// </summary>
    public object ClusteringMetadata { get; private set; }

    /// <summary>
    /// Gets the metadata from the Huffman encoding stage.
    /// </summary>
    public object HuffmanMetadata { get; private set; }
}
