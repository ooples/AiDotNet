using AiDotNet.Enums;

namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for model compression - reducing model size while preserving accuracy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Model compression makes your trained AI model smaller and faster to load.
/// Think of it like compressing a ZIP file - you get a smaller file that can be restored to its original form.
///
/// Why use compression?
/// - Smaller model files (50-90% size reduction)
/// - Faster model loading and deployment
/// - Lower storage and bandwidth costs
/// - Enables deployment on resource-constrained devices
///
/// Trade-offs:
/// - Some compression types are lossy (slight accuracy reduction, typically 1-2%)
/// - Compression/decompression adds a small processing overhead
///
/// Compression happens automatically when you save (serialize) a model and
/// decompression happens automatically when you load (deserialize) it.
/// You never need to handle compression manually.
///
/// Example:
/// <code>
/// // Use automatic compression (recommended for most cases)
/// var result = await builder
///     .ConfigureModel(model)
///     .ConfigureCompression()
///     .BuildAsync();
///
/// // Or customize compression settings
/// var result = await builder
///     .ConfigureCompression(new CompressionConfig
///     {
///         Mode = CompressionMode.Full,
///         Type = CompressionType.HybridHuffmanClustering,
///         NumClusters = 256
///     })
///     .BuildAsync();
/// </code>
/// </para>
/// </remarks>
public class CompressionConfig
{
    /// <summary>
    /// Gets or sets the compression mode (default: Automatic).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Choose how compression is applied:
    /// - None: No compression (full size, maximum accuracy)
    /// - Automatic: System chooses best approach (recommended)
    /// - WeightsOnly: Compress only model weights
    /// - Full: Compress entire serialized model
    /// </para>
    /// </remarks>
    public ModelCompressionMode Mode { get; set; } = ModelCompressionMode.Automatic;

    /// <summary>
    /// Gets or sets the compression algorithm type (default: WeightClustering).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different algorithms offer different trade-offs:
    /// - WeightClustering: Groups similar weights (good balance of speed and compression)
    /// - HuffmanEncoding: Lossless variable-length encoding (no accuracy loss)
    /// - HybridHuffmanClustering: Combines both for maximum compression
    /// </para>
    /// </remarks>
    public CompressionType Type { get; set; } = CompressionType.WeightClustering;

    /// <summary>
    /// Gets or sets the number of clusters for weight clustering (default: 256).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like choosing how many "bins" to sort weights into.
    /// 256 clusters is the industry standard (equivalent to 8-bit quantization).
    /// More clusters = higher accuracy but less compression.
    /// Fewer clusters = more compression but lower accuracy.
    ///
    /// Common values:
    /// - 16: Aggressive compression (4-bit equivalent)
    /// - 256: Standard compression (8-bit equivalent, recommended)
    /// - 65536: Light compression (16-bit equivalent)
    /// </para>
    /// </remarks>
    public int NumClusters { get; set; } = 256;

    /// <summary>
    /// Gets or sets the decimal precision for Huffman encoding (default: 4).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how many decimal places to keep when rounding weights
    /// for Huffman encoding. Higher precision = better accuracy but less compression.
    /// 4 decimal places is a good default for most models.
    /// </para>
    /// </remarks>
    public int Precision { get; set; } = 4;

    /// <summary>
    /// Gets or sets the convergence tolerance for clustering algorithms (default: 1e-6).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This determines when the clustering algorithm stops iterating.
    /// Smaller values = more precise clusters but slower compression.
    /// The default (0.000001) works well for most cases.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum iterations for clustering algorithms (default: 100).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Limits how long the clustering algorithm runs.
    /// More iterations can improve cluster quality but takes longer.
    /// 100 iterations is sufficient for most models.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the random seed for reproducible compression (default: null for random).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set this to a specific number if you want compression
    /// to produce identical results every time. Useful for testing and debugging.
    /// Leave as null for normal usage.
    /// </para>
    /// </remarks>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets the maximum acceptable accuracy loss percentage (default: 2.0).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If compression would cause more than this percentage
    /// of accuracy loss, the system will warn you or use a less aggressive compression.
    /// 2% is acceptable for most applications. Set to 0 for lossless compression only.
    /// </para>
    /// </remarks>
    public double MaxAccuracyLossPercent { get; set; } = 2.0;
}
