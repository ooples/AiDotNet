namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for advanced gradient compression methods (PowerSGD, sketching, adaptive).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure state-of-the-art compression methods
/// that can reduce communication bandwidth by 100-1000x compared to uncompressed gradients.
/// Set this on <see cref="FederatedCompressionOptions"/> when using advanced compression.</para>
/// </remarks>
public class AdvancedCompressionOptions
{
    /// <summary>
    /// Gets or sets the advanced compression strategy. Default is PowerSGD.
    /// </summary>
    public AdvancedCompressionStrategy Strategy { get; set; } = AdvancedCompressionStrategy.PowerSGD;

    /// <summary>
    /// Gets or sets the rank for PowerSGD low-rank approximation.
    /// Lower rank = more compression but less accuracy. Typical values: 1-16.
    /// Default is 4.
    /// </summary>
    public int PowerSGDRank { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether PowerSGD uses warm-start (reuses previous round's factors).
    /// Warm-start improves convergence but uses more memory. Default is true.
    /// </summary>
    public bool PowerSGDWarmStart { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of hash functions for Count Sketch compression.
    /// More hashes = less collision error but more compute. Default is 5.
    /// </summary>
    public int SketchDepth { get; set; } = 5;

    /// <summary>
    /// Gets or sets the sketch width (number of buckets per hash function).
    /// Wider sketch = less collision error but less compression. Default is calculated as
    /// model_size / compression_ratio if 0. Default is 0 (auto).
    /// </summary>
    public int SketchWidth { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to use error feedback (residual accumulation) with any compressor.
    /// Error feedback stores compression residuals and adds them to the next round.
    /// This makes biased compressors converge properly. Default is true.
    /// </summary>
    public bool UseErrorFeedback { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of recent rounds used for bandwidth estimation in adaptive mode.
    /// Default is 10.
    /// </summary>
    public int AdaptiveBandwidthWindow { get; set; } = 10;

    /// <summary>
    /// Gets or sets the minimum compression ratio for adaptive mode (floor).
    /// Even fast clients won't send fully uncompressed. Default is 0.01 (1%).
    /// </summary>
    public double AdaptiveMinRatio { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum compression ratio for adaptive mode (ceiling).
    /// Even slow clients send at least this much. Default is 0.5 (50%).
    /// </summary>
    public double AdaptiveMaxRatio { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the number of top-k elements to recover from a sketch.
    /// Only used with GradientSketch strategy. Default is 0 (auto, uses global compression ratio).
    /// </summary>
    public int SketchTopK { get; set; } = 0;

    // --- SignSGD ---

    /// <summary>
    /// Gets or sets the learning rate for SignSGD compression. Default: 0.01.
    /// </summary>
    public double SignSGDLearningRate { get; set; } = 0.01;

    // --- FetchSGD ---

    /// <summary>
    /// Gets or sets the number of top-K heavy hitters to recover from FetchSGD sketches. Default: 100.
    /// </summary>
    public int FetchSGDTopK { get; set; } = 100;

    // --- FedKD ---

    /// <summary>
    /// Gets or sets the knowledge distillation temperature for FedKD compression. Default: 3.0.
    /// </summary>
    public double FedKDTemperature { get; set; } = 3.0;

    // --- FedDT ---

    /// <summary>
    /// Gets or sets the maximum tree depth for FedDT decision-tree compression. Default: 8.
    /// </summary>
    public int FedDTMaxDepth { get; set; } = 8;
}
