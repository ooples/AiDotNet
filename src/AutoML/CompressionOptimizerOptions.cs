namespace AiDotNet.AutoML;

/// <summary>
/// Configuration options for the compression optimizer.
/// </summary>
public class CompressionOptimizerOptions
{
    /// <summary>
    /// Gets or sets the maximum number of trials to run (default: 20).
    /// </summary>
    public int MaxTrials { get; set; } = 20;

    /// <summary>
    /// Gets or sets the maximum acceptable accuracy loss as a fraction (default: 0.02 = 2%).
    /// </summary>
    public double MaxAccuracyLoss { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the minimum acceptable compression ratio (default: 2.0).
    /// </summary>
    public double MinCompressionRatio { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the weight for accuracy in fitness calculation (default: 0.5).
    /// </summary>
    public double AccuracyWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the weight for compression ratio in fitness calculation (default: 0.3).
    /// </summary>
    public double CompressionWeight { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the weight for inference speed in fitness calculation (default: 0.2).
    /// </summary>
    public double SpeedWeight { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets whether to include pruning techniques (default: true).
    /// </summary>
    public bool IncludePruning { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include quantization techniques (default: true).
    /// </summary>
    public bool IncludeQuantization { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include encoding techniques (default: true).
    /// </summary>
    public bool IncludeEncoding { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include hybrid techniques like Deep Compression (default: true).
    /// </summary>
    public bool IncludeHybrid { get; set; } = true;

    /// <summary>
    /// Gets or sets the random seed for reproducibility (default: null for random).
    /// </summary>
    public int? RandomSeed { get; set; }
}

