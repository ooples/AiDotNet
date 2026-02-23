namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the MiniRocket time series classifier.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MiniRocket is an optimized version of ROCKET that uses
/// deterministic kernels instead of random ones. This makes it faster and more reproducible
/// while maintaining similar accuracy.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MiniRocketOptions<T> : TimeSeriesClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of features to extract.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> MiniRocket extracts PPV (proportion of positive values) features
    /// from multiple dilations. This controls the total number of output features.</para>
    /// <para>Default: 9996 (84 kernels × 119 dilations per kernel on average)</para>
    /// </remarks>
    public int NumFeatures { get; set; } = 9996;

    /// <summary>
    /// Gets or sets whether to use bias terms in the kernel computation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Bias terms help capture patterns at different thresholds.
    /// MiniRocket uses quantile-based biases from the training data.</para>
    /// <para>Default: true</para>
    /// </remarks>
    public bool UseBias { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of bias values to use per dilation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More biases capture more thresholds but increase
    /// computation time and feature dimensionality.</para>
    /// <para>Default: 9 (percentiles: 0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100)</para>
    /// </remarks>
    public int NumBiasesPerDilation { get; set; } = 9;

    /// <summary>
    /// Gets or sets the random seed for reproducible results.
    /// </summary>
    /// <remarks>
    /// <para>Default: null (non-deterministic)</para>
    /// </remarks>
    public int? RandomSeed { get; set; }
}
