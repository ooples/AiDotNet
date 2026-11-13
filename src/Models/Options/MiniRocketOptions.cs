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
    /// Gets or sets the most dilations each kernel is applied at.
    /// </summary>
    /// <value>The most distinct dilations each kernel is applied at. Defaults to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> MiniRocket spreads each kernel's features over exponentially spaced
    /// dilations (how far apart the kernel's taps are), up to this many distinct dilations; the features per
    /// kernel are shared out among them, and each feature gets its own bias from a quantile of a training
    /// example's convolution output.</para>
    /// <para>Default: 32, the paper's <c>max_dilations_per_kernel</c> (Dempster et al. 2021).</para>
    /// </remarks>
    public int MaxDilationsPerKernel { get; set; } = 32;

}
