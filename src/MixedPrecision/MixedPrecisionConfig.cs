namespace AiDotNet.MixedPrecision;

/// <summary>
/// Configuration settings for mixed-precision training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class contains all the settings you can adjust for mixed-precision training.
/// The default values work well for most models, but you can customize them based on your specific needs.
///
/// Key concepts:
/// - **Loss Scaling**: Prevents small gradients from becoming zero in FP16
/// - **Dynamic Scaling**: Automatically adjusts the loss scale during training
/// - **Master Weights**: FP32 copy of parameters for precise updates
/// - **Working Weights**: FP16 copy used for forward/backward passes
/// </para>
/// </remarks>
public class MixedPrecisionConfig
{
    /// <summary>
    /// Initial loss scale factor (default: 65536 = 2^16).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the starting value for loss scaling.
    /// 2^16 = 65536 works well for most models. If you see NaN early in training, try a smaller value like 2^12 = 4096.
    /// </para>
    /// </remarks>
    public double InitialLossScale { get; set; } = 65536.0;

    /// <summary>
    /// Enable dynamic loss scaling (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dynamic scaling automatically adjusts the loss scale during training.
    /// This is generally recommended as it adapts to your model's gradient magnitudes.
    /// Set to false only if you want to manually control the scale.
    /// </para>
    /// </remarks>
    public bool EnableDynamicScaling { get; set; } = true;

    /// <summary>
    /// Number of consecutive successful updates before increasing scale (default: 2000).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After this many updates without overflow, the scale will increase.
    /// Higher values = more conservative (slower to increase scale).
    /// Lower values = more aggressive (faster to increase scale, but more likely to overflow).
    /// </para>
    /// </remarks>
    public int ScaleGrowthInterval { get; set; } = 2000;

    /// <summary>
    /// Factor by which to multiply scale when increasing (default: 2.0).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When increasing the scale, multiply it by this factor.
    /// 2.0 means double the scale. Values between 1.5 and 2.0 are typical.
    /// </para>
    /// </remarks>
    public double ScaleGrowthFactor { get; set; } = 2.0;

    /// <summary>
    /// Factor by which to multiply scale when decreasing (default: 0.5).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When overflow is detected, multiply scale by this factor.
    /// 0.5 means halve the scale. Values between 0.25 and 0.5 are typical.
    /// </para>
    /// </remarks>
    public double ScaleBackoffFactor { get; set; } = 0.5;

    /// <summary>
    /// Minimum allowed loss scale (default: 1.0).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The scale will never go below this value.
    /// 1.0 means no scaling (equivalent to regular FP32 training).
    /// </para>
    /// </remarks>
    public double MinLossScale { get; set; } = 1.0;

    /// <summary>
    /// Maximum allowed loss scale (default: 16777216 = 2^24).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The scale will never go above this value.
    /// 2^24 is a safe upper bound that prevents excessive scaling.
    /// </para>
    /// </remarks>
    public double MaxLossScale { get; set; } = 16777216.0;

    /// <summary>
    /// Whether to keep batch normalization layers in FP32 (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Batch normalization can be numerically unstable in FP16.
    /// Keeping it in FP32 improves training stability with minimal performance impact.
    /// This is recommended for most models.
    /// </para>
    /// </remarks>
    public bool Fp32BatchNorm { get; set; } = true;

    /// <summary>
    /// Whether to keep loss computation in FP32 (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Computing the loss in FP32 improves numerical accuracy
    /// and stability. This is recommended for most models.
    /// </para>
    /// </remarks>
    public bool Fp32Loss { get; set; } = true;

    /// <summary>
    /// Whether to accumulate gradients in FP32 (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Accumulating gradients in FP32 prevents precision loss
    /// when adding many small gradient values. This is essential for mixed-precision training.
    /// </para>
    /// </remarks>
    public bool Fp32GradientAccumulation { get; set; } = true;

    /// <summary>
    /// Creates a configuration with default recommended settings.
    /// </summary>
    public MixedPrecisionConfig()
    {
    }

    /// <summary>
    /// Gets a summary of the configuration.
    /// </summary>
    /// <returns>A string describing the configuration.</returns>
    public override string ToString()
    {
        return $"MixedPrecisionConfig: " +
               $"InitialScale={InitialLossScale:F0}, " +
               $"Dynamic={EnableDynamicScaling}, " +
               $"GrowthInterval={ScaleGrowthInterval}, " +
               $"GrowthFactor={ScaleGrowthFactor:F1}x, " +
               $"BackoffFactor={ScaleBackoffFactor:F2}x, " +
               $"Range=[{MinLossScale:F0}, {MaxLossScale:F0}]";
    }
}
