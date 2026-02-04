using AiDotNet.Enums;

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
/// - **FP8 (New)**: 8-bit formats for 2x throughput on H100+ GPUs
/// </para>
/// </remarks>
public class MixedPrecisionConfig
{
    /// <summary>
    /// The precision type to use for mixed-precision training (default: FP16).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This determines which floating-point format to use:</para>
    /// <list type="bullet">
    /// <item><description><b>FP16:</b> Works on most GPUs (GTX 10 series and newer)</description></item>
    /// <item><description><b>BF16:</b> Better stability on Ampere+ GPUs (RTX 30 series, A100)</description></item>
    /// <item><description><b>FP8_Hybrid:</b> Best throughput on H100/H200 GPUs</description></item>
    /// </list>
    /// </remarks>
    public MixedPrecisionType PrecisionType { get; set; } = MixedPrecisionType.FP16;
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

    #region FP8-Specific Settings

    /// <summary>
    /// Format to use for forward pass in FP8 mode (default: E4M3).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> E4M3 has higher precision, better for weights and activations.
    /// Only used when PrecisionType is FP8_E4M3, FP8_E5M2, or FP8_Hybrid.
    /// </para>
    /// </remarks>
    public MixedPrecisionType FP8ForwardFormat { get; set; } = MixedPrecisionType.FP8_E4M3;

    /// <summary>
    /// Format to use for backward pass in FP8 mode (default: E5M2).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> E5M2 has larger range, better for gradients.
    /// Only used when PrecisionType is FP8_E4M3, FP8_E5M2, or FP8_Hybrid.
    /// </para>
    /// </remarks>
    public MixedPrecisionType FP8BackwardFormat { get; set; } = MixedPrecisionType.FP8_E5M2;

    /// <summary>
    /// Whether to use per-tensor scaling for FP8 (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Per-tensor scaling computes optimal scale factors for each tensor,
    /// which can improve accuracy at the cost of some overhead. Recommended for most models.
    /// </para>
    /// </remarks>
    public bool FP8PerTensorScaling { get; set; } = true;

    /// <summary>
    /// Layers to keep in higher precision (FP16/BF16) even when using FP8.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some layers are numerically sensitive and should stay in higher precision.
    /// This list contains layer name patterns (e.g., "LayerNorm", "Softmax") to keep in FP16/BF16.
    /// </para>
    /// </remarks>
    public List<string> FP8ExcludedLayers { get; set; } = new() { "LayerNorm", "BatchNorm", "Softmax", "Embedding" };

    #endregion

    /// <summary>
    /// Creates a configuration with default recommended settings.
    /// </summary>
    public MixedPrecisionConfig()
    {
    }

    #region Factory Methods

    /// <summary>
    /// Creates a conservative configuration optimized for training stability.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this if you're seeing NaN losses or unstable training.
    /// It uses lower initial scale and more conservative growth settings.
    /// </para>
    /// </remarks>
    public static MixedPrecisionConfig Conservative() => new()
    {
        InitialLossScale = 4096.0,
        ScaleGrowthInterval = 4000,
        ScaleGrowthFactor = 1.5,
        ScaleBackoffFactor = 0.25,
        Fp32BatchNorm = true,
        Fp32Loss = true,
        Fp32GradientAccumulation = true
    };

    /// <summary>
    /// Creates an aggressive configuration optimized for maximum throughput.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this on well-behaved models when you want maximum speed.
    /// May cause some instability on models with large gradient variations.
    /// </para>
    /// </remarks>
    public static MixedPrecisionConfig Aggressive() => new()
    {
        InitialLossScale = 131072.0, // 2^17
        ScaleGrowthInterval = 1000,
        ScaleGrowthFactor = 2.0,
        ScaleBackoffFactor = 0.5,
        Fp32BatchNorm = false, // Keep in FP16 for speed
        Fp32Loss = true,
        Fp32GradientAccumulation = false // Accumulate in FP16
    };

    /// <summary>
    /// Creates a configuration for BF16 precision (Ampere+ GPUs).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> BF16 has the same range as FP32 but lower precision.
    /// It's more stable than FP16 and often doesn't need loss scaling.
    /// Requires RTX 30 series, A100, or newer GPU.
    /// </para>
    /// </remarks>
    public static MixedPrecisionConfig ForBF16() => new()
    {
        PrecisionType = MixedPrecisionType.BF16,
        EnableDynamicScaling = false, // BF16 rarely needs loss scaling
        InitialLossScale = 1.0, // No scaling
        Fp32BatchNorm = false, // BF16 is stable enough
        Fp32Loss = true,
        Fp32GradientAccumulation = false
    };

    /// <summary>
    /// Creates a configuration for FP8 hybrid mode (H100+ GPUs).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> FP8 provides 2x throughput compared to FP16 on H100/H200 GPUs.
    /// Uses E4M3 for forward pass (weights/activations) and E5M2 for backward pass (gradients).
    /// This is the NVIDIA-recommended configuration.
    /// </para>
    /// <para><b>Hardware Requirement:</b> NVIDIA H100, H200, or newer GPU with FP8 Tensor Cores.</para>
    /// </remarks>
    public static MixedPrecisionConfig ForFP8() => new()
    {
        PrecisionType = MixedPrecisionType.FP8_Hybrid,
        FP8ForwardFormat = MixedPrecisionType.FP8_E4M3,
        FP8BackwardFormat = MixedPrecisionType.FP8_E5M2,
        FP8PerTensorScaling = true,
        EnableDynamicScaling = true,
        InitialLossScale = 65536.0,
        ScaleGrowthInterval = 1000, // FP8 needs faster adaptation
        Fp32BatchNorm = true, // Keep normalization layers in FP32
        Fp32Loss = true,
        Fp32GradientAccumulation = true,
        FP8ExcludedLayers = new() { "LayerNorm", "BatchNorm", "Softmax", "Embedding", "RMSNorm" }
    };

    /// <summary>
    /// Creates a configuration for FP8 training on transformer models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Optimized for large language models and other transformers.
    /// Keeps attention softmax and layer normalizations in higher precision for stability.
    /// </para>
    /// </remarks>
    public static MixedPrecisionConfig ForFP8Transformers() => new()
    {
        PrecisionType = MixedPrecisionType.FP8_Hybrid,
        FP8ForwardFormat = MixedPrecisionType.FP8_E4M3,
        FP8BackwardFormat = MixedPrecisionType.FP8_E5M2,
        FP8PerTensorScaling = true,
        EnableDynamicScaling = true,
        InitialLossScale = 32768.0, // Slightly conservative for transformers
        ScaleGrowthInterval = 500,
        Fp32BatchNorm = true,
        Fp32Loss = true,
        Fp32GradientAccumulation = true,
        FP8ExcludedLayers = new()
        {
            "LayerNorm", "RMSNorm", "BatchNorm",  // Normalizations
            "Softmax", "Attention",                // Attention components
            "Embedding", "LMHead"                  // Input/output layers
        }
    };

    #endregion

    /// <summary>
    /// Gets a summary of the configuration.
    /// </summary>
    /// <returns>A string describing the configuration.</returns>
    public override string ToString()
    {
        return $"MixedPrecisionConfig: " +
               $"Type={PrecisionType}, " +
               $"InitialScale={InitialLossScale:F0}, " +
               $"Dynamic={EnableDynamicScaling}, " +
               $"GrowthInterval={ScaleGrowthInterval}, " +
               $"GrowthFactor={ScaleGrowthFactor:F1}x, " +
               $"BackoffFactor={ScaleBackoffFactor:F2}x, " +
               $"Range=[{MinLossScale:F0}, {MaxLossScale:F0}]";
    }
}
