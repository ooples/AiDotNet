using AiDotNet.Enums;
using AiDotNet.Deployment.Optimization.Quantization;

namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for model quantization - compressing models by using lower precision numbers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Quantization makes your AI model smaller and faster by using smaller numbers.
/// Think of it like compressing a high-quality photo - it takes less space but might lose a little quality.</para>
///
/// <para><b>Why use quantization?</b></para>
/// <list type="bullet">
/// <item><description>Smaller model size (50-75% reduction)</description></item>
/// <item><description>Faster inference (2-4x speedup)</description></item>
/// <item><description>Lower memory usage</description></item>
/// <item><description>Enables deployment on mobile/edge devices</description></item>
/// </list>
///
/// <para><b>Quick Start Examples:</b></para>
/// <code>
/// // Simple INT8 quantization (4x compression)
/// config.Mode = QuantizationMode.Int8;
///
/// // High-quality 4-bit with GPTQ (8x compression)
/// config.Mode = QuantizationMode.Int8;
/// config.TargetBitWidth = 4;
/// config.Strategy = QuantizationStrategy.GPTQ;
/// config.Granularity = QuantizationGranularity.PerGroup;
///
/// // Quantization-Aware Training for best accuracy
/// config.UseQuantizationAwareTraining = true;
/// config.QATMethod = QATMethod.EfficientQAT;
/// </code>
/// </remarks>
public class QuantizationConfig
{
    /// <summary>
    /// Gets or sets the quantization mode (default: None).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Choose which type of quantization to use:</para>
    /// <list type="bullet">
    /// <item><description>None: Full precision, no compression</description></item>
    /// <item><description>Float16: Half precision, good balance</description></item>
    /// <item><description>Int8: Maximum compression, slight accuracy loss</description></item>
    /// </list>
    /// </remarks>
    public QuantizationMode Mode { get; set; } = QuantizationMode.None;

    /// <summary>
    /// Gets or sets the quantization strategy (algorithm) to use.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different algorithms for compression:</para>
    /// <list type="bullet">
    /// <item><description><b>MinMax:</b> Simple and fast, good baseline</description></item>
    /// <item><description><b>GPTQ:</b> Best for 3-4 bit, uses Hessian information</description></item>
    /// <item><description><b>AWQ:</b> Best for very large models (70B+)</description></item>
    /// <item><description><b>SmoothQuant:</b> Best when quantizing both weights AND activations</description></item>
    /// </list>
    /// </remarks>
    public QuantizationStrategy Strategy { get; set; } = QuantizationStrategy.Dynamic;

    /// <summary>
    /// Gets or sets the quantization granularity (where to apply scaling factors).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Finer granularity = better accuracy but more overhead:</para>
    /// <list type="bullet">
    /// <item><description><b>PerTensor:</b> One scale for entire layer (fast, less accurate)</description></item>
    /// <item><description><b>PerChannel:</b> One scale per output channel (balanced)</description></item>
    /// <item><description><b>PerGroup:</b> One scale per N elements (most accurate, used by GPTQ/AWQ)</description></item>
    /// </list>
    /// </remarks>
    public QuantizationGranularity Granularity { get; set; } = QuantizationGranularity.PerChannel;

    /// <summary>
    /// Gets or sets the group size for per-group quantization (default: 128).
    /// Only used when Granularity is PerGroup or PerBlock.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Smaller groups = more accuracy but more storage overhead.</para>
    /// <para><b>Typical values:</b> 32, 64, 128 (default), 256</para>
    /// </remarks>
    public int GroupSize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the target bit width for weight quantization.
    /// If null, uses the default bit width for the Mode (8 for Int8, 16 for Float16).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Override the default bit width. For example, use INT8 mode
    /// but target 4-bit weights for more aggressive compression.</para>
    /// <para><b>Common values:</b> 2, 3, 4, 8, 16</para>
    /// </remarks>
    public int? TargetBitWidth { get; set; }

    /// <summary>
    /// Gets or sets the calibration method used to determine optimal scaling factors (default: MinMax).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Calibration determines how to convert large numbers to small numbers.
    /// Only used for Int8 quantization. MinMax is fast and works well for most cases.</para>
    /// </remarks>
    public CalibrationMethod CalibrationMethod { get; set; } = CalibrationMethod.MinMax;

    /// <summary>
    /// Gets or sets whether to use symmetric quantization (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Symmetric quantization treats positive and negative values the same way.
    /// It's faster but asymmetric may be slightly more accurate. Use true for most cases.</para>
    /// </remarks>
    public bool UseSymmetricQuantization { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of calibration samples to use (default: 100).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More samples give better calibration but take longer.
    /// 100 is a good default. Use 1000+ for critical applications.</para>
    /// </remarks>
    public int CalibrationSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to quantize only weights or both weights and activations (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> False means only compress the model parameters (weights).
    /// True means also compress the intermediate values during inference (activations).
    /// Activations give better compression but require calibration data.</para>
    /// </remarks>
    public bool QuantizeActivations { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use Quantization-Aware Training (QAT) instead of Post-Training Quantization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> QAT simulates quantization DURING training so the model
    /// learns to be robust to low precision. Results in better accuracy than PTQ (5-15% improvement).</para>
    /// <para><b>Trade-off:</b> Requires retraining but achieves 95-99% of original accuracy</para>
    /// </remarks>
    public bool UseQuantizationAwareTraining { get; set; } = false;

    /// <summary>
    /// Gets or sets the QAT method to use when UseQuantizationAwareTraining is true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different QAT algorithms with different trade-offs:</para>
    /// <list type="bullet">
    /// <item><description><b>Standard:</b> Basic QAT with Straight-Through Estimator</description></item>
    /// <item><description><b>EfficientQAT:</b> Memory-efficient, good for large models</description></item>
    /// <item><description><b>ZeroQAT:</b> Extreme memory efficiency, fits on 8GB GPU</description></item>
    /// </list>
    /// </remarks>
    public QATMethod QATMethod { get; set; } = QATMethod.Standard;

    /// <summary>
    /// Gets or sets the number of warmup epochs before enabling quantization in QAT.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Train normally for a few epochs first, then enable quantization
    /// simulation. This helps the model converge before adding the quantization constraint.</para>
    /// </remarks>
    public int QATWarmupEpochs { get; set; } = 1;

    /// <summary>
    /// Converts this config to a QuantizationConfiguration for internal use.
    /// </summary>
    internal QuantizationConfiguration ToQuantizationConfiguration()
    {
        return new QuantizationConfiguration
        {
            Mode = Mode,
            Strategy = Strategy,
            Granularity = Granularity,
            GroupSize = GroupSize,
            TargetBitWidth = TargetBitWidth,
            CalibrationMethod = CalibrationMethod,
            UseSymmetricQuantization = UseSymmetricQuantization,
            NumCalibrationSamples = CalibrationSamples,
            QuantizeActivations = QuantizeActivations,
            UseQuantizationAwareTraining = UseQuantizationAwareTraining,
            QATMethod = QATMethod,
            QATWarmupEpochs = QATWarmupEpochs
        };
    }

    /// <summary>
    /// Creates a configuration for GPTQ 4-bit quantization.
    /// Best for achieving high accuracy at 4-bit precision.
    /// </summary>
    public static QuantizationConfig ForGPTQ(int groupSize = 128)
    {
        return new QuantizationConfig
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 4,
            Strategy = QuantizationStrategy.GPTQ,
            Granularity = QuantizationGranularity.PerGroup,
            GroupSize = groupSize,
            CalibrationMethod = CalibrationMethod.MinMax,
            UseSymmetricQuantization = true
        };
    }

    /// <summary>
    /// Creates a configuration for AWQ 4-bit quantization.
    /// Best for very large models (70B+ parameters).
    /// </summary>
    public static QuantizationConfig ForAWQ(int groupSize = 128)
    {
        return new QuantizationConfig
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 4,
            Strategy = QuantizationStrategy.AWQ,
            Granularity = QuantizationGranularity.PerGroup,
            GroupSize = groupSize,
            CalibrationMethod = CalibrationMethod.MinMax,
            UseSymmetricQuantization = true
        };
    }

    /// <summary>
    /// Creates a configuration for SmoothQuant W8A8 quantization.
    /// Enables quantization of both weights and activations to 8-bit.
    /// </summary>
    public static QuantizationConfig ForSmoothQuant()
    {
        return new QuantizationConfig
        {
            Mode = QuantizationMode.Int8,
            Strategy = QuantizationStrategy.SmoothQuant,
            Granularity = QuantizationGranularity.PerChannel,
            CalibrationMethod = CalibrationMethod.MinMax,
            UseSymmetricQuantization = true,
            QuantizeActivations = true
        };
    }

    /// <summary>
    /// Creates a configuration for Quantization-Aware Training (QAT).
    /// Use when you can retrain the model and need maximum accuracy.
    /// </summary>
    public static QuantizationConfig ForQAT(int targetBitWidth = 8, QATMethod method = QATMethod.EfficientQAT)
    {
        return new QuantizationConfig
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = targetBitWidth,
            UseQuantizationAwareTraining = true,
            QATMethod = method,
            QATWarmupEpochs = 1,
            UseSymmetricQuantization = true,
            QuantizeActivations = true
        };
    }
}
