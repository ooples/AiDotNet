using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Deployment.Optimization.Quantization;

/// <summary>
/// Configuration for model quantization - comprehensive settings for PTQ and QAT.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Quantization compresses your model by using smaller numbers.
/// This configuration lets you control exactly how that compression happens.</para>
///
/// <para><b>Quick Start Examples:</b></para>
/// <code>
/// // Simple INT8 quantization (4x compression)
/// config.Mode = QuantizationMode.Int8;
/// config.Strategy = QuantizationStrategy.MinMax;
///
/// // High-quality 4-bit with GPTQ (8x compression)
/// config.Mode = QuantizationMode.Int8;
/// config.TargetBitWidth = 4;
/// config.Strategy = QuantizationStrategy.GPTQ;
/// config.Granularity = QuantizationGranularity.PerGroup;
/// config.GroupSize = 128;
///
/// // Quantization-Aware Training for best accuracy
/// config.UseQuantizationAwareTraining = true;
/// config.QATMethod = QATMethod.EfficientQAT;
/// </code>
///
/// <para><b>Research References:</b></para>
/// <list type="bullet">
/// <item><description>GPTQ: Frantar et al., 2023 - Second-order Hessian-based quantization</description></item>
/// <item><description>AWQ: Lin et al., 2024 - Activation-aware weight quantization</description></item>
/// <item><description>SmoothQuant: Xiao et al., 2023 - Outlier smoothing for W8A8</description></item>
/// <item><description>EfficientQAT: ACL 2025 - Memory-efficient QAT for LLMs</description></item>
/// </list>
/// </remarks>
public class QuantizationConfiguration
{
    /// <summary>
    /// Gets or sets the quantization mode (Int8, Float16, etc.).
    /// </summary>
    public QuantizationMode Mode { get; set; } = QuantizationMode.Int8;

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
    /// Gets or sets the group size for per-group quantization.
    /// Only used when Granularity is PerGroup or PerBlock.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Smaller groups = more accuracy but more storage overhead.</para>
    /// <para><b>Typical values:</b> 32, 64, 128 (default), 256</para>
    /// <para><b>Storage overhead:</b> Group size 128 adds ~0.125 bits per weight</para>
    /// </remarks>
    public int GroupSize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the target bit width for weight quantization.
    /// If null, uses the default bit width for the Mode.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Override the default bit width. For example, use INT8 mode
    /// but target 4-bit weights for more aggressive compression.</para>
    /// <para><b>Common values:</b> 2, 3, 4, 8, 16</para>
    /// </remarks>
    public int? TargetBitWidth { get; set; }

    /// <summary>
    /// Gets the effective bit width (target or default based on mode).
    /// </summary>
    public int EffectiveBitWidth => TargetBitWidth ?? DefaultBitWidth;

    /// <summary>
    /// Gets the default bit width for the current quantization mode.
    /// </summary>
    private int DefaultBitWidth => Mode switch
    {
        QuantizationMode.Int8 => 8,
        QuantizationMode.Float16 => 16,
        QuantizationMode.Float32 => 32,
        QuantizationMode.Dynamic => 8,
        _ => 32
    };

    /// <summary>
    /// Gets the bit width for the current quantization mode.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is computed automatically based on the Mode:
    /// - Int8: 8 bits (smallest, fastest, some accuracy loss)
    /// - Float16: 16 bits (balanced speed and accuracy)
    /// - Float32: 32 bits (full precision, no compression)
    /// - Dynamic: 8 bits (dynamic range quantization)
    /// </para>
    /// </remarks>
    [Obsolete("Use EffectiveBitWidth instead for accurate bit width including TargetBitWidth override")]
    public int BitWidth => DefaultBitWidth;

    /// <summary>
    /// Gets or sets whether to use symmetric quantization (default: true).
    /// </summary>
    public bool UseSymmetricQuantization { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use per-channel quantization.
    /// </summary>
    /// <remarks>
    /// <para><b>Note:</b> This property is kept for backward compatibility.
    /// Setting this to true will automatically set <see cref="Granularity"/> to PerChannel.
    /// For new code, prefer setting <see cref="Granularity"/> directly.</para>
    /// </remarks>
    public bool UsePerChannelQuantization
    {
        get => Granularity == QuantizationGranularity.PerChannel;
        set
        {
            if (value)
            {
                Granularity = QuantizationGranularity.PerChannel;
            }
            else if (Granularity == QuantizationGranularity.PerChannel)
            {
                Granularity = QuantizationGranularity.PerTensor;
            }
        }
    }

    /// <summary>
    /// Gets or sets the calibration method.
    /// </summary>
    public CalibrationMethod CalibrationMethod { get; set; } = CalibrationMethod.MinMax;

    /// <summary>
    /// Gets or sets the number of calibration samples to use.
    /// </summary>
    public int NumCalibrationSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets layers to skip during quantization.
    /// </summary>
    public HashSet<string> SkipLayers { get; set; } = new();

    /// <summary>
    /// Gets or sets whether to quantize only weights or both weights and activations.
    /// </summary>
    public bool QuantizeActivations { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum scale factor to prevent underflow.
    /// </summary>
    public double MinScaleFactor { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum scale factor to prevent overflow.
    /// </summary>
    public double MaxScaleFactor { get; set; } = 1e6;

    /// <summary>
    /// Gets or sets whether to use quantization-aware training (QAT).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> QAT simulates quantization DURING training so the model
    /// learns to be robust to low precision. Results in better accuracy than PTQ.</para>
    /// <para><b>Trade-off:</b> Requires retraining but achieves 95-99% of original accuracy vs 85-95% for PTQ</para>
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
    /// <item><description><b>ParetoQ:</b> Best accuracy across all bit widths</description></item>
    /// </list>
    /// </remarks>
    public QATMethod QATMethod { get; set; } = QATMethod.Standard;

    /// <summary>
    /// Gets or sets the number of QAT warmup epochs before enabling fake quantization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Train normally for a few epochs first, then enable quantization
    /// simulation. This helps the model converge before adding the quantization constraint.</para>
    /// </remarks>
    public int QATWarmupEpochs { get; set; } = 1;

    /// <summary>
    /// Gets or sets the percentile to use for histogram-based calibration.
    /// </summary>
    public double HistogramPercentile { get; set; } = 99.99;

    /// <summary>
    /// Gets or sets the bit width for activation quantization (if QuantizeActivations is true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Activations (intermediate values) can use different precision
    /// than weights. Common configurations:</para>
    /// <list type="bullet">
    /// <item><description>W8A8: 8-bit weights, 8-bit activations (balanced)</description></item>
    /// <item><description>W4A16: 4-bit weights, 16-bit activations (more compression)</description></item>
    /// <item><description>W4A8: 4-bit weights, 8-bit activations (aggressive)</description></item>
    /// </list>
    /// </remarks>
    public int ActivationBitWidth { get; set; } = 8;

    /// <summary>
    /// Gets or sets the smoothing factor alpha for SmoothQuant strategy.
    /// Controls the balance of quantization difficulty between activations and weights.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Only used with SmoothQuant strategy.</para>
    /// <para>Alpha = 0.5 means equal difficulty split between activations and weights.</para>
    /// <para>Alpha closer to 1.0 puts more difficulty on weights (better for weight-only quantization).</para>
    /// <para><b>Default:</b> 0.5 (balanced)</para>
    /// </remarks>
    public double SmoothQuantAlpha { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the damping factor for GPTQ Hessian computation.
    /// Prevents numerical instability when inverting the Hessian matrix.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Only used with GPTQ strategy. Higher values make the algorithm
    /// more stable but potentially less accurate.</para>
    /// <para><b>Default:</b> 0.01 (standard value from GPTQ paper)</para>
    /// </remarks>
    public double GPTQDampingFactor { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether to use ActOrder optimization in GPTQ.
    /// Processes columns in order of decreasing activation magnitude.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Improves GPTQ accuracy by processing important weights first.
    /// Slightly slower but better results.</para>
    /// </remarks>
    public bool GPTQActOrder { get; set; } = true;

    /// <summary>
    /// Gets or sets the protection percentage for AWQ strategy.
    /// Percentage of weights to protect from aggressive quantization based on activation importance.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> AWQ identifies the most important weights and protects them.
    /// Higher values = more protection = better accuracy but less compression benefit.</para>
    /// <para><b>Scale:</b> Uses 0-100 scale (percentage). Value of 1.0 means 1%, not 100%.</para>
    /// <para><b>Default:</b> 1.0 (protect top 1% of important weights)</para>
    /// <para><b>Typical range:</b> 0.5 to 5.0 (i.e., 0.5% to 5% of weights)</para>
    /// </remarks>
    public double AWQProtectionPercentage { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the scale search options for AWQ grid search optimization.
    /// Default: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> AWQ searches for optimal scaling factors. More values = better
    /// accuracy but slower. Reduce for faster calibration or increase for better accuracy.</para>
    /// </remarks>
    public double[] AWQScaleSearchOptions { get; set; } = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0];

    /// <summary>
    /// Gets or sets custom quantization parameters per layer (by layer name).
    /// </summary>
    public Dictionary<string, LayerQuantizationParams> CustomLayerParams { get; set; } = new();

    /// <summary>
    /// Gets or sets per-category bit-width overrides for mixed-precision quantization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different layer types have different sensitivity to quantization.
    /// Attention layers are typically more sensitive and benefit from higher precision, while
    /// dense/MLP layers are more tolerant of aggressive quantization.</para>
    ///
    /// <para><b>How it works:</b> When the model implements <see cref="AiDotNet.Interfaces.ILayeredModel{T}"/>,
    /// quantizers can look up each layer's <see cref="AiDotNet.Interfaces.LayerCategory"/> in this dictionary
    /// to determine the target bit-width. Layers whose category is not present use the default
    /// <see cref="EffectiveBitWidth"/>.</para>
    ///
    /// <para><b>Example (common mixed-precision config):</b></para>
    /// <code>
    /// config.CategoryBitWidths = new Dictionary&lt;LayerCategory, int&gt;
    /// {
    ///     { LayerCategory.Attention, 8 },    // Keep attention at 8-bit (sensitive)
    ///     { LayerCategory.Embedding, 8 },     // Embeddings are sensitive too
    ///     { LayerCategory.Dense, 4 },          // Dense/MLP can handle 4-bit
    ///     { LayerCategory.FeedForward, 4 },    // Feed-forward blocks too
    ///     { LayerCategory.Normalization, 16 }, // Keep normalization at higher precision
    /// };
    /// </code>
    ///
    /// <para><b>Research References:</b></para>
    /// <list type="bullet">
    /// <item><description>CoopQ (2025): Cooperative game theory for per-layer bit-width allocation</description></item>
    /// <item><description>AWQ (MLSys 2024): Per-channel activation-aware weight quantization</description></item>
    /// <item><description>Layer-Sensitive Quantization (2025): Layer sensitivity metrics for mixed-precision</description></item>
    /// </list>
    /// </remarks>
    public Dictionary<LayerCategory, int>? CategoryBitWidths { get; set; }

    /// <summary>
    /// Creates a configuration for INT8 quantization.
    /// </summary>
    public static QuantizationConfiguration ForInt8(CalibrationMethod method = CalibrationMethod.MinMax)
    {
        return new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            CalibrationMethod = method,
            UseSymmetricQuantization = true,
            QuantizeActivations = true
        };
    }

    /// <summary>
    /// Creates a configuration for FP16 quantization.
    /// </summary>
    public static QuantizationConfiguration ForFloat16()
    {
        return new QuantizationConfiguration
        {
            Mode = QuantizationMode.Float16,
            CalibrationMethod = CalibrationMethod.None,
            UseSymmetricQuantization = false,
            QuantizeActivations = true
        };
    }

    /// <summary>
    /// Creates a configuration for dynamic quantization (weights only).
    /// </summary>
    public static QuantizationConfiguration ForDynamic()
    {
        return new QuantizationConfiguration
        {
            Mode = QuantizationMode.Dynamic,
            CalibrationMethod = CalibrationMethod.None,
            UseSymmetricQuantization = true,
            QuantizeActivations = false
        };
    }

    /// <summary>
    /// Creates a configuration for GPTQ 4-bit quantization.
    /// Best for achieving high accuracy at 4-bit precision.
    /// </summary>
    /// <param name="groupSize">Group size for per-group quantization (default: 128)</param>
    /// <param name="actOrder">Whether to use activation order optimization (default: true)</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> GPTQ is the gold standard for 4-bit quantization.
    /// It uses advanced math (Hessian matrix) to minimize accuracy loss.</para>
    /// <para><b>Typical results:</b> Within 1-2% of full precision accuracy at 4-bit</para>
    /// </remarks>
    public static QuantizationConfiguration ForGPTQ(int groupSize = 128, bool actOrder = true)
    {
        return new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 4,
            Strategy = QuantizationStrategy.GPTQ,
            Granularity = QuantizationGranularity.PerGroup,
            GroupSize = groupSize,
            CalibrationMethod = CalibrationMethod.MinMax,
            UseSymmetricQuantization = true,
            QuantizeActivations = false,
            GPTQActOrder = actOrder
        };
    }

    /// <summary>
    /// Creates a configuration for AWQ 4-bit quantization.
    /// Best for very large models (70B+ parameters).
    /// </summary>
    /// <param name="groupSize">Group size for per-group quantization (default: 128)</param>
    /// <param name="protectionPercentage">Percentage of weights to protect (default: 1.0)</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> AWQ is particularly good for very large models where some
    /// weights are disproportionately important. It identifies and protects these weights.</para>
    /// </remarks>
    public static QuantizationConfiguration ForAWQ(int groupSize = 128, double protectionPercentage = 1.0)
    {
        return new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 4,
            Strategy = QuantizationStrategy.AWQ,
            Granularity = QuantizationGranularity.PerGroup,
            GroupSize = groupSize,
            CalibrationMethod = CalibrationMethod.MinMax,
            UseSymmetricQuantization = true,
            QuantizeActivations = false,
            AWQProtectionPercentage = protectionPercentage
        };
    }

    /// <summary>
    /// Creates a configuration for SmoothQuant W8A8 quantization.
    /// Enables quantization of both weights and activations to 8-bit.
    /// </summary>
    /// <param name="alpha">Smoothing factor (0.0-1.0), balances difficulty between activations and weights</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> SmoothQuant makes it possible to quantize both weights AND
    /// activations (W8A8), which gives better speedup than weight-only quantization.</para>
    /// <para><b>Alpha guidance:</b> 0.5 is balanced, higher values favor weight quantization</para>
    /// </remarks>
    public static QuantizationConfiguration ForSmoothQuant(double alpha = 0.5)
    {
        return new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            Strategy = QuantizationStrategy.SmoothQuant,
            Granularity = QuantizationGranularity.PerChannel,
            CalibrationMethod = CalibrationMethod.MinMax,
            UseSymmetricQuantization = true,
            QuantizeActivations = true,
            ActivationBitWidth = 8,
            SmoothQuantAlpha = alpha
        };
    }

    /// <summary>
    /// Creates a configuration for Quantization-Aware Training (QAT).
    /// Use when you can retrain the model and need maximum accuracy.
    /// </summary>
    /// <param name="targetBitWidth">Target bit width after training (default: 8)</param>
    /// <param name="method">QAT method to use (default: EfficientQAT)</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> QAT trains the model with quantization simulation, resulting
    /// in better accuracy than post-training quantization (PTQ). Requires retraining but worth it
    /// for production deployments.</para>
    /// <para><b>Typical improvement:</b> 5-15% better accuracy than PTQ at same bit width</para>
    /// </remarks>
    public static QuantizationConfiguration ForQAT(int targetBitWidth = 8, QATMethod method = QATMethod.EfficientQAT)
    {
        return new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = targetBitWidth,
            Strategy = QuantizationStrategy.MinMax,
            Granularity = QuantizationGranularity.PerChannel,
            UseQuantizationAwareTraining = true,
            QATMethod = method,
            QATWarmupEpochs = 1,
            UseSymmetricQuantization = true,
            QuantizeActivations = true
        };
    }

    /// <summary>
    /// Creates a configuration optimized for 4-bit QLoRA fine-tuning.
    /// Combines NF4 quantization with LoRA adapters.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> QLoRA lets you fine-tune large models on consumer GPUs by
    /// keeping the base model in 4-bit and only training small adapter matrices.</para>
    /// <para><b>Memory savings:</b> Fine-tune a 65B model on a single 48GB GPU</para>
    /// </remarks>
    public static QuantizationConfiguration ForQLoRA()
    {
        return new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 4,
            Strategy = QuantizationStrategy.Dynamic,
            Granularity = QuantizationGranularity.PerGroup,
            GroupSize = 64,
            UseSymmetricQuantization = false,
            QuantizeActivations = false,
            UseQuantizationAwareTraining = true,
            QATMethod = QATMethod.QABLoRA
        };
    }

    /// <summary>
    /// Gets the effective bit-width for a specific layer category, considering
    /// per-category overrides in <see cref="CategoryBitWidths"/>.
    /// </summary>
    /// <param name="category">The layer category to look up.</param>
    /// <returns>The bit-width for this category, or <see cref="EffectiveBitWidth"/> if no override exists.</returns>
    public int GetBitWidthForCategory(AiDotNet.Interfaces.LayerCategory category)
    {
        if (CategoryBitWidths is null || !CategoryBitWidths.TryGetValue(category, out int bitWidth))
        {
            return EffectiveBitWidth;
        }

        if (bitWidth <= 0)
        {
            throw new ArgumentException(
                $"CategoryBitWidths contains an invalid bit-width ({bitWidth}) for category '{category}'. " +
                "Bit-widths must be positive.");
        }

        return bitWidth;
    }

    /// <summary>
    /// Gets the effective bit-width for a specific layer, checking name-based overrides first,
    /// then category-based overrides, then the default.
    /// </summary>
    /// <param name="layerInfo">The layer metadata.</param>
    /// <returns>The effective bit-width for this specific layer.</returns>
    public int GetBitWidthForLayer<T>(AiDotNet.Interfaces.LayerInfo<T> layerInfo)
    {
        if (layerInfo is null)
        {
            return EffectiveBitWidth;
        }

        // Name-based override takes highest priority
        if (CustomLayerParams.TryGetValue(layerInfo.Name, out var layerParams) &&
            layerParams.BitWidth.HasValue)
        {
            if (layerParams.BitWidth.Value <= 0)
            {
                throw new ArgumentException(
                    $"CustomLayerParams entry for layer '{layerInfo.Name}' has invalid BitWidth " +
                    $"{layerParams.BitWidth.Value}. BitWidth must be a positive integer.",
                    nameof(layerInfo));
            }
            return layerParams.BitWidth.Value;
        }

        // Skip layers get full precision
        if (SkipLayers.Contains(layerInfo.Name))
        {
            return 32;
        }

        // Category-based override
        return GetBitWidthForCategory(layerInfo.Category);
    }

    /// <summary>
    /// Creates a mixed-precision configuration for layer-aware quantization.
    /// Attention and embedding layers get higher precision while dense/MLP layers get lower.
    /// </summary>
    /// <remarks>
    /// <para><see cref="LayerCategory.Normalization"/> layers are always set to 16-bit regardless of
    /// <paramref name="sensitiveBitWidth"/>. Normalization layers (e.g., BatchNorm, LayerNorm)
    /// are extremely sensitive to quantization noise because small shifts in their statistics
    /// propagate through all subsequent layers, so they require higher precision than other
    /// sensitive layers.</para>
    /// </remarks>
    /// <param name="sensitiveBitWidth">Bit-width for sensitive layers (attention, embedding). Default: 8.
    /// Note: <see cref="LayerCategory.Normalization"/> layers are overridden to 16-bit and will
    /// not follow this parameter.</param>
    /// <param name="aggressiveBitWidth">Bit-width for tolerant layers (dense, feedforward). Default: 4.</param>
    /// <param name="groupSize">Group size for per-group quantization. Default: 128.</param>
    public static QuantizationConfiguration ForMixedPrecision(
        int sensitiveBitWidth = 8, int aggressiveBitWidth = 4, int groupSize = 128)
    {
        if (sensitiveBitWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(sensitiveBitWidth), "Bit width must be positive.");
        if (aggressiveBitWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(aggressiveBitWidth), "Bit width must be positive.");
        if (groupSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(groupSize), "Group size must be positive.");

        return new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = aggressiveBitWidth,
            Strategy = QuantizationStrategy.GPTQ,
            Granularity = QuantizationGranularity.PerGroup,
            GroupSize = groupSize,
            CalibrationMethod = CalibrationMethod.MinMax,
            UseSymmetricQuantization = true,
            QuantizeActivations = false,
            CategoryBitWidths = new Dictionary<AiDotNet.Interfaces.LayerCategory, int>
            {
                { AiDotNet.Interfaces.LayerCategory.Attention, sensitiveBitWidth },
                { AiDotNet.Interfaces.LayerCategory.Embedding, sensitiveBitWidth },
                { AiDotNet.Interfaces.LayerCategory.Normalization, 16 },
                { AiDotNet.Interfaces.LayerCategory.Dense, aggressiveBitWidth },
                { AiDotNet.Interfaces.LayerCategory.FeedForward, aggressiveBitWidth },
                { AiDotNet.Interfaces.LayerCategory.Convolution, aggressiveBitWidth },
            }
        };
    }
}
