using AiDotNet.Enums;

namespace AiDotNet.Models.Results;

/// <summary>
/// Contains information about model quantization applied during or after training.
/// Provides metrics on compression ratio, accuracy impact, and quantization parameters.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After quantizing (compressing) your model, this class tells you
/// how much smaller it got, what technique was used, and other useful information about the
/// compression process.</para>
///
/// <para><b>Key Metrics:</b></para>
/// <list type="bullet">
/// <item><description><b>CompressionRatio:</b> How many times smaller the model is (e.g., 4.0 means 4x smaller)</description></item>
/// <item><description><b>BitWidth:</b> How many bits per weight (e.g., 8-bit, 4-bit)</description></item>
/// <item><description><b>OriginalSizeBytes:</b> Model size before compression</description></item>
/// <item><description><b>QuantizedSizeBytes:</b> Model size after compression</description></item>
/// </list>
/// </remarks>
public class QuantizationInfo
{
    /// <summary>
    /// Gets whether quantization was applied to this model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If false, the model is in full precision (32-bit floats)
    /// and was not compressed.</para>
    /// </remarks>
    public bool IsQuantized { get; init; }

    /// <summary>
    /// Gets the quantization mode used (Int8, Float16, etc.).
    /// </summary>
    public QuantizationMode Mode { get; init; } = QuantizationMode.None;

    /// <summary>
    /// Gets the quantization strategy (algorithm) used (GPTQ, AWQ, etc.).
    /// </summary>
    public QuantizationStrategy Strategy { get; init; } = QuantizationStrategy.Dynamic;

    /// <summary>
    /// Gets the quantization granularity (PerTensor, PerChannel, PerGroup).
    /// </summary>
    public QuantizationGranularity Granularity { get; init; } = QuantizationGranularity.PerTensor;

    /// <summary>
    /// Gets the bit width used for quantized weights.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lower bit width = more compression but potentially less accuracy.</para>
    /// <list type="bullet">
    /// <item><description>32-bit: Full precision (no compression)</description></item>
    /// <item><description>16-bit: Half precision (2x compression)</description></item>
    /// <item><description>8-bit: Standard quantization (4x compression)</description></item>
    /// <item><description>4-bit: Aggressive quantization (8x compression)</description></item>
    /// </list>
    /// </remarks>
    public int BitWidth { get; init; } = 32;

    /// <summary>
    /// Gets the group size used for per-group quantization.
    /// Only applicable when Granularity is PerGroup or PerBlock.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Smaller groups = more accuracy but more storage overhead.
    /// Typical values are 32, 64, 128, or 256.</para>
    /// </remarks>
    public int GroupSize { get; init; } = 128;

    /// <summary>
    /// Gets the original model size in bytes before quantization.
    /// </summary>
    public long OriginalSizeBytes { get; init; }

    /// <summary>
    /// Gets the quantized model size in bytes after quantization.
    /// </summary>
    public long QuantizedSizeBytes { get; init; }

    /// <summary>
    /// Gets the compression ratio (original size / quantized size).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A compression ratio of 4.0 means the model is 4 times smaller.
    /// Higher is better for storage/memory, but may impact accuracy.</para>
    /// </remarks>
    public double CompressionRatio => QuantizedSizeBytes > 0
        ? (double)OriginalSizeBytes / QuantizedSizeBytes
        : 1.0;

    /// <summary>
    /// Gets the total number of quantized parameters.
    /// </summary>
    public long TotalParameters { get; init; }

    /// <summary>
    /// Gets the number of parameters that were actually quantized.
    /// Some sensitive parameters may be kept at higher precision.
    /// </summary>
    public long QuantizedParameters { get; init; }

    /// <summary>
    /// Gets the percentage of parameters that were quantized.
    /// </summary>
    public double QuantizedPercentage => TotalParameters > 0
        ? (double)QuantizedParameters / TotalParameters * 100.0
        : 0.0;

    /// <summary>
    /// Gets whether Quantization-Aware Training (QAT) was used.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If true, the model was trained with quantization simulation,
    /// typically resulting in better accuracy than post-training quantization (PTQ).</para>
    /// </remarks>
    public bool UsedQAT { get; init; }

    /// <summary>
    /// Gets the QAT method used if QAT was enabled.
    /// </summary>
    public QATMethod? QATMethod { get; init; }

    /// <summary>
    /// Gets the number of calibration samples used for quantization.
    /// </summary>
    public int CalibrationSamples { get; init; }

    /// <summary>
    /// Gets the calibration method used to determine quantization parameters.
    /// </summary>
    public CalibrationMethod CalibrationMethod { get; init; } = CalibrationMethod.MinMax;

    /// <summary>
    /// Gets whether symmetric quantization was used.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Symmetric quantization treats positive and negative values
    /// the same way (e.g., -127 to 127 for INT8). Asymmetric can be more accurate but slightly
    /// slower (e.g., 0 to 255 with a zero-point offset).</para>
    /// </remarks>
    public bool IsSymmetric { get; init; } = true;

    /// <summary>
    /// Gets whether activations were also quantized (in addition to weights).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Weights are the model's learned parameters. Activations are
    /// the intermediate values computed during inference. Quantizing both (W8A8) gives more
    /// speedup but requires careful calibration.</para>
    /// </remarks>
    public bool ActivationsQuantized { get; init; }

    /// <summary>
    /// Gets the bit width used for activation quantization (if applicable).
    /// </summary>
    public int? ActivationBitWidth { get; init; }

    /// <summary>
    /// Gets the time taken to perform quantization in milliseconds.
    /// </summary>
    public double QuantizationTimeMs { get; init; }

    /// <summary>
    /// Gets per-layer quantization information if available.
    /// Key is layer name, value contains layer-specific quantization parameters.
    /// </summary>
    public IReadOnlyDictionary<string, LayerQuantizationInfo>? LayerInfo { get; init; }

    /// <summary>
    /// Gets any warnings or notes generated during quantization.
    /// </summary>
    public IReadOnlyList<string> Warnings { get; init; } = [];

    /// <summary>
    /// Creates a default QuantizationInfo indicating no quantization was applied.
    /// </summary>
    public static QuantizationInfo None => new()
    {
        IsQuantized = false,
        Mode = QuantizationMode.None,
        BitWidth = 32
    };

    /// <summary>
    /// Returns a human-readable summary of the quantization.
    /// </summary>
    public override string ToString()
    {
        if (!IsQuantized)
            return "Not quantized (FP32)";

        var qatStr = UsedQAT ? $" with {QATMethod} QAT" : " (PTQ)";
        return $"{Mode} ({BitWidth}-bit, {Strategy}, {Granularity}){qatStr}, {CompressionRatio:F1}x compression";
    }
}

/// <summary>
/// Contains quantization information for a specific layer.
/// </summary>
public class LayerQuantizationInfo
{
    /// <summary>
    /// Gets the layer name.
    /// </summary>
    public string LayerName { get; init; } = string.Empty;

    /// <summary>
    /// Gets the layer type (e.g., "Dense", "Conv2D").
    /// </summary>
    public string LayerType { get; init; } = string.Empty;

    /// <summary>
    /// Gets whether this layer was quantized.
    /// </summary>
    public bool IsQuantized { get; init; }

    /// <summary>
    /// Gets the bit width used for this layer's weights.
    /// </summary>
    public int BitWidth { get; init; }

    /// <summary>
    /// Gets the scale factor used for quantization.
    /// </summary>
    public double Scale { get; init; }

    /// <summary>
    /// Gets the zero point used for asymmetric quantization.
    /// </summary>
    public int ZeroPoint { get; init; }

    /// <summary>
    /// Gets the number of parameters in this layer.
    /// </summary>
    public long ParameterCount { get; init; }

    /// <summary>
    /// Gets the quantization error (mean squared error) for this layer.
    /// Lower is better.
    /// </summary>
    public double QuantizationError { get; init; }

    /// <summary>
    /// Gets the reason if this layer was skipped during quantization.
    /// </summary>
    public string? SkipReason { get; init; }
}
