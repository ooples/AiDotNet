using AiDotNet.Enums;

namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for model quantization - compressing models by using lower precision numbers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Quantization makes your AI model smaller and faster by using smaller numbers.
/// Think of it like compressing a high-quality photo - it takes less space but might lose a little quality.
///
/// Why use quantization?
/// - Smaller model size (50-75% reduction)
/// - Faster inference (2-4x speedup)
/// - Lower memory usage
/// - Enables deployment on mobile/edge devices
///
/// Trade-offs:
/// - Slightly lower accuracy (usually 1-5%)
/// - Some models are more sensitive than others
///
/// Modes:
/// - None: No quantization (full precision)
/// - Float16: Half precision (50% size reduction, minimal accuracy loss)
/// - Int8: 8-bit integers (75% size reduction, small accuracy loss)
///
/// For most models, Float16 is a great choice - significant benefits with minimal accuracy loss.
/// </para>
/// </remarks>
public class QuantizationConfig
{
    /// <summary>
    /// Gets or sets the quantization mode (default: None).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Choose which type of quantization to use:
    /// - None: Full precision, no compression
    /// - Float16: Half precision, good balance
    /// - Int8: Maximum compression, slight accuracy loss
    /// </para>
    /// </remarks>
    public QuantizationMode Mode { get; set; } = QuantizationMode.None;

    /// <summary>
    /// Gets or sets the calibration method used to determine optimal scaling factors (default: MinMax).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Calibration determines how to convert large numbers to small numbers.
    /// Only used for Int8 quantization. MinMax is fast and works well for most cases.
    /// Ignored for Float16 or None modes.
    /// </para>
    /// </remarks>
    public CalibrationMethod CalibrationMethod { get; set; } = CalibrationMethod.MinMax;

    /// <summary>
    /// Gets or sets whether to use symmetric quantization (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Symmetric quantization treats positive and negative values the same way.
    /// It's faster but asymmetric may be slightly more accurate. Use true for most cases.
    /// </para>
    /// </remarks>
    public bool UseSymmetricQuantization { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of calibration samples to use (default: 100).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More samples give better calibration but take longer.
    /// 100 is a good default. Use 1000+ for critical applications where accuracy is paramount.
    /// </para>
    /// </remarks>
    public int CalibrationSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to quantize only weights or both weights and activations (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> False means only compress the model parameters (weights).
    /// True means also compress the intermediate values during inference (activations).
    /// Activations give better compression but require calibration data.
    /// </para>
    /// </remarks>
    public bool QuantizeActivations { get; set; } = false;

    /// <summary>
    /// Creates a Float16 quantization configuration (recommended for most cases).
    /// </summary>
    /// <returns>A Float16 quantization configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for a good balance between model size and accuracy.
    /// Float16 provides 50% size reduction with minimal accuracy loss (typically less than 1%).
    /// This is the recommended starting point for quantization.
    /// </para>
    /// </remarks>
    public static QuantizationConfig Float16()
    {
        return new QuantizationConfig
        {
            Mode = QuantizationMode.Float16,
            CalibrationMethod = CalibrationMethod.None // Float16 doesn't need calibration
        };
    }

    /// <summary>
    /// Creates an Int8 quantization configuration with specified calibration method.
    /// </summary>
    /// <param name="calibrationMethod">The calibration method to use (default: MinMax).</param>
    /// <returns>An Int8 quantization configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for maximum compression (75% size reduction).
    /// Int8 provides the smallest models but requires calibration data and may lose 1-5% accuracy.
    /// Best for deployment to mobile or edge devices with limited storage.
    /// </para>
    /// </remarks>
    public static QuantizationConfig Int8(CalibrationMethod calibrationMethod = CalibrationMethod.MinMax)
    {
        return new QuantizationConfig
        {
            Mode = QuantizationMode.Int8,
            CalibrationMethod = calibrationMethod,
            QuantizeActivations = true // Int8 benefits from activation quantization
        };
    }

    /// <summary>
    /// Creates a configuration with no quantization (full precision).
    /// </summary>
    /// <returns>A configuration with no quantization.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to disable quantization entirely.
    /// The model will use full precision (32-bit floats), providing maximum accuracy
    /// but larger file size and slower inference. Use when accuracy is critical.
    /// </para>
    /// </remarks>
    public static QuantizationConfig None()
    {
        return new QuantizationConfig
        {
            Mode = QuantizationMode.None
        };
    }
}
