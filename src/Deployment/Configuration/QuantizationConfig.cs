using AiDotNet.Enums;

namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for model quantization - compressing models by using lower precision numbers.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Quantization makes your AI model smaller and faster by using smaller numbers.
/// Think of it like compressing a high-quality photo - it takes less space but might lose a little quality.
///
/// **Why use quantization?**
/// - Smaller model size (50-75% reduction)
/// - Faster inference (2-4x speedup)
/// - Lower memory usage
/// - Enables deployment on mobile/edge devices
///
/// **Trade-offs:**
/// - Slightly lower accuracy (usually 1-5%)
/// - Some models are more sensitive than others
///
/// **Modes:**
/// - **None**: No quantization (full precision)
/// - **Float16**: Half precision (50% size reduction, minimal accuracy loss)
/// - **Int8**: 8-bit integers (75% size reduction, small accuracy loss)
///
/// For most models, **Float16** is a great choice - significant benefits with minimal accuracy loss.
/// </remarks>
public class QuantizationConfig
{
    /// <summary>
    /// Gets or sets the quantization mode (default: None).
    /// </summary>
    public QuantizationMode Mode { get; set; } = QuantizationMode.None;

    /// <summary>
    /// Gets or sets the calibration method used to determine optimal scaling factors (default: MinMax).
    /// Only used for Int8 quantization. Ignored for Float16 or None.
    /// </summary>
    public CalibrationMethod CalibrationMethod { get; set; } = CalibrationMethod.MinMax;

    /// <summary>
    /// Gets or sets whether to use symmetric quantization (default: true).
    /// Symmetric is faster, asymmetric may be slightly more accurate.
    /// </summary>
    public bool UseSymmetricQuantization { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of calibration samples to use (default: 100).
    /// More samples = better calibration but slower.
    /// </summary>
    public int CalibrationSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to quantize only weights or both weights and activations (default: false = weights only).
    /// Quantizing activations gives better compression but requires calibration data.
    /// </summary>
    public bool QuantizeActivations { get; set; } = false;

    /// <summary>
    /// Creates a Float16 quantization configuration (recommended for most cases).
    /// </summary>
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
    public static QuantizationConfig None()
    {
        return new QuantizationConfig
        {
            Mode = QuantizationMode.None
        };
    }
}
