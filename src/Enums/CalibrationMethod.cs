namespace AiDotNet.Enums;

/// <summary>
/// Calibration methods for quantization - techniques to determine optimal scaling factors
/// when converting high-precision models to low-precision formats.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Quantization compresses AI models by using smaller numbers (e.g., 8-bit
/// instead of 32-bit). Calibration is the process of figuring out the best way to map large
/// numbers to small numbers without losing too much accuracy.
///
/// Think of it like adjusting a thermostat - you need to find the right scale to represent
/// temperatures accurately. Different calibration methods use different strategies:
///
/// - **None**: No calibration, uses default scaling. Fastest but least accurate.
/// - **MinMax**: Simple approach using the min/max values seen in data. Fast and usually good enough.
/// - **Histogram**: Analyzes distribution of values to find better scaling. More accurate than MinMax.
/// - **Entropy**: Uses information theory (KL divergence) to minimize information loss. Most accurate but slowest.
/// - **MSE**: Minimizes the mean squared error between original and quantized values.
/// - **Percentile**: Ignores outliers by using percentiles instead of absolute min/max. Good for noisy data.
///
/// For most cases, **MinMax** or **Histogram** provides a good balance of speed and accuracy.
/// </remarks>
public enum CalibrationMethod
{
    /// <summary>
    /// No calibration - uses default symmetric scaling.
    /// Fastest but may reduce accuracy. Use only if calibration data is unavailable.
    /// </summary>
    None,

    /// <summary>
    /// Min-Max calibration - simple range-based scaling.
    /// Fast and effective for most cases. Good default choice.
    /// </summary>
    MinMax,

    /// <summary>
    /// Histogram-based calibration - analyzes value distribution using percentiles.
    /// More robust than MinMax, handles outliers better. Slightly slower.
    /// </summary>
    Histogram,

    /// <summary>
    /// Entropy-based calibration - uses KL divergence to minimize information loss.
    /// Most accurate but slowest. Best for when you need maximum model quality.
    /// </summary>
    Entropy,

    /// <summary>
    /// Mean Squared Error (MSE) based calibration.
    /// Minimizes difference between original and quantized values.
    /// </summary>
    MSE,

    /// <summary>
    /// Percentile-based calibration - uses percentiles to handle outliers.
    /// Good for data with extreme values that shouldn't affect scaling.
    /// </summary>
    Percentile
}
