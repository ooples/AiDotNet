namespace AiDotNet.Enums;

/// <summary>
/// Quality levels for adaptive inference on resource-constrained devices.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> When running AI models on devices with limited resources (like phones or
/// edge devices), you often need to trade off quality for speed or battery life. This enum lets you
/// choose how the model should balance these tradeoffs:
///
/// - **Low**: Fastest inference, uses least battery, but lower accuracy. Good for when battery is low
///   or the device is under heavy load.
/// - **Medium**: Balanced - decent speed and accuracy. Good default for most situations.
/// - **High**: Best accuracy, slower inference, uses more battery. Use when you need the best results
///   and the device has plenty of power.
///
/// The library can automatically switch between these levels based on battery level and CPU load.
/// </remarks>
public enum QualityLevel
{
    /// <summary>
    /// Low quality, maximum speed - prioritizes performance over accuracy.
    /// Uses aggressive optimizations like quantization and layer skipping.
    /// </summary>
    Low,

    /// <summary>
    /// Medium quality, balanced - good compromise between speed and accuracy.
    /// Uses moderate optimizations.
    /// </summary>
    Medium,

    /// <summary>
    /// High quality, maximum accuracy - prioritizes accuracy over speed.
    /// Uses minimal optimizations to preserve model quality.
    /// </summary>
    High
}
