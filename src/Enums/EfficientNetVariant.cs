namespace AiDotNet.Enums;

/// <summary>
/// Specifies the EfficientNet model variant.
/// </summary>
/// <remarks>
/// <para>
/// EfficientNet variants use compound scaling to balance network depth, width, and resolution.
/// Each variant (B0-B7) represents a different scale factor, with larger variants offering
/// better accuracy at the cost of more computation.
/// </para>
/// <para>
/// <b>For Beginners:</b> The variant number (B0-B7) indicates the scale of the network.
/// B0 is the smallest and fastest, while B7 is the largest with the highest accuracy.
/// Choose based on your accuracy requirements and computational budget.
/// </para>
/// </remarks>
public enum EfficientNetVariant
{
    /// <summary>
    /// EfficientNet-B0: Base model (5.3M parameters, 224x224 input).
    /// </summary>
    /// <remarks>
    /// The baseline model, offering good accuracy with minimal compute.
    /// </remarks>
    B0,

    /// <summary>
    /// EfficientNet-B1: Scaled model (7.8M parameters, 240x240 input).
    /// </summary>
    B1,

    /// <summary>
    /// EfficientNet-B2: Scaled model (9.2M parameters, 260x260 input).
    /// </summary>
    B2,

    /// <summary>
    /// EfficientNet-B3: Scaled model (12M parameters, 300x300 input).
    /// </summary>
    B3,

    /// <summary>
    /// EfficientNet-B4: Scaled model (19M parameters, 380x380 input).
    /// </summary>
    B4,

    /// <summary>
    /// EfficientNet-B5: Scaled model (30M parameters, 456x456 input).
    /// </summary>
    B5,

    /// <summary>
    /// EfficientNet-B6: Scaled model (43M parameters, 528x528 input).
    /// </summary>
    B6,

    /// <summary>
    /// EfficientNet-B7: Largest model (66M parameters, 600x600 input).
    /// </summary>
    /// <remarks>
    /// The largest standard variant, offering state-of-the-art accuracy.
    /// </remarks>
    B7
}
