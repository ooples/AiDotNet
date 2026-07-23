namespace AiDotNet.Enums;

/// <summary>
/// Specifies the MobileNetV3 model variant.
/// </summary>
/// <remarks>
/// <para>
/// MobileNetV3 comes in two main variants: Large and Small.
/// Large is optimized for high accuracy, while Small is optimized for low latency.
/// </para>
/// <para>
/// <b>For Beginners:</b> Choose Large for tasks where accuracy is most important,
/// and Small for tasks where speed/latency is the priority.
/// </para>
/// </remarks>
public enum MobileNetV3Variant
{
    /// <summary>
    /// MobileNetV3-Large: Higher accuracy variant.
    /// </summary>
    /// <remarks>
    /// Optimized for accuracy with more layers and higher computational cost.
    /// </remarks>
    Large,

    /// <summary>
    /// MobileNetV3-Small: Lower latency variant.
    /// </summary>
    /// <remarks>
    /// Optimized for speed with fewer layers and lower computational cost.
    /// </remarks>
    Small
}
