namespace AiDotNet.Enums;

/// <summary>
/// Specifies the width multiplier for MobileNetV2.
/// </summary>
/// <remarks>
/// <para>
/// The width multiplier (alpha) scales the number of channels in each layer.
/// Smaller multipliers result in faster, more compact models at the cost of accuracy.
/// </para>
/// <para>
/// <b>For Beginners:</b> The width multiplier controls how "wide" the network is.
/// A multiplier of 1.0 gives the standard network, while 0.5 uses half as many channels,
/// making the network faster but potentially less accurate.
/// </para>
/// </remarks>
public enum MobileNetV2WidthMultiplier
{
    /// <summary>
    /// Width multiplier of 0.35 - Extremely compact model.
    /// </summary>
    Alpha035,

    /// <summary>
    /// Width multiplier of 0.5 - Very compact model.
    /// </summary>
    Alpha050,

    /// <summary>
    /// Width multiplier of 0.75 - Compact model.
    /// </summary>
    Alpha075,

    /// <summary>
    /// Width multiplier of 1.0 - Standard model.
    /// </summary>
    Alpha100,

    /// <summary>
    /// Width multiplier of 1.3 - Wider model for higher accuracy.
    /// </summary>
    Alpha130,

    /// <summary>
    /// Width multiplier of 1.4 - Even wider model for maximum accuracy.
    /// </summary>
    Alpha140
}
