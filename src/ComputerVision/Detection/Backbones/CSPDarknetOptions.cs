using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Configuration options for the CSPDarknet detection backbone.
/// </summary>
/// <remarks>
/// <para>
/// Introduced by issue #2090: these values were constructor parameters, so the backbone advertised
/// no configuration surface at all and nothing could be set through an options object.
/// </para>
/// </remarks>
public class CSPDarknetOptions : DetectionBackboneOptions
{
    /// <summary>
    /// Multiplier on the number of residual blocks per stage. Default: <c>1.0</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Scales how DEEP the network is. YOLO ships small/medium/large variants that differ mostly by this number and the width one below.</para>
    /// </remarks>
    public double Depth { get; set; } = 1.0;

    /// <summary>
    /// Multiplier on the channel count of every stage. Default: <c>1.0</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Scales how WIDE the network is - how many feature detectors each layer has.</para>
    /// </remarks>
    public double WidthMultiplier { get; set; } = 1.0;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working backbone.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a dimension or multiplier is not greater than zero.
    /// </exception>
    public void Validate()
    {
        ValidateBackboneCore();
        Require(Depth, nameof(Depth));
        Require(WidthMultiplier, nameof(WidthMultiplier));
    }
}
