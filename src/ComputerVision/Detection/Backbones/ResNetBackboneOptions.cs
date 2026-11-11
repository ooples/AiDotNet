using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Configuration options for the ResNetBackbone detection backbone.
/// </summary>
/// <remarks>
/// <para>
/// Introduced by issue #2090: these values were constructor parameters, so the backbone advertised
/// no configuration surface at all and nothing could be set through an options object.
/// </para>
/// </remarks>
public class ResNetBackboneOptions : DetectionBackboneOptions
{
    /// <summary>
    /// Gets or sets which published size of the model to build. Default: <c>ResNetVariant.ResNet50</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The same architecture is usually published at several sizes. A bigger
    /// one is more accurate and slower; a smaller one fits where the big one will not. Picking a
    /// variant selects the widths and depths the paper reports for it - it changes the network that
    /// gets built, not merely how it is labelled.
    /// </para>
    /// </remarks>
    public ResNetVariant Variant { get; set; } = ResNetVariant.ResNet50;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working backbone.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a dimension or multiplier is not greater than zero.
    /// </exception>
    public void Validate()
    {
        ValidateBackboneCore();
    }
}
