using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Configuration options for the VGG16 detection backbone.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> VGG16 is the feature extractor EAST's text detector sits on. Its knobs are how many
/// channels the input image has and the seed of its initial weights.</para>
/// </remarks>
public class VGG16BackboneOptions : DetectionBackboneOptions
{
    /// <summary>
    /// Throws when a value on this instance cannot produce a working backbone.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="DetectionBackboneOptions.InChannels"/> is not positive.
    /// </exception>
    public void Validate() => ValidateBackboneCore();

    /// <summary>The given options, or the defaults when none were supplied.</summary>
    internal static VGG16BackboneOptions OrDefault(VGG16BackboneOptions? options) => options ?? new VGG16BackboneOptions();
}
