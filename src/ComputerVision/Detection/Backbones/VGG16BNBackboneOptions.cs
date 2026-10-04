using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Configuration options for the VGG16-BN detection backbone.
/// </summary>
/// <remarks>
/// <para>
/// The input channel count was a constructor parameter, so the backbone advertised no configuration
/// surface and nothing could be set through an options object.
/// </para>
/// <para><b>For Beginners:</b> VGG16 with batch normalization is the feature extractor CRAFT's text
/// detector sits on. Its only knob is how many channels the input image has.</para>
/// </remarks>
public class VGG16BNBackboneOptions : DetectionBackboneOptions
{
    /// <summary>
    /// Throws when a value on this instance cannot produce a working backbone.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="DetectionBackboneOptions.InChannels"/> is not positive.
    /// </exception>
    public void Validate() => ValidateBackboneCore();

    /// <summary>The given options, or the defaults when none were supplied.</summary>
    internal static VGG16BNBackboneOptions OrDefault(VGG16BNBackboneOptions? options) => options ?? new VGG16BNBackboneOptions();
}
