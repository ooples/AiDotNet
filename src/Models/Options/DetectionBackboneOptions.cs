namespace AiDotNet.Models.Options;

/// <summary>
/// Options shared by the detection backbones in <c>src/ComputerVision/Detection/Backbones</c> —
/// the feature extractors (ResNet, EfficientNet, SwinTransformer, CSPDarknet) that a detector sits
/// on top of.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A backbone is the part of a vision model that turns an image into a stack
/// of increasingly abstract feature maps. Detectors and segmenters bolt their own heads onto one of
/// these rather than starting from pixels themselves, which is why the same four backbones appear
/// across many different models.
/// </para>
/// <para>
/// Only <see cref="InChannels"/> is declared here: it is the one knob all four share and all four
/// called by that name. Each backbone's <c>Variant</c> names its own enum
/// (<c>ResNetVariant</c>, <c>SwinVariant</c>, …) so there is no common type to declare, and it
/// stays on the leaf — the same reasoning that kept <c>ModelSize</c> off
/// <see cref="SegmentationModelOptions"/>.
/// </para>
/// <para>
/// These classes are named <c>…BackboneOptions</c> rather than <c>ResNetOptions</c> /
/// <c>EfficientNetOptions</c> because those names are already taken by the options for the
/// standalone <c>ResNetNetwork</c> and <c>EfficientNetNetwork</c> classifiers, which are different
/// types with their own configuration.
/// </para>
/// </remarks>
public abstract class DetectionBackboneOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the number of channels in the input image. Default: 3.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Three for an ordinary colour image (red, green, blue). One for
    /// greyscale, and more for imagery that records bands the eye cannot see — satellite and
    /// medical scans often have four or more.
    /// </para>
    /// <para>
    /// This sizes the very first convolution, so it is part of the network's shape rather than a
    /// property of the data alone: change it and the stem layer changes with it.
    /// </para>
    /// </remarks>
    public int InChannels { get; set; } = 3;

    /// <summary>
    /// Throws when the values shared by every backbone cannot produce a working network.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="InChannels"/> is not positive.
    /// </exception>
    protected void ValidateBackboneCore()
    {
        Require(InChannels, nameof(InChannels));
    }
}
