using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// Configuration options for MedSAM 2 3D medical segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MedSAM2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class MedSAM2Options : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MedSAM2Options()
    {
        NumClasses = 1;
        DropRate = 0;
        ModelSize = MedSAM2ModelSize.Tiny;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MedSAM2Options(MedSAM2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
        ChannelDims = other.ChannelDims is null ? null : (int[])other.ChannelDims.Clone();
        Depths = other.Depths is null ? null : (int[])other.Depths.Clone();
        DecoderDim = other.DecoderDim;
    }

    /// <summary>
    /// Per-stage channel widths of the hierarchical image encoder. Null (the default) uses the paper
    /// configuration for the selected <c>MedSAM2ModelSize</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The presets in <c>GetModelConfig</c> (Tiny [96,192,384,768], Base [112,224,448,896], Large
    /// [144,288,576,1152]) were previously the only reachable configurations, so even the smallest
    /// build was a full Hiera encoder. Exposing these keeps every preset exactly as published — null
    /// selects the preset — while allowing a bounded fixture or a memory-constrained deployment.
    /// </para>
    /// <para><b>For Beginners:</b> this sets how wide each stage of the image encoder is. Larger values
    /// give a more capable but slower and more memory-hungry model. Leave unset for the published
    /// configuration.</para>
    /// </remarks>
    public int[]? ChannelDims { get; set; }

    /// <summary>
    /// Number of transformer blocks per encoder stage. Null (the default) uses the paper configuration.
    /// Must match the length of <see cref="ChannelDims"/> when both are supplied.
    /// </summary>
    public int[]? Depths { get; set; }

    /// <summary>
    /// Width of the mask decoder. Null (the default) uses the paper configuration (256).
    /// </summary>
    public int? DecoderDim { get; set; }

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>MedSAM2ModelSize.Tiny</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Most of these models were published in several sizes that
    /// trade accuracy against speed and memory. Picking a variant selects the widths and
    /// depths the paper reports for it; it is not a hint, it changes the network that gets
    /// built.
    /// </para>
    /// <para>
    /// Declared here rather than on <see cref="SegmentationModelOptions"/> because each
    /// model names its own variants with its own enum, so there is no shared type to
    /// declare.
    /// </para>
    /// </remarks>
    public MedSAM2ModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
