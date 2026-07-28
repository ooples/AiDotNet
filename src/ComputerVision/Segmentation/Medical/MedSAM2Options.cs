using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// Configuration options for MedSAM 2 3D medical segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MedSAM2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class MedSAM2Options : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MedSAM2Options() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MedSAM2Options(MedSAM2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
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
}
