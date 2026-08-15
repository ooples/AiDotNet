using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// Configuration options for MedSAM universal medical segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MedSAM model. Default values follow the original paper settings.</para>
/// </remarks>
public class MedSAMOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MedSAMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MedSAMOptions(MedSAMOptions other)
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
    /// Per-stage channel widths of the hierarchical image encoder. Null (the default) uses the
    /// paper configuration for the selected <c>MedSAMModelSize</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These were previously unreachable: <c>GetModelConfig</c> hard-codes
    /// <c>([64, 128, 320, 768], [2, 2, 4, 12], 256)</c> and <c>MedSAMModelSize</c> declares only
    /// <c>ViTBase</c>, so no caller could build MedSAM at any size other than the full paper encoder —
    /// which is a ~90M-parameter ViT-Base. Exposing them keeps the paper defaults exactly as they were
    /// (null means "use GetModelConfig") while letting callers configure the encoder, which is what
    /// makes a bounded CI fixture or a memory-constrained deployment possible.
    /// </para>
    /// <para><b>For Beginners:</b> this controls how wide each stage of the image encoder is. Bigger
    /// numbers mean a more capable but slower and more memory-hungry model. Leave it unset to get the
    /// published configuration.</para>
    /// </remarks>
    public int[]? ChannelDims { get; set; }

    /// <summary>
    /// Number of transformer blocks in each encoder stage. Null (the default) uses the paper
    /// configuration for the selected <c>MedSAMModelSize</c>. Must be the same length as
    /// <see cref="ChannelDims"/> when both are supplied.
    /// </summary>
    public int[]? Depths { get; set; }

    /// <summary>
    /// Width of the mask decoder. Null (the default) uses the paper configuration (256).
    /// </summary>
    public int? DecoderDim { get; set; }
}
