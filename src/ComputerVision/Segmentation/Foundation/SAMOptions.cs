using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the Segment Anything Model (SAM).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM is Meta AI's foundation model for image segmentation.
/// Options inherit from NeuralNetworkOptions and can be extended with SAM-specific settings.
/// </para>
/// </remarks>
public class SAMOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SAMOptions()
    {
        NumClasses = 1;
        DropRate = 0.1;
        ModelSize = SAMModelSize.ViTHuge;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SAMOptions(SAMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
        EncoderDim = other.EncoderDim;
        EncoderDepth = other.EncoderDepth;
        EncoderHeads = other.EncoderHeads;
        GlobalAttentionBlocks = (int[])other.GlobalAttentionBlocks.Clone();
        MaskFocalWeight = other.MaskFocalWeight;
        MaskDiceWeight = other.MaskDiceWeight;
        FocalAlpha = other.FocalAlpha;
        FocalGamma = other.FocalGamma;
    }

    /// <summary>
    /// Gets or sets the focal-loss weight in the mask objective. The default, 20, is the paper's
    /// focal:dice ratio of 20:1 (Kirillov et al. 2023, §3).
    /// </summary>
    public double MaskFocalWeight { get; set; } = 20.0;

    /// <summary>
    /// Gets or sets the dice-loss weight in the mask objective. The default, 1, is the paper's
    /// focal:dice ratio of 20:1.
    /// </summary>
    public double MaskDiceWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets focal loss's focusing parameter gamma. The default, 2, is the RetinaNet value
    /// the Segment Anything paper cites.
    /// </summary>
    public double FocalGamma { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets focal loss's class-balance parameter alpha. The default, 0.25, is the RetinaNet
    /// value the Segment Anything paper cites.
    /// </summary>
    public double FocalAlpha { get; set; } = 0.25;

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>SAMModelSize.ViTHuge</c>.
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
    /// <para>Setting it writes the variant's encoder width, depth, heads and global-attention blocks into
    /// <see cref="EncoderDim"/>, <see cref="EncoderDepth"/>, <see cref="EncoderHeads"/> and
    /// <see cref="GlobalAttentionBlocks"/>, so the options hold the sizes they build. A value set to something other
    /// than the previous variant's is kept, whichever order an initializer uses.</para>
    /// </remarks>
    public SAMModelSize ModelSize
    {
        get => _modelSize;
        set
        {
            var previous = SamConfiguration.For(_modelSize);
            var next = SamConfiguration.For(value);
            if (EncoderDim == previous.EncoderDim) EncoderDim = next.EncoderDim;
            if (EncoderDepth == previous.EncoderDepth) EncoderDepth = next.EncoderDepth;
            if (EncoderHeads == previous.EncoderHeads) EncoderHeads = next.EncoderHeads;
            if (GlobalAttentionBlocks.SequenceEqual(previous.GlobalAttention)) GlobalAttentionBlocks = (int[])next.GlobalAttention.Clone();
            _modelSize = value;
        }
    }

    private SAMModelSize _modelSize = SAMModelSize.ViTBase;

    /// <summary>Gets or sets the image encoder's width (ViT-B 768, ViT-L 1024, ViT-H 1280; set by <see cref="ModelSize"/>).</summary>
    public int EncoderDim { get; set; } = SamConfiguration.For(SAMModelSize.ViTBase).EncoderDim;

    /// <summary>Gets or sets the image encoder's blocks (12, 24, 32; set by <see cref="ModelSize"/>).</summary>
    public int EncoderDepth { get; set; } = SamConfiguration.For(SAMModelSize.ViTBase).EncoderDepth;

    /// <summary>Gets or sets the image encoder's attention heads (12, 16, 16; set by <see cref="ModelSize"/>).</summary>
    public int EncoderHeads { get; set; } = SamConfiguration.For(SAMModelSize.ViTBase).EncoderHeads;

    /// <summary>Gets or sets the encoder blocks that attend globally; the others attend within windows (set by
    /// <see cref="ModelSize"/>; each must be below <see cref="EncoderDepth"/>).</summary>
    public int[] GlobalAttentionBlocks { get; set; } = (int[])SamConfiguration.For(SAMModelSize.ViTBase).GlobalAttention.Clone();

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
