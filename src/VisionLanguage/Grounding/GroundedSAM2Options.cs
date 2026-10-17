using AiDotNet.Models.Options;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Grounded SAM 2: Grounding DINO finds the boxes a caption names, and SAM 2 segments each box.
/// </summary>
/// <remarks>
/// <para>The defaults are the official demo's (IDEA-Research/Grounded-SAM-2, grounded_sam2_local_demo.py): Grounding
/// DINO-T with box and text thresholds 0.35 and 0.25, and SAM 2.1 Hiera-L (sam2.1_hiera_large: blocks 2, 6, 36, 4;
/// widths 144 to 1152; heads 2 to 16; windows 8, 4, 16, 8; global attention in blocks 23, 33 and 43) at 1024 px.</para>
/// <para><b>For Beginners:</b> These options configure the two models Grounded SAM 2 runs one after the other.</para>
/// </remarks>
public class GroundedSAM2Options : ModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GroundedSAM2Options(GroundedSAM2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        Detector = new GroundingDINOOptions(other.Detector);
        SamImageSize = other.SamImageSize;
        SamBlocksPerStage = (int[])other.SamBlocksPerStage.Clone();
        SamEmbedDims = (int[])other.SamEmbedDims.Clone();
        SamHeadsPerStage = (int[])other.SamHeadsPerStage.Clone();
        SamWindowSizes = (int[])other.SamWindowSizes.Clone();
        SamGlobalAttentionBlocks = (int[])other.SamGlobalAttentionBlocks.Clone();
        SamDim = other.SamDim;
        SamMaskInChannels = other.SamMaskInChannels;
        SamDecoderHeads = other.SamDecoderHeads;
        SamDecoderMlpDim = other.SamDecoderMlpDim;
        SamIouHiddenDim = other.SamIouHiddenDim;
        MaskThreshold = other.MaskThreshold;
    }

    public GroundedSAM2Options()
    {
    }

    /// <summary>Gets or sets the detector's options (Grounding DINO-T by default). Its seed seeds the detector when set;
    /// otherwise <see cref="ModelOptions.Seed"/> does.</summary>
    public GroundingDINOOptions Detector { get; set; } = new();

    /// <summary>Gets or sets the side SAM 2 resizes every image to (1024).</summary>
    public int SamImageSize { get; set; } = 1024;

    /// <summary>Gets or sets the Hiera blocks per stage (Hiera-L: 2, 6, 36, 4).</summary>
    public int[] SamBlocksPerStage { get; set; } = [2, 6, 36, 4];

    /// <summary>Gets or sets the Hiera widths per stage (Hiera-L: 144, 288, 576, 1152).</summary>
    public int[] SamEmbedDims { get; set; } = [144, 288, 576, 1152];

    /// <summary>Gets or sets the Hiera heads per stage (Hiera-L: 2, 4, 8, 16).</summary>
    public int[] SamHeadsPerStage { get; set; } = [2, 4, 8, 16];

    /// <summary>Gets or sets the Hiera attention windows per stage (Hiera-L: 8, 4, 16, 8).</summary>
    public int[] SamWindowSizes { get; set; } = [8, 4, 16, 8];

    /// <summary>Gets or sets the Hiera blocks, counted from 0 over all stages, that attend globally (Hiera-L: 23, 33, 43).</summary>
    public int[] SamGlobalAttentionBlocks { get; set; } = [23, 33, 43];

    /// <summary>Gets or sets the width of SAM 2's neck, prompt encoder and mask decoder (256).</summary>
    public int SamDim { get; set; } = 256;

    /// <summary>Gets or sets the channels of the prompt encoder's mask embedding (16).</summary>
    public int SamMaskInChannels { get; set; } = 16;

    /// <summary>Gets or sets the mask decoder's attention heads (8).</summary>
    public int SamDecoderHeads { get; set; } = 8;

    /// <summary>Gets or sets the mask decoder's feed-forward width (2048).</summary>
    public int SamDecoderMlpDim { get; set; } = 2048;

    /// <summary>Gets or sets the IoU and object-score heads' hidden width (256).</summary>
    public int SamIouHiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the mask logit above which a pixel belongs to the mask (0, the image predictor's).</summary>
    public double MaskThreshold { get; set; } = 0.0;
}
