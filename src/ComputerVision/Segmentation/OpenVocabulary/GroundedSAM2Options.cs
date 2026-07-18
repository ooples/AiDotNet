using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.OpenVocabulary;

/// <summary>
/// Configuration options for Grounded SAM 2 text-prompted detection and tracking.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GroundedSAM2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class GroundedSAM2Options : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public GroundedSAM2Options() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GroundedSAM2Options(GroundedSAM2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        VisionDim = other.VisionDim;
        DecoderDim = other.DecoderDim;
        NumVisionLayers = other.NumVisionLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads;
        PatchSize = other.PatchSize;
    }

    /// <summary>Token/embedding dimension of the vision transformer (SAM ViT hidden size). Default: 256.</summary>
    /// <value>The positive hidden width used by the image encoder.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Larger values increase representation capacity, memory use, and compute cost.</para>
    /// <para>Default 256 follows the GroundingDINO / Grounded SAM 2 detection-transformer hidden width (Ren et al., 2024, "Grounded SAM"; Liu et al., 2023, "Grounding DINO").</para>
    /// </remarks>
    public int VisionDim { get; set; } = 256;

    /// <summary>Token/embedding dimension of the mask-decoder transformer. Default: 256.</summary>
    /// <value>The positive hidden width used by the mask decoder.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Larger values give the decoder more capacity to turn detected regions into masks, at higher memory and compute cost.</para>
    /// <para>Default 256 matches <see cref="VisionDim"/> so encoder tokens flow into the decoder without a projection, per the Grounded SAM 2 / SAM 2 mask-decoder configuration (Ravi et al., 2024, "SAM 2").</para>
    /// </remarks>
    public int DecoderDim { get; set; } = 256;

    /// <summary>Number of image-encoder transformer blocks. Default: 6.</summary>
    /// <value>The positive number of stacked encoder transformer blocks.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More layers let the encoder learn richer image features but make training and inference slower.</para>
    /// <para>Default 6 follows the GroundingDINO feature-enhancer depth used by Grounded SAM 2 (Liu et al., 2023, "Grounding DINO").</para>
    /// </remarks>
    public int NumVisionLayers { get; set; } = 6;

    /// <summary>Number of mask-decoder transformer blocks. Default: 6.</summary>
    /// <value>The positive number of stacked mask-decoder transformer blocks.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More decoder layers refine the predicted boxes/masks further but cost more compute.</para>
    /// <para>Default 6 follows the GroundingDINO cross-modal decoder depth used by Grounded SAM 2 (Liu et al., 2023, "Grounding DINO").</para>
    /// </remarks>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>Number of attention heads per transformer block. Default: 8.</summary>
    /// <value>The positive number of attention heads; must divide <see cref="VisionDim"/> and <see cref="DecoderDim"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Attention heads let the model attend to several places at once; each head splits the hidden width, so this must divide the dimension evenly.</para>
    /// <para>Default 8 follows the DETR / GroundingDINO multi-head configuration used by Grounded SAM 2 (Liu et al., 2023, "Grounding DINO").</para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Patch size for the image tokenizer. The image is split into non-overlapping
    /// <c>PatchSize × PatchSize</c> patches (a strided convolution), so the token grid is
    /// <c>(H / PatchSize) × (W / PatchSize)</c>, and the mask head upsamples that token grid back to the
    /// full input resolution by this same factor. Default: 16 (SAM/ViT-style).
    /// </summary>
    /// <value>The positive patch edge length; must be &gt; 0 and should evenly divide the input height and width.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Smaller patches produce a finer token grid (more detail, more compute); larger patches are coarser and faster.</para>
    /// <para>Default 16 follows the SAM / ViT-B patch size used by Grounded SAM 2 (Ravi et al., 2024, "SAM 2"; Dosovitskiy et al., 2021, "ViT").</para>
    /// </remarks>
    public int PatchSize { get; set; } = 16;
}
