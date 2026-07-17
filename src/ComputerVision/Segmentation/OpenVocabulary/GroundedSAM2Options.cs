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
    public int VisionDim { get; set; } = 256;

    /// <summary>Token/embedding dimension of the mask-decoder transformer. Default: 256.</summary>
    public int DecoderDim { get; set; } = 256;

    /// <summary>Number of image-encoder transformer blocks. Default: 6.</summary>
    public int NumVisionLayers { get; set; } = 6;

    /// <summary>Number of mask-decoder transformer blocks. Default: 6.</summary>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>Number of attention heads per transformer block. Default: 8.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Patch size for the image tokenizer. The image is split into non-overlapping
    /// <c>PatchSize × PatchSize</c> patches (a strided convolution), so the token grid is
    /// <c>(H / PatchSize) × (W / PatchSize)</c>, and the mask head upsamples that token grid back to the
    /// full input resolution by this same factor. Default: 16 (SAM/ViT-style).
    /// </summary>
    public int PatchSize { get; set; } = 16;
}
