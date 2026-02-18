namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for DINOv3, the next evolution of self-supervised vision encoders.
/// </summary>
/// <remarks>
/// <para>DINOv3 (Meta, 2025) scales self-supervised ViT to 7B parameters trained on 1.7B images,
/// outperforming SigLIP 2 on most vision benchmarks. It introduces improved training recipes
/// with enhanced data augmentation and longer training schedules.</para>
/// </remarks>
public class DINOv3Options : VisionEncoderOptions
{
    /// <summary>
    /// Gets or sets the number of register tokens.
    /// </summary>
    public int NumRegisterTokens { get; set; } = 8;

    /// <summary>
    /// Gets or sets whether to use register tokens.
    /// </summary>
    public bool UseRegisterTokens { get; set; } = true;

    /// <summary>
    /// Gets or sets the self-supervised head dimension.
    /// </summary>
    public int DINOHeadDim { get; set; } = 131072;

    /// <summary>
    /// Gets or sets the iBOT mask ratio.
    /// </summary>
    public double IBOTMaskRatio { get; set; } = 0.4;

    /// <summary>
    /// Gets or sets whether to use SwiGLU activation in the FFN.
    /// </summary>
    public bool UseSwiGLU { get; set; } = true;

    public DINOv3Options()
    {
        ImageSize = 518;
        EmbeddingDim = 1536;
        PatchSize = 14;
        NumLayers = 40;
        NumHeads = 24;
        ImageMean = [0.485, 0.456, 0.406];
        ImageStd = [0.229, 0.224, 0.225];
    }
}
