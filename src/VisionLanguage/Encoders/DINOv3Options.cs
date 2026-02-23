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
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DINOv3Options(DINOv3Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        EmbeddingDim = other.EmbeddingDim;
        PatchSize = other.PatchSize;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        FfnMultiplier = other.FfnMultiplier;
        DropoutRate = other.DropoutRate;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        NumRegisterTokens = other.NumRegisterTokens;
        UseRegisterTokens = other.UseRegisterTokens;
        DINOHeadDim = other.DINOHeadDim;
        IBOTMaskRatio = other.IBOTMaskRatio;
        UseSwiGLU = other.UseSwiGLU;
    }

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
