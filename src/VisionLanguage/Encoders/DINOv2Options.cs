namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the DINOv2 self-supervised vision encoder.
/// </summary>
/// <remarks>
/// <para>DINOv2 (Oquab et al., 2024) trains ViT with self-supervised objectives (iBOT + DINO head)
/// on 142M curated images (LVD-142M). It produces universal visual features without labels,
/// achieving linear-probe results competitive with fine-tuned CLIP on many benchmarks.</para>
/// </remarks>
public class DINOv2Options : VisionEncoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DINOv2Options(DINOv2Options other)
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
    }

    /// <summary>
    /// Gets or sets the number of register tokens appended to the patch sequence.
    /// </summary>
    public int NumRegisterTokens { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to use register tokens (DINOv2 with registers variant).
    /// </summary>
    public bool UseRegisterTokens { get; set; } = true;

    /// <summary>
    /// Gets or sets the self-supervised head dimension for DINO loss.
    /// </summary>
    public int DINOHeadDim { get; set; } = 65536;

    /// <summary>
    /// Gets or sets the iBOT mask ratio for masked image modeling.
    /// </summary>
    public double IBOTMaskRatio { get; set; } = 0.3;

    public DINOv2Options()
    {
        ImageSize = 518;
        EmbeddingDim = 768;
        PatchSize = 14;
        NumLayers = 12;
        NumHeads = 12;
        ImageMean = [0.485, 0.456, 0.406];
        ImageStd = [0.229, 0.224, 0.225];
    }
}
