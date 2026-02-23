namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for SigLIP-SO (Shape-Optimized SigLIP), a 400M vision encoder.
/// </summary>
/// <remarks>
/// <para>SigLIP-SO (Zhai et al., 2023) is a shape-optimized variant of SigLIP designed for
/// use as a standalone vision encoder in VLMs. The SO-400M version (ViT-SO400M/14) uses
/// an optimized width/depth ratio for the 400M parameter budget, producing high-quality
/// visual features widely adopted in LLaVA, PaliGemma, and other VLMs.</para>
/// </remarks>
public class SigLIPSOOptions : VisionEncoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SigLIPSOOptions(SigLIPSOOptions other)
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
        NumOutputTokens = other.NumOutputTokens;
        UseSigmoidLoss = other.UseSigmoidLoss;
        TrainingResolution = other.TrainingResolution;
    }

    /// <summary>
    /// Gets or sets the number of output feature tokens after pooling.
    /// </summary>
    public int NumOutputTokens { get; set; } = 729;

    /// <summary>
    /// Gets or sets whether to use sigmoid loss (SigLIP) for any fine-tuning.
    /// </summary>
    public bool UseSigmoidLoss { get; set; } = true;

    /// <summary>
    /// Gets or sets the resolution at which the model was trained.
    /// </summary>
    public int TrainingResolution { get; set; } = 384;

    public SigLIPSOOptions()
    {
        ImageSize = 384;
        EmbeddingDim = 1152;
        PatchSize = 14;
        NumLayers = 27;
        NumHeads = 16;
        ImageMean = [0.5, 0.5, 0.5];
        ImageStd = [0.5, 0.5, 0.5];
    }
}
