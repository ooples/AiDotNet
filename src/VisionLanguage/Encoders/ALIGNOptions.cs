namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the ALIGN (A Large-scale ImaGe and Noisy-text embedding) model.
/// </summary>
/// <remarks>
/// <para>
/// ALIGN (Jia et al., ICML 2021) demonstrates that a simple dual-encoder contrastive model
/// can achieve strong vision-language alignment when trained on a massive noisy dataset of
/// 1.8 billion image-alt-text pairs from the web. Unlike CLIP which uses a ViT,
/// ALIGN uses an EfficientNet as its vision encoder.
/// </para>
/// <para>
/// <b>For Beginners:</b> ALIGN is similar to CLIP but uses a different image processing backbone
/// (EfficientNet instead of ViT) and was trained on a much larger but noisier dataset. The key
/// insight is that scale can compensate for noise in training data.
/// </para>
/// </remarks>
public class ALIGNOptions : ContrastiveEncoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ALIGNOptions(ALIGNOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionEmbeddingDim = other.VisionEmbeddingDim;
        VisionEncoderVariant = other.VisionEncoderVariant;
        PatchSize = other.PatchSize;
        NumVisionLayers = other.NumVisionLayers;
        NumVisionHeads = other.NumVisionHeads;
        VisionFfnMultiplier = other.VisionFfnMultiplier;
        TextEmbeddingDim = other.TextEmbeddingDim;
        TextEncoderVariant = other.TextEncoderVariant;
        MaxSequenceLength = other.MaxSequenceLength;
        VocabSize = other.VocabSize;
        NumTextLayers = other.NumTextLayers;
        NumTextHeads = other.NumTextHeads;
        ProjectionDim = other.ProjectionDim;
        Temperature = other.Temperature;
        DropoutRate = other.DropoutRate;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ImageEncoderModelPath = other.ImageEncoderModelPath;
        TextEncoderModelPath = other.TextEncoderModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        WarmUpSteps = other.WarmUpSteps;
        LabelSmoothing = other.LabelSmoothing;
        LossType = other.LossType;
        EfficientNetCompoundCoefficient = other.EfficientNetCompoundCoefficient;
        UseSqueezeExcitation = other.UseSqueezeExcitation;
    }

    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets the EfficientNet compound scaling coefficient.
    /// </summary>
    /// <remarks>
    /// <para>Controls the width, depth, and resolution scaling of the EfficientNet backbone.
    /// B0=1.0, B3=1.4, B5=2.0, B7=2.6.</para>
    /// </remarks>
    public double EfficientNetCompoundCoefficient { get; set; } = 2.6;

    /// <summary>
    /// Gets or sets whether to use squeeze-and-excitation attention blocks in EfficientNet.
    /// </summary>
    public bool UseSqueezeExcitation { get; set; } = true;

    /// <summary>
    /// Initializes default ALIGN options.
    /// </summary>
    public ALIGNOptions()
    {
        VisionEncoderVariant = ViTVariant.EfficientNetB7;
        ImageSize = 289; // ALIGN uses higher resolution with EfficientNet
        VisionEmbeddingDim = 640; // EfficientNet-B7 feature dimension
        TextEmbeddingDim = 640;
        ProjectionDim = 640;
        Temperature = 0.07;
        MaxSequenceLength = 64;
    }
}
