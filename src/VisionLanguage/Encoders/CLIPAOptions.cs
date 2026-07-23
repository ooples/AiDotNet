namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the CLIPA (CLIP with Inverse scaling law and Accelerated training) model.
/// </summary>
/// <remarks>
/// <para>
/// CLIPA (Li et al., 2023) discovers an "inverse scaling law" for CLIP training: using shorter
/// image sequences and text lengths during the bulk of training, then fine-tuning at full resolution.
/// This reduces training cost by 7-8x while maintaining performance, enabling efficient scaling.
/// </para>
/// <para>
/// <b>For Beginners:</b> CLIPA finds that you can train CLIP much faster by using lower resolution
/// images and shorter text during most of the training, then switching to full resolution at the end.
/// This is like studying cliff notes first to get the big picture, then reading the full text for details.
/// </para>
/// </remarks>
public class CLIPAOptions : ContrastiveEncoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CLIPAOptions(CLIPAOptions other)
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
        InitialImageSize = other.InitialImageSize;
        InitialSequenceLength = other.InitialSequenceLength;
        ReducedResolutionFraction = other.ReducedResolutionFraction;
        IsFineTuningPhase = other.IsFineTuningPhase;
    }

    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets the initial (reduced) image size for the bulk of training.
    /// </summary>
    public int InitialImageSize { get; set; } = 112;

    /// <summary>
    /// Gets or sets the initial (reduced) text sequence length for the bulk of training.
    /// </summary>
    public int InitialSequenceLength { get; set; } = 16;

    /// <summary>
    /// Gets or sets the fraction of training done at reduced resolution before switching to full.
    /// </summary>
    public double ReducedResolutionFraction { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets whether the model is in the fine-tuning phase (full resolution).
    /// </summary>
    public bool IsFineTuningPhase { get; set; }

    /// <summary>
    /// Initializes default CLIPA options.
    /// </summary>
    public CLIPAOptions()
    {
        VisionEncoderVariant = ViTVariant.ViTH14;
        ImageSize = 224;
        VisionEmbeddingDim = 1280;
        TextEmbeddingDim = 1024;
        ProjectionDim = 1024;
        NumVisionLayers = 32;
        NumVisionHeads = 16;
        PatchSize = 14;
        Temperature = 0.07;
    }
}
