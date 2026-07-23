namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the OpenCLIP model.
/// </summary>
/// <remarks>
/// <para>
/// OpenCLIP (Ilharco et al., 2021) is an open-source reproduction of CLIP trained on the LAION-2B
/// and LAION-5B datasets. It supports a wide range of vision encoder architectures (ViT-B/32 through
/// ViT-bigG/14) and achieves comparable or better performance than OpenAI's original CLIP.
/// </para>
/// <para>
/// <b>For Beginners:</b> OpenCLIP is essentially the same as CLIP but trained on publicly available
/// data instead of proprietary data. This makes it more transparent and reproducible while achieving
/// similar or better performance. It supports more model sizes and configurations.
/// </para>
/// </remarks>
public class OpenCLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public OpenCLIPOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OpenCLIPOptions(OpenCLIPOptions other)
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
        Dataset = other.Dataset;
        UseCoCaVariant = other.UseCoCaVariant;
        Precision = other.Precision;
        LossType = other.LossType;
    }

    /// <summary>
    /// Gets or sets the pre-training dataset.
    /// </summary>
    public PretrainingDataset Dataset { get; set; } = PretrainingDataset.LAION2B;

    /// <summary>
    /// Gets or sets whether to use the CoCa (Contrastive Captioners) variant.
    /// </summary>
    /// <remarks>
    /// <para>When enabled, the model uses a dual-loss approach combining contrastive loss
    /// with captioning loss, improving both retrieval and generation capabilities.</para>
    /// </remarks>
    public bool UseCoCaVariant { get; set; }

    /// <summary>
    /// Gets or sets the precision mode for inference.
    /// </summary>
    public ModelPrecision Precision { get; set; } = ModelPrecision.Float32;

    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.SymmetricCrossEntropy;
}
