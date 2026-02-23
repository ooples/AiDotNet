namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the RemoteCLIP model.
/// </summary>
/// <remarks>
/// <para>
/// RemoteCLIP (Liu et al., 2023) adapts CLIP for remote sensing imagery. It is fine-tuned on
/// curated remote sensing image-text datasets covering satellite imagery, aerial photography,
/// and geospatial data. It supports tasks like zero-shot scene classification, image-text
/// retrieval, and object counting in remote sensing contexts.
/// </para>
/// <para>
/// <b>For Beginners:</b> RemoteCLIP is CLIP trained specifically to understand satellite and
/// aerial images. It knows what "urban area", "agricultural land", "forest", "water body" etc.
/// look like from above, and can classify and search through satellite imagery using text queries.
/// </para>
/// </remarks>
public class RemoteCLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RemoteCLIPOptions(RemoteCLIPOptions other)
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
        Domain = other.Domain;
        GroundSampleDistance = other.GroundSampleDistance;
        UseMultiScale = other.UseMultiScale;
        PromptTemplate = other.PromptTemplate;
    }

    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets the domain specialization.
    /// </summary>
    public DomainSpecialization Domain { get; set; } = DomainSpecialization.RemoteSensing;

    /// <summary>
    /// Gets or sets the ground sample distance in meters per pixel.
    /// </summary>
    /// <remarks>
    /// <para>Typical values: 0.3m for very high resolution, 10m for Sentinel-2, 30m for Landsat.</para>
    /// </remarks>
    public double GroundSampleDistance { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use multi-scale features for remote sensing.
    /// </summary>
    public bool UseMultiScale { get; set; } = true;

    /// <summary>
    /// Gets or sets the remote sensing specific text prompt template.
    /// </summary>
    public string PromptTemplate { get; set; } = "a satellite image of {label}";

    /// <summary>
    /// Initializes default RemoteCLIP options.
    /// </summary>
    public RemoteCLIPOptions()
    {
        VisionEncoderVariant = ViTVariant.ViTB32;
        ImageSize = 224;
        VisionEmbeddingDim = 768;
        TextEmbeddingDim = 512;
        ProjectionDim = 512;
        Temperature = 0.07;
    }
}
