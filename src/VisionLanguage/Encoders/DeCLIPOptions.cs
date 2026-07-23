namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the DeCLIP (Data-efficient CLIP) model.
/// </summary>
/// <remarks>
/// <para>
/// DeCLIP (Li et al., ICLR 2022) improves CLIP's data efficiency by adding self-supervised learning
/// objectives alongside the contrastive image-text loss. It uses image self-supervision (SimSiam),
/// text self-supervision (masked language modeling), and nearest-neighbor supervision to extract
/// more learning signal from each image-text pair.
/// </para>
/// <para>
/// <b>For Beginners:</b> DeCLIP learns more from less data by using multiple learning strategies
/// at once. While regular CLIP only learns from image-text pairs, DeCLIP also learns from images
/// alone (self-supervision) and from similar images nearby in the dataset. This makes it much more
/// data-efficient - achieving the same performance with fewer training examples.
/// </para>
/// </remarks>
public class DeCLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DeCLIPOptions(DeCLIPOptions other)
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
        ImageSelfSupervisedWeight = other.ImageSelfSupervisedWeight;
        TextMLMWeight = other.TextMLMWeight;
        NearestNeighborWeight = other.NearestNeighborWeight;
        NumNearestNeighbors = other.NumNearestNeighbors;
        TextMaskingRatio = other.TextMaskingRatio;
    }

    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets the weight for the image self-supervised (SimSiam) loss.
    /// </summary>
    public double ImageSelfSupervisedWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the weight for the text masked language modeling loss.
    /// </summary>
    public double TextMLMWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the weight for the nearest-neighbor supervision loss.
    /// </summary>
    public double NearestNeighborWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of nearest neighbors for supervision.
    /// </summary>
    public int NumNearestNeighbors { get; set; } = 5;

    /// <summary>
    /// Gets or sets the masking ratio for text masked language modeling.
    /// </summary>
    public double TextMaskingRatio { get; set; } = 0.15;

    /// <summary>
    /// Initializes default DeCLIP options.
    /// </summary>
    public DeCLIPOptions()
    {
        VisionEncoderVariant = ViTVariant.ViTB16;
        ImageSize = 224;
        VisionEmbeddingDim = 768;
        TextEmbeddingDim = 512;
        ProjectionDim = 512;
        Temperature = 0.07;
    }
}
