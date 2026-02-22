namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the BiomedCLIP model.
/// </summary>
/// <remarks>
/// <para>
/// BiomedCLIP (Zhang et al., 2023) from Microsoft is a CLIP model fine-tuned on PMC-15M, a dataset
/// of 15 million biomedical image-text pairs from PubMed Central. It uses a ViT-B/16 vision encoder
/// and PubMedBERT text encoder, achieving state-of-the-art zero-shot biomedical image classification.
/// </para>
/// <para>
/// <b>For Beginners:</b> BiomedCLIP is a version of CLIP trained specifically on medical and
/// biological images and their descriptions from scientific papers. This means it understands
/// medical images (X-rays, microscopy, etc.) much better than general-purpose CLIP.
/// </para>
/// </remarks>
public class BiomedCLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BiomedCLIPOptions(BiomedCLIPOptions other)
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
        Dataset = other.Dataset;
        MedicalTextEncoder = other.MedicalTextEncoder;
        UseBiomedicalAugmentations = other.UseBiomedicalAugmentations;
    }

    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets the domain specialization.
    /// </summary>
    public DomainSpecialization Domain { get; set; } = DomainSpecialization.Biomedical;

    /// <summary>
    /// Gets or sets the pre-training dataset.
    /// </summary>
    public PretrainingDataset Dataset { get; set; } = PretrainingDataset.PMC15M;

    /// <summary>
    /// Gets or sets the medical text encoder variant.
    /// </summary>
    public string MedicalTextEncoder { get; set; } = "PubMedBERT";

    /// <summary>
    /// Gets or sets whether to use domain-specific image augmentations.
    /// </summary>
    public bool UseBiomedicalAugmentations { get; set; } = true;

    /// <summary>
    /// Initializes default BiomedCLIP options.
    /// </summary>
    public BiomedCLIPOptions()
    {
        VisionEncoderVariant = ViTVariant.ViTB16;
        TextEncoderVariant = TextEncoderVariant.BERT;
        ImageSize = 224;
        VisionEmbeddingDim = 768;
        TextEmbeddingDim = 768;
        ProjectionDim = 512;
        PatchSize = 16;
        Temperature = 0.07;
    }
}
