namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the MedCLIP model.
/// </summary>
/// <remarks>
/// <para>
/// MedCLIP (Wang et al., 2022) from UCSD addresses the challenge of limited medical image-text pairs
/// by decoupling image and text inputs during contrastive learning. Instead of requiring exact
/// image-text pairs, it uses a semantic matching loss that allows any image to be paired with any
/// text description that shares the same medical concepts (e.g., diagnosis, anatomy).
/// </para>
/// <para>
/// <b>For Beginners:</b> MedCLIP is designed for medical imaging but with a clever twist: instead
/// of needing perfectly matched pairs of images and descriptions, it can learn from any image-text
/// combination that describes the same medical condition. This greatly increases the amount of
/// usable training data in the medical domain.
/// </para>
/// </remarks>
public class MedCLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MedCLIPOptions(MedCLIPOptions other)
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
        ImageMean = (double[])other.ImageMean.Clone();
        ImageStd = (double[])other.ImageStd.Clone();
        ImageEncoderModelPath = other.ImageEncoderModelPath;
        TextEncoderModelPath = other.TextEncoderModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        WarmUpSteps = other.WarmUpSteps;
        LabelSmoothing = other.LabelSmoothing;
        LossType = other.LossType;
        Domain = other.Domain;
        SemanticMatchingWeight = other.SemanticMatchingWeight;
        UseEntityExtraction = other.UseEntityExtraction;
        EntitySimilarityThreshold = other.EntitySimilarityThreshold;
        VisionBackbone = other.VisionBackbone;
        VisionModelId = other.VisionModelId;
        TextModelId = other.TextModelId;
        TokenizerDirectory = other.TokenizerDirectory;
    }

    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets the domain specialization.
    /// </summary>
    public DomainSpecialization Domain { get; set; } = DomainSpecialization.Medical;

    /// <summary>
    /// Gets or sets the weight for the semantic matching loss.
    /// </summary>
    /// <remarks>
    /// <para>MedCLIP uses a semantic matching loss alongside contrastive loss to handle
    /// decoupled image-text pairs that share medical concepts but aren't exact matches.</para>
    /// </remarks>
    public double SemanticMatchingWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use entity extraction for concept alignment.
    /// </summary>
    public bool UseEntityExtraction { get; set; } = true;

    /// <summary>
    /// Gets or sets the medical entity similarity threshold for soft labeling.
    /// </summary>
    public double EntitySimilarityThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the vision backbone used by MedCLIP.
    /// </summary>
    public string VisionBackbone { get; set; } = "ResNet50";

    /// <summary>Gets or sets the reference pretrained vision checkpoint identifier.</summary>
    public string VisionModelId { get; set; } = "torchvision/resnet50";

    /// <summary>Gets or sets the reference pretrained clinical text checkpoint identifier.</summary>
    public string TextModelId { get; set; } = "emilyalsentzer/Bio_ClinicalBERT";

    /// <summary>
    /// Gets or sets an optional local Hugging Face tokenizer directory. When supplied, native
    /// MedCLIP loads the checkpoint's WordPiece vocabulary.
    /// </summary>
    public string? TokenizerDirectory { get; set; }

    /// <summary>
    /// Initializes default MedCLIP options.
    /// </summary>
    public MedCLIPOptions()
    {
        TextEncoderVariant = TextEncoderVariant.BERT;
        ImageSize = 224;
        VisionEmbeddingDim = 2048;
        TextEmbeddingDim = 768;
        ProjectionDim = 512;
        VocabSize = 28996;
        MaxSequenceLength = 512;
        NumTextLayers = 12;
        NumTextHeads = 12;
        DropoutRate = 0.1;
        ImageMean = [0.5862785803043838, 0.5862785803043838, 0.5862785803043838];
        ImageStd = [0.27950088968644304, 0.27950088968644304, 0.27950088968644304];
        Temperature = 0.07;
    }
}
