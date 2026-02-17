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
    public string VisionBackbone { get; set; } = "Swin-Tiny";

    /// <summary>
    /// Initializes default MedCLIP options.
    /// </summary>
    public MedCLIPOptions()
    {
        TextEncoderVariant = TextEncoderVariant.BERT;
        ImageSize = 224;
        VisionEmbeddingDim = 768;
        TextEmbeddingDim = 768;
        ProjectionDim = 512;
        Temperature = 0.07;
    }
}
