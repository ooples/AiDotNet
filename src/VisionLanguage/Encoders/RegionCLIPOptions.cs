namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the RegionCLIP model.
/// </summary>
/// <remarks>
/// <para>
/// RegionCLIP (Zhong et al., CVPR 2022) extends CLIP to learn region-level (object-level) visual
/// representations rather than just image-level ones. It generates region-text pairs from image
/// captions using object proposals and learns to align individual image regions with their
/// corresponding text descriptions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular CLIP understands whole images ("a dog in a park"), but RegionCLIP
/// understands specific parts of images ("the dog" vs "the park" vs "the bench"). This is useful
/// for tasks like object detection, where you need to understand individual objects in an image.
/// </para>
/// </remarks>
public class RegionCLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets the domain specialization.
    /// </summary>
    public DomainSpecialization Domain { get; set; } = DomainSpecialization.RegionLevel;

    /// <summary>
    /// Gets or sets the maximum number of region proposals per image.
    /// </summary>
    public int MaxRegionsPerImage { get; set; } = 100;

    /// <summary>
    /// Gets or sets the region feature dimension from the RoI pooling layer.
    /// </summary>
    public int RegionFeatureDim { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the RoI (Region of Interest) pooling output size.
    /// </summary>
    public int RoIPoolSize { get; set; } = 7;

    /// <summary>
    /// Gets or sets the IoU threshold for region-text assignment.
    /// </summary>
    public double RegionTextIoUThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether to use pseudo-labels generated from CLIP for region-text pairs.
    /// </summary>
    public bool UsePseudoLabels { get; set; } = true;

    /// <summary>
    /// Initializes default RegionCLIP options.
    /// </summary>
    public RegionCLIPOptions()
    {
        VisionEncoderVariant = ViTVariant.ViTB32;
        ImageSize = 224;
        VisionEmbeddingDim = 768;
        TextEmbeddingDim = 512;
        ProjectionDim = 512;
        Temperature = 0.07;
    }
}
