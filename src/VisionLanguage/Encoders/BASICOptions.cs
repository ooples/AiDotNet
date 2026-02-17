namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the BASIC (Batch-wise Alignment of Scaled Image-text Contrastive) model.
/// </summary>
/// <remarks>
/// <para>
/// BASIC (Pham et al., 2022) scales up the dual-encoder contrastive learning paradigm using a CoAtNet
/// (hybrid CNN-Transformer) as the vision encoder and a large text transformer, training on 6.6 billion
/// image-text pairs. It achieves 85.7% zero-shot accuracy on ImageNet.
/// </para>
/// <para>
/// <b>For Beginners:</b> BASIC is a scaled-up version of CLIP that uses a hybrid CNN-Transformer for images
/// (combining the strengths of both architectures) and was trained on an even larger dataset. The name
/// "BASIC" highlights that even a basic contrastive approach works extremely well at sufficient scale.
/// </para>
/// </remarks>
public class BASICOptions : ContrastiveEncoderOptions
{
    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets the CoAtNet variant to use as the vision encoder.
    /// </summary>
    public int CoAtNetVariant { get; set; } = 7;

    /// <summary>
    /// Gets or sets whether to use gradient checkpointing for memory-efficient training.
    /// </summary>
    public bool UseGradientCheckpointing { get; set; }

    /// <summary>
    /// Initializes default BASIC options.
    /// </summary>
    public BASICOptions()
    {
        VisionEncoderVariant = ViTVariant.CoAtNet;
        ImageSize = 224;
        VisionEmbeddingDim = 1536; // CoAtNet-7 feature dimension
        TextEmbeddingDim = 768;
        ProjectionDim = 1024;
        Temperature = 0.07;
        NumVisionLayers = 24;
        NumVisionHeads = 24;
    }
}
