namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the SigLIP (Sigmoid Loss for Language-Image Pre-training) model.
/// </summary>
/// <remarks>
/// <para>
/// SigLIP (Zhai et al., ICCV 2023) replaces the standard softmax-based InfoNCE contrastive loss
/// with a sigmoid loss that operates on individual image-text pairs. This eliminates the need for
/// a global normalization across the batch, enabling better scaling to large batch sizes and
/// achieving 84.5% zero-shot accuracy on ImageNet with ViT-L/16@384.
/// </para>
/// <para>
/// <b>For Beginners:</b> SigLIP is like CLIP but with a smarter training strategy. Instead of
/// comparing all images with all texts at once (which gets expensive), it compares each image-text
/// pair independently. This makes it easier to train on very large batches and often gives
/// better results.
/// </para>
/// </remarks>
public class SigLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>
    /// Gets or sets the contrastive loss type (default: Sigmoid for SigLIP).
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.Sigmoid;

    /// <summary>
    /// Gets or sets the bias term for sigmoid loss.
    /// </summary>
    /// <remarks>
    /// <para>The sigmoid loss computes: -log(sigmoid(z * (sim/t + b))) where z=+1 for positive
    /// and z=-1 for negative pairs. The bias b is learnable and initialized here.</para>
    /// </remarks>
    public double SigmoidBias { get; set; } = -10.0;

    /// <summary>
    /// Gets or sets whether to use the SigLIP 2 variant with additional loss terms.
    /// </summary>
    /// <remarks>
    /// <para>SigLIP 2 (2025) adds captioning loss and self-supervised losses on top of the
    /// sigmoid contrastive loss, improving both alignment quality and multilingual support.</para>
    /// </remarks>
    public bool UseSigLIP2 { get; set; }

    /// <summary>
    /// Gets or sets whether to enable multilingual text support (SigLIP 2 feature).
    /// </summary>
    public bool Multilingual { get; set; }

    /// <summary>
    /// Gets or sets the weight for the captioning loss (SigLIP 2).
    /// </summary>
    public double CaptioningLossWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the weight for the self-supervised loss (SigLIP 2).
    /// </summary>
    public double SelfSupervisedLossWeight { get; set; } = 0.1;

    /// <summary>
    /// Initializes default SigLIP options.
    /// </summary>
    public SigLIPOptions()
    {
        // SigLIP defaults differ from CLIP
        VisionEncoderVariant = ViTVariant.ViTB16;
        ImageSize = 224;
        VisionEmbeddingDim = 768;
        ProjectionDim = 768;
        PatchSize = 16;
        Temperature = 1.0; // SigLIP uses higher temperature with sigmoid loss
    }
}
