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
