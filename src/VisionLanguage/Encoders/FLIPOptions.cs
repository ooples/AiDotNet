namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the FLIP (Fast Language-Image Pre-training) model.
/// </summary>
/// <remarks>
/// <para>
/// FLIP (Li et al., 2022) accelerates CLIP training by randomly masking 50-75% of image patches
/// during training. The unmasked patches are processed by the vision encoder, reducing computation
/// while maintaining strong zero-shot performance. At inference time, all patches are used.
/// </para>
/// <para>
/// <b>For Beginners:</b> FLIP makes CLIP training much faster by a simple trick: during training,
/// it randomly hides most of the image (like covering parts of a puzzle) and only processes the
/// visible pieces. This speeds up training 2-4x with minimal performance loss, because the model
/// learns to understand images from partial information.
/// </para>
/// </remarks>
public class FLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets the masking ratio for image patches during training.
    /// </summary>
    /// <remarks>
    /// <para>Controls what fraction of image patches are masked (hidden) during training.
    /// 0.5 = 50% masked (2x speedup), 0.75 = 75% masked (4x speedup). Default: 0.5.</para>
    /// </remarks>
    public double MaskingRatio { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether to use unmasked tuning after pre-training.
    /// </summary>
    /// <remarks>
    /// <para>Fine-tunes the model with all patches visible after masked pre-training to close
    /// the train/inference gap.</para>
    /// </remarks>
    public bool UseUnmaskedTuning { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of unmasked tuning epochs.
    /// </summary>
    public int UnmaskedTuningEpochs { get; set; } = 1;

    /// <summary>
    /// Initializes default FLIP options.
    /// </summary>
    public FLIPOptions()
    {
        VisionEncoderVariant = ViTVariant.ViTL14;
        ImageSize = 224;
        VisionEmbeddingDim = 1024;
        TextEmbeddingDim = 768;
        ProjectionDim = 768;
        NumVisionLayers = 24;
        NumVisionHeads = 16;
        Temperature = 0.07;
    }
}
