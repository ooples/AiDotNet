namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the DFN-CLIP (Data Filtering Networks for CLIP) model.
/// </summary>
/// <remarks>
/// <para>
/// DFN-CLIP (Fang et al., 2023) from Apple uses a small CLIP model as a "data filtering network"
/// to score and select high-quality image-text pairs from a large noisy pool. The filtered data
/// is then used to train a larger CLIP model, achieving 83.0% zero-shot on ImageNet with ViT-H/14.
/// </para>
/// <para>
/// <b>For Beginners:</b> DFN-CLIP uses a clever bootstrapping trick: first train a small model,
/// then use that model to find the best training data from a huge noisy pool, and finally train
/// a bigger model on just the good data. It's like having a teacher pre-screen study materials.
/// </para>
/// </remarks>
public class DFNCLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets the pre-training dataset.
    /// </summary>
    public PretrainingDataset Dataset { get; set; } = PretrainingDataset.DFNFiltered;

    /// <summary>
    /// Gets or sets the filtering threshold score for data selection.
    /// </summary>
    /// <remarks>
    /// <para>Image-text pairs with CLIP similarity below this threshold are filtered out.
    /// Higher values produce cleaner but smaller datasets.</para>
    /// </remarks>
    public double FilteringThreshold { get; set; } = 0.28;

    /// <summary>
    /// Gets or sets the target dataset size after filtering (in millions).
    /// </summary>
    public int FilteredDatasetSizeMillions { get; set; } = 2000;

    /// <summary>
    /// Initializes default DFN-CLIP options.
    /// </summary>
    public DFNCLIPOptions()
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
