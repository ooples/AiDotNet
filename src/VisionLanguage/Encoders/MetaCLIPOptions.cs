namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the MetaCLIP model.
/// </summary>
/// <remarks>
/// <para>
/// MetaCLIP (Xu et al., 2023) improves CLIP training through metadata-driven data curation.
/// Instead of using raw web-scraped data, MetaCLIP balances the training distribution by using
/// metadata from WordNet and Wikipedia to ensure diverse, high-quality image-text pairs. This
/// data-centric approach achieves 70.8% zero-shot on ImageNet with ViT-B/16 on 400M pairs.
/// </para>
/// <para>
/// <b>For Beginners:</b> MetaCLIP focuses on training data quality rather than model architecture.
/// It carefully selects and balances the image-text pairs used for training, making sure the model
/// sees a diverse and representative set of concepts. Better data leads to a better model.
/// </para>
/// </remarks>
public class MetaCLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets the pre-training dataset.
    /// </summary>
    public PretrainingDataset Dataset { get; set; } = PretrainingDataset.MetaCLIPCurated;

    /// <summary>
    /// Gets or sets the maximum number of entries per metadata concept for data balancing.
    /// </summary>
    public int MaxEntriesPerConcept { get; set; } = 20000;

    /// <summary>
    /// Gets or sets whether to use sub-string matching for metadata alignment.
    /// </summary>
    public bool UseSubStringMatching { get; set; } = true;

    /// <summary>
    /// Initializes default MetaCLIP options.
    /// </summary>
    public MetaCLIPOptions()
    {
        VisionEncoderVariant = ViTVariant.ViTB16;
        ImageSize = 224;
        VisionEmbeddingDim = 768;
        TextEmbeddingDim = 512;
        ProjectionDim = 512;
        Temperature = 0.07;
    }
}
