namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the Vision Transformer (ViT) model.
/// </summary>
/// <remarks>
/// <para>ViT (Dosovitskiy et al., ICLR 2021) splits an image into fixed-size patches, linearly
/// embeds them, adds position embeddings, and processes the sequence through a standard Transformer
/// encoder. A [CLS] token aggregates information for classification.</para>
/// </remarks>
public class ViTOptions : VisionEncoderOptions
{
    /// <summary>
    /// Gets or sets the ViT model variant.
    /// </summary>
    public ViTVariant Variant { get; set; } = ViTVariant.ViTB16;

    /// <summary>
    /// Gets or sets whether to use a [CLS] token for classification.
    /// </summary>
    public bool UseClsToken { get; set; } = true;

    /// <summary>
    /// Gets or sets the positional embedding type.
    /// </summary>
    public PositionalEmbeddingType PositionalEmbedding { get; set; } = PositionalEmbeddingType.Learned;

    public ViTOptions()
    {
        ImageSize = 224;
        EmbeddingDim = 768;
        PatchSize = 16;
        NumLayers = 12;
        NumHeads = 12;
    }
}
