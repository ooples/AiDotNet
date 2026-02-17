namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the EVA-CLIP model.
/// </summary>
/// <remarks>
/// <para>
/// EVA-CLIP (Sun et al., 2023) combines the EVA-02 Vision Transformer backbone with CLIP-style
/// contrastive learning. EVA-02 uses masked image modeling (MIM) pre-training to produce a
/// stronger vision encoder, which is then used for contrastive image-text alignment. The largest
/// EVA-CLIP variant (ViT-E/14, 4.4B params) achieves 82.0% zero-shot on ImageNet.
/// </para>
/// <para>
/// <b>For Beginners:</b> EVA-CLIP is like CLIP but with a better-trained image encoder. The EVA
/// vision model first learns to understand images on its own (by filling in masked patches), and
/// then learns to connect images with text. This two-step process produces stronger results.
/// </para>
/// </remarks>
public class EVACLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets whether to use EVA-02 (improved) backbone instead of EVA-01.
    /// </summary>
    public bool UseEVA02 { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use rotary position embeddings (RoPE) in the vision encoder.
    /// </summary>
    public bool UseRoPE { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use SwiGLU activation in the FFN layers.
    /// </summary>
    public bool UseSwiGLU { get; set; } = true;

    /// <summary>
    /// Gets or sets the MIM pre-training method used for the vision encoder.
    /// </summary>
    public string MIMPretraining { get; set; } = "EVA-02-CLIP-E";

    /// <summary>
    /// Initializes default EVA-CLIP options.
    /// </summary>
    public EVACLIPOptions()
    {
        VisionEncoderVariant = ViTVariant.ViTE14;
        ImageSize = 224;
        VisionEmbeddingDim = 1024;
        TextEmbeddingDim = 768;
        ProjectionDim = 1024;
        NumVisionLayers = 24;
        NumVisionHeads = 16;
        PatchSize = 14;
        Temperature = 0.07;
    }
}
