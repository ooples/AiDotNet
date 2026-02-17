namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for Florence-2, Microsoft's unified vision foundation model.
/// </summary>
/// <remarks>
/// <para>Florence-2 (Xiao et al., 2024) is a lightweight sequence-to-sequence vision foundation model
/// (0.23B-0.77B) that handles captioning, object detection, grounding, OCR, and segmentation through
/// a unified prompt-based approach. It uses DaViT as the vision encoder and a multi-task decoder.</para>
/// </remarks>
public class Florence2Options : VisionEncoderOptions
{
    /// <summary>
    /// Gets or sets the maximum number of output tokens for the text decoder.
    /// </summary>
    public int MaxOutputTokens { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the number of decoder layers.
    /// </summary>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>
    /// Gets or sets the decoder embedding dimension.
    /// </summary>
    public int DecoderEmbeddingDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of decoder attention heads.
    /// </summary>
    public int NumDecoderHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the vocabulary size for the text decoder.
    /// </summary>
    public int VocabSize { get; set; } = 51289;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    public Florence2ModelSize ModelSize { get; set; } = Florence2ModelSize.Base;

    /// <summary>
    /// Gets or sets whether to use the DaViT (Dual Attention Vision Transformer) backbone.
    /// </summary>
    public bool UseDaViT { get; set; } = true;

    public Florence2Options()
    {
        ImageSize = 768;
        EmbeddingDim = 768;
        PatchSize = 16;
        NumLayers = 12;
        NumHeads = 12;
    }
}
