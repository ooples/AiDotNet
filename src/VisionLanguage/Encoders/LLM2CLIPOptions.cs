namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Configuration options for the LLM2CLIP model.
/// </summary>
/// <remarks>
/// <para>
/// LLM2CLIP (Huang et al., 2024) from Microsoft enhances CLIP's text encoder by replacing it with
/// an LLM (such as LLaMA or Mistral). The LLM text embeddings provide richer semantic understanding,
/// especially for complex captions and long-form text. The LLM is fine-tuned with contrastive learning
/// to align with the existing CLIP vision encoder.
/// </para>
/// <para>
/// <b>For Beginners:</b> LLM2CLIP upgrades CLIP's text understanding by replacing its simple text model
/// with a powerful language model (like ChatGPT's text engine). This means it can understand complex
/// descriptions, nuanced language, and longer text much better than regular CLIP.
/// </para>
/// </remarks>
public class LLM2CLIPOptions : ContrastiveEncoderOptions
{
    /// <summary>
    /// Gets or sets the contrastive loss type.
    /// </summary>
    public ContrastiveLossType LossType { get; set; } = ContrastiveLossType.InfoNCE;

    /// <summary>
    /// Gets or sets the LLM backbone name for the text encoder.
    /// </summary>
    public string LLMBackbone { get; set; } = "LLaMA-7B";

    /// <summary>
    /// Gets or sets the LLM hidden dimension.
    /// </summary>
    public int LLMHiddenDim { get; set; } = 4096;

    /// <summary>
    /// Gets or sets whether to use LoRA for efficient LLM fine-tuning.
    /// </summary>
    public bool UseLoRA { get; set; } = true;

    /// <summary>
    /// Gets or sets the LoRA rank for efficient fine-tuning.
    /// </summary>
    public int LoRARank { get; set; } = 16;

    /// <summary>
    /// Gets or sets whether to freeze the vision encoder during LLM alignment.
    /// </summary>
    public bool FreezeVisionEncoder { get; set; } = true;

    /// <summary>
    /// Initializes default LLM2CLIP options.
    /// </summary>
    public LLM2CLIPOptions()
    {
        TextEncoderVariant = TextEncoderVariant.LLMEnhanced;
        VisionEncoderVariant = ViTVariant.ViTL14;
        ImageSize = 224;
        VisionEmbeddingDim = 1024;
        TextEmbeddingDim = 4096;
        ProjectionDim = 1024;
        NumVisionLayers = 24;
        NumVisionHeads = 16;
        MaxSequenceLength = 512; // LLMs support longer sequences
        Temperature = 0.07;
    }
}
