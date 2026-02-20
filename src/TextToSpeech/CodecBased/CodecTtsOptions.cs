namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Base configuration options for codec-based TTS models.
/// </summary>
public class CodecTtsOptions : TtsModelOptions
{
    /// <summary>Gets or sets the number of RVQ codebooks.</summary>
    public int NumCodebooks { get; set; } = 8;

    /// <summary>Gets or sets the codebook vocabulary size.</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>Gets or sets the codec frame rate in Hz.</summary>
    public int CodecFrameRate { get; set; } = 50;

    /// <summary>Gets or sets the LLM hidden dimension.</summary>
    public int LLMDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of LLM decoder layers.</summary>
    public int NumLLMLayers { get; set; } = 12;

    /// <summary>Gets or sets the text encoder dimension.</summary>
    public int TextEncoderDim { get; set; } = 512;

    /// <summary>Gets or sets the speaker embedding dimension (for multi-speaker or cloning).</summary>
    public int SpeakerEmbeddingDim { get; set; } = 256;

    /// <summary>Gets or sets the maximum generation length in codec frames.</summary>
    public int MaxCodecFrames { get; set; } = 3000;

    /// <summary>Gets or sets the name of the underlying language model (e.g., "LLaMA", "GPT-2").</summary>
    public string LanguageModelName { get; set; } = string.Empty;
}
