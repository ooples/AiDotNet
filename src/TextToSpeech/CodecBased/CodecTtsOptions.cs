namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Base configuration options for codec-based TTS models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the CodecTts model. Default values follow the original paper settings.</para>
/// </remarks>
public class CodecTtsOptions : TtsModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CodecTtsOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CodecTtsOptions(CodecTtsOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumCodebooks = other.NumCodebooks;
        CodebookSize = other.CodebookSize;
        CodecFrameRate = other.CodecFrameRate;
        LLMDim = other.LLMDim;
        NumLLMLayers = other.NumLLMLayers;
        TextEncoderDim = other.TextEncoderDim;
        SpeakerEmbeddingDim = other.SpeakerEmbeddingDim;
        MaxCodecFrames = other.MaxCodecFrames;
        LanguageModelName = other.LanguageModelName;
    }

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
