namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for UniAudio TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the UniAudio model. Default values follow the original paper settings.</para>
/// </remarks>
public class UniAudioOptions : CodecTtsOptions
{
    public UniAudioOptions()
    {
        // Yang et al. 2024, Appendix model configuration: the multi-scale
        // Transformer uses three RVQ codebooks, a 24-layer global GPT and an
        // 8-layer local GPT, both 1536-wide with 12 attention heads.
        NumCodebooks = 3;
        TextEncoderDim = 1536;
        LLMDim = 1536;
        NumEncoderLayers = 24;
        NumLLMLayers = 8;
        NumHeads = 12;
        DropoutRate = 0.1;
    }

    /// <summary>Initializes an independent copy of another UniAudio configuration.</summary>
    public UniAudioOptions(UniAudioOptions other)
        : base(other)
    {
    }
}
