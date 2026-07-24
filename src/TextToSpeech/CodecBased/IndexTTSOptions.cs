namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for IndexTTS.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the IndexTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class IndexTTSOptions : CodecTtsOptions
{
    public IndexTTSOptions()
    {
        // IndexTTS v1.5 reference configuration (Zhou et al., 2025):
        // 24 kHz / 100-bin mel input, a single 8192-entry VQ codebook at
        // 25 tokens/s, and a 1024-wide 20-layer GPT with 16 attention heads.
        SampleRate = 24000;
        MelChannels = 100;
        FftSize = 1024;
        HopSize = 256;
        NumCodebooks = 1;
        CodebookSize = 8192;
        CodecFrameRate = 25;
        HiddenDim = 1024;
        LLMDim = 1024;
        NumLLMLayers = 20;
        NumHeads = 16;
        VocabSize = 12000;
        MaxTextLength = 402;
        MaxCodecFrames = 605;
        TextEncoderDim = 512;
        NumEncoderLayers = 6;
        SpeakerEmbeddingDim = 512;
        LanguageModelName = "GPT";
    }
}
