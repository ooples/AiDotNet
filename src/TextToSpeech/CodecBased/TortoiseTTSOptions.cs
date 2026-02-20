namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for TortoiseTTS TTS model.</summary>
public class TortoiseTTSOptions : CodecTtsOptions
{
    public TortoiseTTSOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
