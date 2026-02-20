namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for SeedTTS.</summary>
public class SeedTTSOptions : CodecTtsOptions
{
    public SeedTTSOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 8;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        LLMDim = 2048;
        NumLLMLayers = 24;
    }
}
