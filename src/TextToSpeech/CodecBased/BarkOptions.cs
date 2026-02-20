namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for Bark.</summary>
public class BarkOptions : CodecTtsOptions
{
    public BarkOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 8;
        CodebookSize = 1024;
        CodecFrameRate = 75;
        LLMDim = 768;
        NumLLMLayers = 12;
        LanguageModelName = "GPT";
    }
}
