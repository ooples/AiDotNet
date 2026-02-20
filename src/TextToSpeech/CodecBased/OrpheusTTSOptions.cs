namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for OrpheusTTS.</summary>
public class OrpheusTTSOptions : CodecTtsOptions
{
    public OrpheusTTSOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 3;
        CodebookSize = 4096;
        CodecFrameRate = 12;
        LLMDim = 3200;
        NumLLMLayers = 28;
        LanguageModelName = "LLaMA";
    }
}
