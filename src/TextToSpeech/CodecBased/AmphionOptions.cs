namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for Amphion.</summary>
public class AmphionOptions : CodecTtsOptions
{
    public AmphionOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 8;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        LLMDim = 1024;
        NumLLMLayers = 12;
    }
}
