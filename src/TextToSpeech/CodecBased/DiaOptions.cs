namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for Dia.</summary>
public class DiaOptions : CodecTtsOptions
{
    public DiaOptions()
    {
        SampleRate = 44100;
        NumCodebooks = 9;
        CodebookSize = 1024;
        CodecFrameRate = 86;
        LLMDim = 3072;
        NumLLMLayers = 36;
    }
}
