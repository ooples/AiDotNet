namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for IndexTTS.</summary>
public class IndexTTSOptions : CodecTtsOptions
{
    public IndexTTSOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 8;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        LLMDim = 1024;
        NumLLMLayers = 12;
    }
}
