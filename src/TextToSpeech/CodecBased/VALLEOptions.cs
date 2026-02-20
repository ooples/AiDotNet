namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for VALL-E (neural codec language model with AR + NAR transformers for zero-shot TTS).</summary>
public class VALLEOptions : CodecTtsOptions
{
    public VALLEOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 8;
        CodebookSize = 1024;
        CodecFrameRate = 75;
        LLMDim = 1024;
        NumLLMLayers = 12;
    }
}
