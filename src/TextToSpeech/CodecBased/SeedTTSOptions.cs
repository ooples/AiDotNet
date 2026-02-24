namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for SeedTTS.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SeedTTS model. Default values follow the original paper settings.</para>
/// </remarks>
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
