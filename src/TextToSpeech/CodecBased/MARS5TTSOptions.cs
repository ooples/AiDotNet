namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for MARS5TTS.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MARS5TTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class MARS5TTSOptions : CodecTtsOptions
{
    public MARS5TTSOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 8;
        CodebookSize = 1024;
        CodecFrameRate = 75;
        LLMDim = 1536;
        NumLLMLayers = 24;
    }
}
