using AiDotNet.TextToSpeech.CodecBased;
namespace AiDotNet.TextToSpeech.FlowDiffusion;
/// <summary>Options for E2TTS.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the E2TTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class E2TTSOptions : CodecTtsOptions
{
    public E2TTSOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 1;
        CodebookSize = 4096;
        CodecFrameRate = 75;
        LLMDim = 1024;
        NumLLMLayers = 24;
    }
}
