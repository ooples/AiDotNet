namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for CosyVoice.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the CosyVoice model. Default values follow the original paper settings.</para>
/// </remarks>
public class CosyVoiceOptions : CodecTtsOptions
{
    public CosyVoiceOptions()
    {
        SampleRate = 22050;
        NumCodebooks = 1;
        CodebookSize = 4096;
        CodecFrameRate = 25;
        LLMDim = 1024;
        NumLLMLayers = 14;
    }
}
