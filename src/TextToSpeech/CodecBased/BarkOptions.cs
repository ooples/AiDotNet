namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for Bark.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Bark model. Default values follow the original paper settings.</para>
/// </remarks>
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
