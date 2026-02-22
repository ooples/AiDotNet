namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for OrpheusTTS.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the OrpheusTTS model. Default values follow the original paper settings.</para>
/// </remarks>
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
