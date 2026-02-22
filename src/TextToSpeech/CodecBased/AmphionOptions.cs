namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for Amphion.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Amphion model. Default values follow the original paper settings.</para>
/// </remarks>
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
