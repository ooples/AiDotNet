namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for Dia.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Dia model. Default values follow the original paper settings.</para>
/// </remarks>
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
