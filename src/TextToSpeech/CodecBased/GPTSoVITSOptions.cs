namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for GPTSoVITS.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GPTSoVITS model. Default values follow the original paper settings.</para>
/// </remarks>
public class GPTSoVITSOptions : CodecTtsOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public GPTSoVITSOptions()
    {
        SampleRate = 32000;
        NumCodebooks = 1;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        LLMDim = 512;
        NumLLMLayers = 12;
    }
}
