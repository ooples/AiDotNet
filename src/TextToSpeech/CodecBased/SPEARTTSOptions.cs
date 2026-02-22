namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for SPEARTTS TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SPEARTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class SPEARTTSOptions : CodecTtsOptions
{
    public SPEARTTSOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
