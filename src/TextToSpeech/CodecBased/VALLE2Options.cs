namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for VALLE2 TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VALLE2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class VALLE2Options : CodecTtsOptions
{
    public VALLE2Options()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
