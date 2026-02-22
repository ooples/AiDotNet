namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for FishSpeechV15 TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the FishSpeechV15 model. Default values follow the original paper settings.</para>
/// </remarks>
public class FishSpeechV15Options : CodecTtsOptions
{
    public FishSpeechV15Options()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
