using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Options for KaniTTS2 TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the KaniTTS2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class KaniTTS2Options : CodecTtsOptions
{
    public KaniTTS2Options()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
