using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Options for KaniTTS TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the KaniTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class KaniTTSOptions : CodecTtsOptions
{
    public KaniTTSOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
