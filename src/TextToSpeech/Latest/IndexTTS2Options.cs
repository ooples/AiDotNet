using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Options for IndexTTS2 TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the IndexTTS2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class IndexTTS2Options : CodecTtsOptions
{
    public IndexTTS2Options()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }
}
