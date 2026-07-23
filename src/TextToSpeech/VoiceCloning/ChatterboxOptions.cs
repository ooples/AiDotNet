using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for Chatterbox.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Chatterbox model. Default values follow the original paper settings.</para>
/// </remarks>
public class ChatterboxOptions : CodecTtsOptions
{
    public ChatterboxOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 8;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        LLMDim = 1024;
        NumLLMLayers = 16;
    }

    /// <summary>Initializes a new instance by copying all Chatterbox settings.</summary>
    public ChatterboxOptions(ChatterboxOptions other)
        : base(other)
    {
    }
}
