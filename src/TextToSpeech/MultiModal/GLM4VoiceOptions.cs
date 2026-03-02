using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Options for GLM4Voice TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GLM4Voice model. Default values follow the original paper settings.</para>
/// </remarks>
public class GLM4VoiceOptions : CodecTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GLM4VoiceOptions(GLM4VoiceOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        FirstPacketLatencyMs = other.FirstPacketLatencyMs;
    }

    public GLM4VoiceOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    public int FirstPacketLatencyMs { get; set; } = 200;
}
