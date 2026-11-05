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
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        FirstPacketLatencyMs = other.FirstPacketLatencyMs;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public GLM4VoiceOptions()
    {
        TextEncoderDim = 256;
        LLMDim = 1024;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    /// <summary>Gets or sets the target delay, in milliseconds, before the first audio packet of a streamed reply. The model reports it and does not schedule by it.</summary>
    public int FirstPacketLatencyMs { get; set; } = 200;
}
