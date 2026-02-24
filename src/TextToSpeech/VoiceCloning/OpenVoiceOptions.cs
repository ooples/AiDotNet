using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for OpenVoice TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the OpenVoice model. Default values follow the original paper settings.</para>
/// </remarks>
public class OpenVoiceOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OpenVoiceOptions(OpenVoiceOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SpeakerEmbeddingDim = other.SpeakerEmbeddingDim;
        NumToneColorLayers = other.NumToneColorLayers;
    }

    public OpenVoiceOptions()
    {
        NumEncoderLayers = 6;
        NumDecoderLayers = 4;
        NumHeads = 2;
        DropoutRate = 0.1;
    }

    public int SpeakerEmbeddingDim { get; set; } = 256;
    public int NumToneColorLayers { get; set; } = 3;
}
