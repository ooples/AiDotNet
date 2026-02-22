namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for OpenVoiceV2 TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the OpenVoiceV2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class OpenVoiceV2Options : VoiceCloningOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OpenVoiceV2Options(OpenVoiceV2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        EncoderDim = other.EncoderDim;
        DecoderDim = other.DecoderDim;
    }

    public OpenVoiceV2Options()
    {
        MinReferenceDurationSec = 3.0;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    public int EncoderDim { get; set; } = 512;
    public int DecoderDim { get; set; } = 192;
}
