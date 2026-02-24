using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Base options for voice cloning TTS models.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VoiceCloning model. Default values follow the original paper settings.</para>
/// </remarks>
public class VoiceCloningOptions : CodecTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VoiceCloningOptions(VoiceCloningOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        MinReferenceDurationSec = other.MinReferenceDurationSec;
    }

    public VoiceCloningOptions()
    {
        SpeakerEmbeddingDim = 256;
        NumEncoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    public double MinReferenceDurationSec { get; set; } = 3.0;
}
