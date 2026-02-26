namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for MetaVoice1B TTS model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MetaVoice1B model. Default values follow the original paper settings.</para>
/// </remarks>
public class MetaVoice1BOptions : VoiceCloningOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MetaVoice1BOptions(MetaVoice1BOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        EncoderDim = other.EncoderDim;
        DecoderDim = other.DecoderDim;
    }

    public MetaVoice1BOptions()
    {
        MinReferenceDurationSec = 3.0;
        NumEncoderLayers = 6;
        NumLLMLayers = 12;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    public int EncoderDim { get; set; } = 512;
    public int DecoderDim { get; set; } = 1024;
}
