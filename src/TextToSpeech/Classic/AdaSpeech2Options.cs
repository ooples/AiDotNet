namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for AdaSpeech 2 (adaptive TTS with untranscribed speech data).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the AdaSpeech2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class AdaSpeech2Options : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AdaSpeech2Options(AdaSpeech2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Mel2PhDim = other.Mel2PhDim;
    }

    public AdaSpeech2Options() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 4; NumDecoderLayers = 4; NumHeads = 2; }

    /// <summary>Gets or sets the mel-to-phoneme encoder dimension for untranscribed data.</summary>
    public int Mel2PhDim { get; set; } = 256;
}
