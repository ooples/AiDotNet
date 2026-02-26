namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for AlignTTS (alignment-free non-autoregressive TTS with mix density network).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the AlignTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class AlignTTSOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AlignTTSOptions(AlignTTSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumMixtures = other.NumMixtures;
    }

    public AlignTTSOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 6; NumDecoderLayers = 6; NumHeads = 2; }

    /// <summary>Gets or sets the number of mixture components for the alignment loss.</summary>
    public int NumMixtures { get; set; } = 1;
}
