namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Tacotron 2 (location-sensitive attention with WaveNet vocoder).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Tacotron2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class Tacotron2Options : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Tacotron2Options(Tacotron2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        PrenetDim = other.PrenetDim;
        AttentionRnnDim = other.AttentionRnnDim;
        DecoderRnnDim = other.DecoderRnnDim;
        AttentionLocationChannels = other.AttentionLocationChannels;
    }

    public Tacotron2Options() { EncoderDim = 512; DecoderDim = 80; HiddenDim = 512; NumEncoderLayers = 3; NumDecoderLayers = 2; NumHeads = 1; OutputsPerStep = 1; UsePostnet = true; PostnetDim = 512; PostnetLayers = 5; }

    /// <summary>Gets or sets the prenet dimension.</summary>
    public int PrenetDim { get; set; } = 256;

    /// <summary>Gets or sets the attention RNN dimension.</summary>
    public int AttentionRnnDim { get; set; } = 1024;

    /// <summary>Gets or sets the decoder RNN dimension.</summary>
    public int DecoderRnnDim { get; set; } = 1024;

    /// <summary>Gets or sets the attention location feature channels.</summary>
    public int AttentionLocationChannels { get; set; } = 32;
}
