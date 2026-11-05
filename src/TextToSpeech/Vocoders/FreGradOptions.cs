namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for FreGrad (frequency-domain diffusion vocoder with DWT sub-band processing).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the FreGrad model. Default values follow the original paper settings.</para>
/// </remarks>
public class FreGradOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FreGradOptions(FreGradOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumResBlocks = other.NumResBlocks;
        NumWaveletLevels = other.NumWaveletLevels;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public FreGradOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        NumDiffusionSteps = 4;
    }

    /// <summary>Gets or sets the number of residual blocks.</summary>
    public int NumResBlocks { get; set; } = 15;
    /// <summary>Gets or sets the number of wavelet decomposition levels the waveform is split into.</summary>
    public int NumWaveletLevels { get; set; } = 3;
}
