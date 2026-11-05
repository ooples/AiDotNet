namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for Multi-band MelGAN (multi-band signal decomposition for faster vocoding).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MultiBandMelGAN model. Default values follow the original paper settings.</para>
/// </remarks>
public class MultiBandMelGANOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MultiBandMelGANOptions(MultiBandMelGANOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumBands = other.NumBands;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public MultiBandMelGANOptions()
    {
        SampleRate = 24000;
        MelChannels = 80;
        HopSize = 300;
    }

    /// <summary>Gets or sets the number of sub-bands. The paper-faithful layers fix them, so a non-default value is reported as not applied.</summary>
    public int NumBands { get; set; } = 4;
}
