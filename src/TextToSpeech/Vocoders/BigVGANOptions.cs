namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for BigVGAN (universal vocoder with anti-aliased multi-periodicity composition and Snake activation).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the BigVGAN model. Default values follow the original paper settings.</para>
/// </remarks>
public class BigVGANOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BigVGANOptions(BigVGANOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        HiddenChannels = other.HiddenChannels;
        NumUpsampleLayers = other.NumUpsampleLayers;
        NumPeriods = other.NumPeriods;
        SnakeAlpha = other.SnakeAlpha;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public BigVGANOptions()
    {
        SampleRate = 24000;
        MelChannels = 100;
        HopSize = 256;
    }

    /// <summary>Gets or sets the channel width of the first upsampling block.</summary>
    public int HiddenChannels { get; set; } = 512;
    /// <summary>Gets or sets the number of upsampling blocks between the mel frames and the waveform.</summary>
    public int NumUpsampleLayers { get; set; } = 4;
    /// <summary>Gets or sets the number of periodic components summed in each anti-aliased multi-periodicity block.</summary>
    public int NumPeriods { get; set; } = 5;
    /// <summary>Gets or sets the frequency alpha of the Snake activation x + sin^2(alpha x) / alpha.</summary>
    public double SnakeAlpha { get; set; } = 1.0;
}
