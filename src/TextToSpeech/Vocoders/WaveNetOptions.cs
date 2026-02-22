namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for WaveNet (autoregressive dilated causal convolution vocoder).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the WaveNet model. Default values follow the original paper settings.</para>
/// </remarks>
public class WaveNetOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public WaveNetOptions(WaveNetOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumDilatedLayers = other.NumDilatedLayers;
        ResidualChannels = other.ResidualChannels;
        SkipChannels = other.SkipChannels;
        MuLawLevels = other.MuLawLevels;
    }

    public WaveNetOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 256; }
    /// <summary>Gets or sets the number of dilated causal convolution layers.</summary>
    public int NumDilatedLayers { get; set; } = 30;
    /// <summary>Gets or sets the residual channel count.</summary>
    public int ResidualChannels { get; set; } = 64;
    /// <summary>Gets or sets the skip channel count.</summary>
    public int SkipChannels { get; set; } = 256;
    /// <summary>Gets or sets the number of mu-law quantization levels.</summary>
    public int MuLawLevels { get; set; } = 256;
}
