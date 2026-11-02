namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for iSTFTNet (inverse STFT-based vocoder that outputs STFT coefficients then iSTFT).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ISTFTNet model. Default values follow the original paper settings.</para>
/// </remarks>
public class ISTFTNetOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ISTFTNetOptions(ISTFTNetOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        StftWindow = other.StftWindow;
        NumUpsampleLayers = other.NumUpsampleLayers;
    }

    /// <summary>Initializes a new instance with default values.</summary>
    public ISTFTNetOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
    }

    /// <summary>Gets or sets the inverse-STFT window length in samples; the network predicts StftWindow / 2 + 1 frequency bins.</summary>
    public int StftWindow { get; set; } = 1024;
    /// <summary>Gets or sets the number of upsampling stages. The paper-faithful layers fix the stages, so a non-default value is reported as not applied.</summary>
    public int NumUpsampleLayers { get; set; } = 4;
}
