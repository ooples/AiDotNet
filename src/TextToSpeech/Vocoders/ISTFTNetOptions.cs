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

    public ISTFTNetOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
    }

    public int StftWindow { get; set; } = 1024;
    public int NumUpsampleLayers { get; set; } = 4;
}
