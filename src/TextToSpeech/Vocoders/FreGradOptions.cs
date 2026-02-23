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
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumResBlocks = other.NumResBlocks;
        NumWaveletLevels = other.NumWaveletLevels;
    }

    public FreGradOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        NumDiffusionSteps = 4;
    }

    public int NumResBlocks { get; set; } = 15;
    public int NumWaveletLevels { get; set; } = 3;
}
