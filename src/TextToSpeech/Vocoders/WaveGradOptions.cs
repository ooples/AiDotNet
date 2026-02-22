namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for WaveGrad (gradient-based conditional waveform diffusion).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the WaveGrad model. Default values follow the original paper settings.</para>
/// </remarks>
public class WaveGradOptions : VocoderOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public WaveGradOptions(WaveGradOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumDownsampleBlocks = other.NumDownsampleBlocks;
    }
 public WaveGradOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 300; NumDiffusionSteps = 50; } public int NumDownsampleBlocks { get; set; } = 4; }
