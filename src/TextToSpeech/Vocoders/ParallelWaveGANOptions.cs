namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for Parallel WaveGAN (non-autoregressive GAN vocoder with multi-resolution STFT loss).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ParallelWaveGAN model. Default values follow the original paper settings.</para>
/// </remarks>
public class ParallelWaveGANOptions : VocoderOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ParallelWaveGANOptions(ParallelWaveGANOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumResBlocks = other.NumResBlocks;
        ResChannels = other.ResChannels;
    }
 public ParallelWaveGANOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 300; } public int NumResBlocks { get; set; } = 30; public int ResChannels { get; set; } = 64; }
