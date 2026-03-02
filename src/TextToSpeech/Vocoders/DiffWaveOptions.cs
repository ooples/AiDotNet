namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for DiffWave (diffusion-based vocoder using denoising score matching).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the DiffWave model. Default values follow the original paper settings.</para>
/// </remarks>
public class DiffWaveOptions : VocoderOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DiffWaveOptions(DiffWaveOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumResLayers = other.NumResLayers;
        ResChannels = other.ResChannels;
    }
 public DiffWaveOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; NumDiffusionSteps = 50; } public int NumResLayers { get; set; } = 30; public int ResChannels { get; set; } = 64; }
