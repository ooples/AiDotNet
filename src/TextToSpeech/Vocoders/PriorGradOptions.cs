namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for PriorGrad (diffusion vocoder with data-dependent prior for adaptive noise).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the PriorGrad model. Default values follow the original paper settings.</para>
/// </remarks>
public class PriorGradOptions : VocoderOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PriorGradOptions(PriorGradOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumResBlocks = other.NumResBlocks;
    }
 public PriorGradOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; NumDiffusionSteps = 6; } public int NumResBlocks { get; set; } = 15; }
