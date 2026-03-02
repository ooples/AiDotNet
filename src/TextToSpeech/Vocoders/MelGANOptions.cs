namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for MelGAN (lightweight GAN vocoder with no need for paired training data).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MelGAN model. Default values follow the original paper settings.</para>
/// </remarks>
public class MelGANOptions : VocoderOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MelGANOptions(MelGANOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumResStacks = other.NumResStacks;
        NgfBase = other.NgfBase;
    }
 public MelGANOptions() { SampleRate = 22050; MelChannels = 80; HopSize = 256; } public int NumResStacks { get; set; } = 3; public int NgfBase { get; set; } = 512; }
