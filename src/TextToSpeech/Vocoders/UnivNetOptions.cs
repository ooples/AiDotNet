namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>Options for UnivNet (universal neural vocoder with multi-resolution spectrogram discriminator).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the UnivNet model. Default values follow the original paper settings.</para>
/// </remarks>
public class UnivNetOptions : VocoderOptions {
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UnivNetOptions(UnivNetOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumKernels = other.NumKernels;
        NumLMBlocks = other.NumLMBlocks;
    }
 public UnivNetOptions() { SampleRate = 24000; MelChannels = 80; HopSize = 256; } public int NumKernels { get; set; } = 3; public int NumLMBlocks { get; set; } = 5; }
