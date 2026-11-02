namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for UnivNet (universal neural vocoder with multi-resolution spectrogram discriminator).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the UnivNet model. Default values follow the original paper settings.</para>
/// </remarks>
public class UnivNetOptions : VocoderOptions
{
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

    /// <summary>Initializes a new instance with default values.</summary>
    public UnivNetOptions()
    {
        SampleRate = 24000;
        MelChannels = 80;
        HopSize = 256;
    }

    /// <summary>Gets or sets the number of location-variable kernels of the paper's configuration. The native layers do not read it.</summary>
    public int NumKernels { get; set; } = 3;
    /// <summary>Gets or sets the number of location-variable convolution blocks. The paper-faithful layers fix them, so a non-default value is reported as not applied.</summary>
    public int NumLMBlocks { get; set; } = 5;
}
