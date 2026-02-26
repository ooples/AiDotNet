namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// Base configuration options for neural vocoder models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Vocoder model. Default values follow the original paper settings.</para>
/// </remarks>
public class VocoderOptions : TtsModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public VocoderOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VocoderOptions(VocoderOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        UpsampleRates = other.UpsampleRates;
        UpsampleKernelSizes = other.UpsampleKernelSizes;
        UpsampleInitialChannels = other.UpsampleInitialChannels;
        ResblockKernelSizes = other.ResblockKernelSizes;
        NumDiffusionSteps = other.NumDiffusionSteps;
    }

    /// <summary>Gets or sets the upsample rates for each upsampling block.</summary>
    public int[] UpsampleRates { get; set; } = [8, 8, 2, 2];

    /// <summary>Gets or sets the kernel sizes for each upsampling block.</summary>
    public int[] UpsampleKernelSizes { get; set; } = [16, 16, 4, 4];

    /// <summary>Gets or sets the initial upsample channel count.</summary>
    public int UpsampleInitialChannels { get; set; } = 512;

    /// <summary>Gets or sets the resblock kernel sizes.</summary>
    public int[] ResblockKernelSizes { get; set; } = [3, 7, 11];

    /// <summary>Gets or sets the number of diffusion steps (for diffusion vocoders).</summary>
    public int NumDiffusionSteps { get; set; } = 50;
}
