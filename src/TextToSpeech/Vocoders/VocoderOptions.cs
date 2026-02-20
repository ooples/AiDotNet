namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// Base configuration options for neural vocoder models.
/// </summary>
public class VocoderOptions : TtsModelOptions
{
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
