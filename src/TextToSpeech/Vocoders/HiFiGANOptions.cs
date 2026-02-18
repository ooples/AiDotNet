namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for HiFi-GAN (high-fidelity GAN-based vocoder with multi-receptive field fusion).</summary>
public class HiFiGANOptions : VocoderOptions
{
    public HiFiGANOptions() { UpsampleRates = [8, 8, 2, 2]; UpsampleKernelSizes = [16, 16, 4, 4]; UpsampleInitialChannels = 512; ResblockKernelSizes = [3, 7, 11]; SampleRate = 22050; MelChannels = 80; HopSize = 256; }

    /// <summary>Gets or sets the resblock dilation sizes for each kernel.</summary>
    public int[][] ResblockDilationSizes { get; set; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];
}
