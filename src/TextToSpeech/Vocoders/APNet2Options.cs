namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for APNet2 (improved amplitude-phase network with ResNet backbone and multi-resolution STFT loss).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the APNet2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class APNet2Options : VocoderOptions
{
    public APNet2Options()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
    }

    /// <summary>
    /// Channel width of the ConvNeXt v2 backbone in both the amplitude and phase branches.
    /// </summary>
    /// <value>Defaults to 512, the paper's value.</value>
    public int ConvNeXtChannels { get; set; } = 512;

    /// <summary>
    /// Width of the point-wise expansion inside each ConvNeXt v2 block.
    /// </summary>
    /// <value>Defaults to 1536, the paper's value (3x the 512-channel width).</value>
    public int ConvNeXtIntermediateChannels { get; set; } = 1536;

    /// <summary>
    /// Number of ConvNeXt v2 blocks per branch.
    /// </summary>
    /// <value>Defaults to 8, the paper's <c>k = 8</c>.</value>
    public int NumConvNeXtBlocks { get; set; } = 8;

    /// <summary>
    /// Depth-wise convolution kernel size inside each ConvNeXt v2 block.
    /// </summary>
    /// <value>Defaults to 7, the paper's value.</value>
    public int DepthwiseKernelSize { get; set; } = 7;

    /// <summary>
    /// Window length of the STFT used to reconstruct the waveform.
    /// </summary>
    /// <value>Defaults to 1024, matching the paper's FFT size.</value>
    public int WindowLength { get; set; } = 1024;
}
