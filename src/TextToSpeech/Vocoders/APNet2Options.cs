namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for APNet2 (Du et al. 2023): APNet with ConvNeXt v2 amplitude and phase predictors, linear
/// anti-wrapping phase losses, and a hinge GAN loss against a multi-period and a multi-resolution discriminator.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (§3–4.1): 22.05 kHz audio; 80-band mel spectrograms (0–8 kHz) and spectra from a 1024-point
/// FFT with a 1024-sample window and a 256-sample shift; 8 ConvNeXt v2 blocks of 512 channels with depth-wise kernel 7
/// and a 1536-wide point-wise expansion in both predictors; APNet's loss weights; AdamW (0.8, 0.99, weight decay 0.01) at
/// 2e-4 decayed by 0.999 per epoch; batch 16 and 8,192-sample segments.
/// </para>
/// <para>What the paper leaves open follows redmist328/APNet2: input and output convolutions with kernel 7 and every
/// convolution and linear map initialized from N(0, 0.02²) truncated at ±2 with zero biases; the MRD of three
/// resolutions (FFT, hop, window) = (1024, 256, 1024), (2048, 512, 2048), (512, 128, 512) over unwindowed magnitude
/// spectrograms with 64-channel strided 2-D convolutions and leaky ReLU 0.1; the feature-matching weight 1, 0.1 on the MRD's
/// feature maps; 750 updates per epoch (12,000 clips at batch 16). The hinge GAN losses average the MPD's and MRD's
/// sub-discriminators equally, as the paper's Eq. 2–3 state (the reference sums them and weights the MRD's by 0.1).</para>
/// <para><b>For Beginners:</b> These options configure the APNet2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class APNet2Options : AmplitudePhaseVocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public APNet2Options(APNet2Options other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        ConvNeXtChannels = other.ConvNeXtChannels;
        ConvNeXtIntermediateChannels = other.ConvNeXtIntermediateChannels;
        NumConvNeXtBlocks = other.NumConvNeXtBlocks;
        DepthwiseKernelSize = other.DepthwiseKernelSize;
        ResolutionFftSizes = (int[])other.ResolutionFftSizes.Clone();
        ResolutionHopSizes = (int[])other.ResolutionHopSizes.Clone();
        ResolutionWindowSizes = (int[])other.ResolutionWindowSizes.Clone();
        ResolutionDiscriminatorChannels = other.ResolutionDiscriminatorChannels;
        ResolutionFeatureMatchingWeight = other.ResolutionFeatureMatchingWeight;
        FeatureMatchingWeight = other.FeatureMatchingWeight;
    }

    /// <summary>Creates the paper's APNet2 configuration.</summary>
    public APNet2Options()
    {
        SampleRate = 22050;
        HopSize = 256;
        WindowSize = 1024;
        SegmentSize = 8192;
        UpdatesPerEpoch = 750;
    }

    /// <summary>Gets or sets the ConvNeXt v2 channels (512).</summary>
    public int ConvNeXtChannels { get; set; } = 512;

    /// <summary>Gets or sets the ConvNeXt v2 point-wise expansion (1536).</summary>
    public int ConvNeXtIntermediateChannels { get; set; } = 1536;

    /// <summary>Gets or sets the ConvNeXt v2 blocks per predictor (8).</summary>
    public int NumConvNeXtBlocks { get; set; } = 8;

    /// <summary>Gets or sets the depth-wise convolution's kernel (7).</summary>
    public int DepthwiseKernelSize { get; set; } = 7;

    /// <summary>Gets or sets the MRD's FFT sizes (1024, 2048, 512).</summary>
    public int[] ResolutionFftSizes { get; set; } = [1024, 2048, 512];

    /// <summary>Gets or sets the MRD's hops (256, 512, 128).</summary>
    public int[] ResolutionHopSizes { get; set; } = [256, 512, 128];

    /// <summary>Gets or sets the MRD's windows (1024, 2048, 512).</summary>
    public int[] ResolutionWindowSizes { get; set; } = [1024, 2048, 512];

    /// <summary>Gets or sets the MRD's channels (64).</summary>
    public int ResolutionDiscriminatorChannels { get; set; } = 64;

    /// <summary>Gets or sets the weight of the MRD's feature-matching terms relative to the MPD's (0.1).</summary>
    public double ResolutionFeatureMatchingWeight { get; set; } = 0.1;

    /// <summary>Gets or sets the feature-matching weight (1).</summary>
    public double FeatureMatchingWeight { get; set; } = 1;
}
