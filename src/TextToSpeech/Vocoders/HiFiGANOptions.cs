namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for HiFi-GAN (Kong et al. 2020).</summary>
/// <remarks>
/// <para>
/// Defaults are HiFi-GAN V1 (Table 5) with the values the paper leaves to its reference implementation (jik876/hifi-gan
/// <c>config_v1.json</c>, <c>meldataset.py</c>): upsampling rates 8, 8, 2, 2 with kernels 16, 16, 4, 4 from 512
/// channels; residual block type 1 with kernels 3, 7, 11 and dilations (1, 3, 5) each; discriminator periods 2, 3, 5, 7,
/// 11 and three scales; 8192-sample training segments; 22.05 kHz, 1024-point FFT and window, hop 256, 80 mel bins up to
/// 8 kHz (the loss mel spans the full band); λ_fm = 2, λ_mel = 45; AdamW at 2e-4 decayed 0.999 per epoch.
/// </para>
/// <para><b>For Beginners:</b> These options configure the HiFiGAN model. Default values follow the original paper settings.</para>
/// </remarks>
public class HiFiGANOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public HiFiGANOptions(HiFiGANOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        ResblockDilationSizes = other.ResblockDilationSizes.Select(d => (int[])d.Clone()).ToArray();
        ResblockType = other.ResblockType;
        DiscriminatorPeriods = (int[])other.DiscriminatorPeriods.Clone();
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor;
        SegmentSize = other.SegmentSize;
        WindowSize = other.WindowSize;
        MelMaxFrequency = other.MelMaxFrequency;
        FeatureMatchingWeight = other.FeatureMatchingWeight;
        MelLossWeight = other.MelLossWeight;
        LearningRateDecay = other.LearningRateDecay;
        UpdatesPerEpoch = other.UpdatesPerEpoch;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates HiFi-GAN V1.</summary>
    public HiFiGANOptions()
    {
        UpsampleRates = [8, 8, 2, 2];
        UpsampleKernelSizes = [16, 16, 4, 4];
        UpsampleInitialChannels = 512;
        ResblockKernelSizes = [3, 7, 11];
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        LearningRate = 2e-4;
        WeightDecay = 0.01;
        DropoutRate = 0.0;
    }

    /// <summary>Gets or sets the dilations of each residual block's convolutions ((1, 3, 5) for each kernel).</summary>
    public int[][] ResblockDilationSizes { get; set; } =
    [
        [1, 3, 5],
        [1, 3, 5],
        [1, 3, 5],
    ];

    /// <summary>Gets or sets the residual block type: 1 (two convolutions per dilation, V1/V2) or 2 (one, V3).</summary>
    public int ResblockType { get; set; } = 1;

    /// <summary>Gets or sets the multi-period discriminator's periods (2, 3, 5, 7, 11).</summary>
    public int[] DiscriminatorPeriods { get; set; } = [2, 3, 5, 7, 11];

    /// <summary>Gets or sets a divisor on every discriminator width (1: the paper's 32–1024 channels).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>Gets or sets the training segment in samples (8192).</summary>
    public int SegmentSize { get; set; } = 8192;

    /// <summary>Gets or sets the STFT window length (1024).</summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>Gets or sets the highest frequency of the input mel filterbank (8000 Hz).</summary>
    public double MelMaxFrequency { get; set; } = 8000.0;

    /// <summary>Gets or sets λ_fm, the feature-matching weight (2).</summary>
    public double FeatureMatchingWeight { get; set; } = 2.0;

    /// <summary>Gets or sets λ_mel, the mel-spectrogram loss weight (45).</summary>
    public double MelLossWeight { get; set; } = 45.0;

    /// <summary>Gets or sets the per-epoch learning-rate decay (0.999).</summary>
    public double LearningRateDecay { get; set; } = 0.999;

    /// <summary>Gets or sets the updates in one epoch, over which the decay applies once (809: LJSpeech's 12,950 clips at a
    /// batch of 16).</summary>
    public int UpdatesPerEpoch { get; set; } = 809;

    /// <summary>Gets or sets the seed of the segment crops, for repeatable runs.</summary>
    public int SamplingSeed { get; set; }
}
