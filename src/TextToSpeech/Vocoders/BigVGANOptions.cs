namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for BigVGAN (Lee et al. 2023): a HiFi-GAN generator with anti-aliased Snake activations, trained with
/// multi-period and multi-resolution discriminators.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's 112M-parameter BigVGAN (§3.4, §4.2, Table 6): 24 kHz LibriTTS audio with 100-band log-mel
/// input up to 12 kHz from a 1024-point FFT with hop 256; h = 1536; upsampling 4, 4, 2, 2, 2, 2 (kernels 8, 8, 4, 4, 4,
/// 4); AMP kernels 3, 7, 11 with dilations 1, 3, 5; period discriminators 2, 3, 5, 7, 11 and resolution discriminators
/// (1024, 120, 600), (2048, 240, 1200), (512, 50, 240); batch 32, 8192-sample segments, learning rate 1e-4; gradient
/// norms clipped at 1000. HiFi-GAN's official configuration supplies the rest, as the paper states (§4.2): AdamW
/// (β = 0.8, 0.99, PyTorch's weight decay 0.01) decayed 0.999 per epoch, λ_FM = 2, λ_mel = 45 with the loss mel to
/// Nyquist. <see cref="Base"/> gives BigVGAN-base (h = 512, upsampling 8, 8, 2, 2 with kernels 16, 16, 4, 4; 14M
/// parameters).
/// </para>
/// <para><b>For Beginners:</b> These options configure the BigVGAN model. Default values follow the original paper settings.</para>
/// </remarks>
public class BigVGANOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BigVGANOptions(BigVGANOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        ResblockDilationSizes = other.ResblockDilationSizes.Select(d => (int[])d.Clone()).ToArray();
        DiscriminatorPeriods = (int[])other.DiscriminatorPeriods.Clone();
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor;
        ResolutionDiscriminatorChannels = other.ResolutionDiscriminatorChannels;
        ResolutionFftSizes = (int[])other.ResolutionFftSizes.Clone();
        ResolutionHopSizes = (int[])other.ResolutionHopSizes.Clone();
        ResolutionWindowSizes = (int[])other.ResolutionWindowSizes.Clone();
        FeatureMatchingWeight = other.FeatureMatchingWeight;
        MelLossWeight = other.MelLossWeight;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        LearningRateDecay = other.LearningRateDecay;
        UpdatesPerEpoch = other.UpdatesPerEpoch;
        GradientClipNorm = other.GradientClipNorm;
        SegmentSize = other.SegmentSize;
        WindowSize = other.WindowSize;
        MelMaxFrequency = other.MelMaxFrequency;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's 112M BigVGAN configuration.</summary>
    public BigVGANOptions()
    {
        SampleRate = 24000;
        MelChannels = 100;
        HopSize = 256;
        FftSize = 1024;
        UpsampleRates = [4, 4, 2, 2, 2, 2];
        UpsampleKernelSizes = [8, 8, 4, 4, 4, 4];
        UpsampleInitialChannels = 1536;
        ResblockKernelSizes = [3, 7, 11];
        LearningRate = 1e-4;
        WeightDecay = 0.01;
    }

    /// <summary>BigVGAN-base: h = 512, upsampling 8, 8, 2, 2 with kernels 16, 16, 4, 4 (14M parameters).</summary>
    public static BigVGANOptions Base() => new()
    {
        UpsampleRates = [8, 8, 2, 2],
        UpsampleKernelSizes = [16, 16, 4, 4],
        UpsampleInitialChannels = 512,
    };

    /// <summary>Gets or sets the AMP blocks' dilations ((1, 3, 5) for each kernel).</summary>
    public int[][] ResblockDilationSizes { get; set; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    /// <summary>Gets or sets the period discriminators' periods (2, 3, 5, 7, 11).</summary>
    public int[] DiscriminatorPeriods { get; set; } = [2, 3, 5, 7, 11];

    /// <summary>Gets or sets a divisor on the period discriminators' widths (1: the paper's).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>Gets or sets the resolution discriminators' channels (32).</summary>
    public int ResolutionDiscriminatorChannels { get; set; } = 32;

    /// <summary>Gets or sets the resolution discriminators' FFT sizes (1024, 2048, 512).</summary>
    public int[] ResolutionFftSizes { get; set; } = [1024, 2048, 512];

    /// <summary>Gets or sets the resolution discriminators' hops (120, 240, 50).</summary>
    public int[] ResolutionHopSizes { get; set; } = [120, 240, 50];

    /// <summary>Gets or sets the resolution discriminators' windows (600, 1200, 240).</summary>
    public int[] ResolutionWindowSizes { get; set; } = [600, 1200, 240];

    /// <summary>Gets or sets λ_FM (2).</summary>
    public double FeatureMatchingWeight { get; set; } = 2.0;

    /// <summary>Gets or sets λ_mel (45).</summary>
    public double MelLossWeight { get; set; } = 45.0;

    /// <summary>Gets or sets AdamW's β₁ (0.8).</summary>
    public double Beta1 { get; set; } = 0.8;

    /// <summary>Gets or sets AdamW's β₂ (0.99).</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>Gets or sets the per-epoch learning-rate decay (0.999).</summary>
    public double LearningRateDecay { get; set; } = 0.999;

    /// <summary>Gets or sets the updates in one epoch (11,087: LibriTTS train-full's 354,780 utterances at a batch of 32).</summary>
    public int UpdatesPerEpoch { get; set; } = 11087;

    /// <summary>Gets or sets the gradient-norm clip of the generator and of each discriminator family (1000).</summary>
    public double GradientClipNorm { get; set; } = 1000.0;

    /// <summary>Gets or sets the training segment in samples (8192).</summary>
    public int SegmentSize { get; set; } = 8192;

    /// <summary>Gets or sets the analysis window of the input features (1024).</summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>Gets or sets the input mel's highest frequency (12,000 Hz).</summary>
    public double MelMaxFrequency { get; set; } = 12000;

    /// <summary>Gets or sets the seed of the training draws and of the reference initialization.</summary>
    public int SamplingSeed { get; set; }
}
