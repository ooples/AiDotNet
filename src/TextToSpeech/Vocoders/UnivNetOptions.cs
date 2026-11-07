namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for UnivNet (Jang et al. 2021): a GAN vocoder with location-variable convolutions, trained with
/// multi-resolution spectrogram and multi-period discriminators and the multi-resolution STFT loss.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's UnivNet-c32 (§4.1, §4.3): 24 kHz audio with full-band (0–12 kHz) 100-band log-mel input from a
/// 1024-point FFT with hop 256; 64 noise channels; c_G = 32; LVC stacks upsampling 8, 8, 4 with dilations 1, 3, 9, 27;
/// leaky ReLU 0.2; STFT resolutions (1024, 120, 600), (2048, 240, 1200), (512, 50, 240) for the resolution discriminators
/// and the auxiliary loss; λ = 2.5; 200k generator-only steps; Adam (β = 0.5, 0.9) at 1e-4.
/// </para>
/// <para>What the paper leaves open follows maum-ai/univnet (<c>config/default_c32.yaml</c>): the kernel predictor's 64
/// hidden channels and 3-wide convolutions; period discriminators of 64, 128, 256, 512, 1024 channels; 32-channel
/// resolution discriminators; 16,384-sample segments, batch 32; adversarial terms averaged over the discriminators;
/// at synthesis ten frames of log(1e-5) appended and their samples trimmed.</para>
/// <para><b>For Beginners:</b> These options configure the UnivNet model. Default values follow the original paper settings.</para>
/// </remarks>
public class UnivNetOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UnivNetOptions(UnivNetOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        NoiseDim = other.NoiseDim;
        ChannelSize = other.ChannelSize;
        Dilations = (int[])other.Dilations.Clone();
        LeakyReluSlope = other.LeakyReluSlope;
        KernelPredictorHidden = other.KernelPredictorHidden;
        KernelPredictorConvSize = other.KernelPredictorConvSize;
        DiscriminatorPeriods = (int[])other.DiscriminatorPeriods.Clone();
        PeriodDiscriminatorChannels = (int[])other.PeriodDiscriminatorChannels.Clone();
        ResolutionDiscriminatorChannels = other.ResolutionDiscriminatorChannels;
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor;
        StftFftSizes = (int[])other.StftFftSizes.Clone();
        StftHopSizes = (int[])other.StftHopSizes.Clone();
        StftWindowSizes = (int[])other.StftWindowSizes.Clone();
        StftLossWeight = other.StftLossWeight;
        PretrainingSteps = other.PretrainingSteps;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        SegmentSize = other.SegmentSize;
        WindowSize = other.WindowSize;
        MelMaxFrequency = other.MelMaxFrequency;
        InferencePaddingFrames = other.InferencePaddingFrames;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's UnivNet-c32 configuration.</summary>
    public UnivNetOptions()
    {
        SampleRate = 24000;
        MelChannels = 100;
        HopSize = 256;
        FftSize = 1024;
        UpsampleRates = [8, 8, 4];
        LearningRate = 1e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the noise channels (64).</summary>
    public int NoiseDim { get; set; } = 64;

    /// <summary>Gets or sets c_G, the generator's channels (32; UnivNet-c16 uses 16).</summary>
    public int ChannelSize { get; set; } = 32;

    /// <summary>Gets or sets the dilations of each LVC stack (1, 3, 9, 27).</summary>
    public int[] Dilations { get; set; } = [1, 3, 9, 27];

    /// <summary>Gets or sets the leaky ReLU slope (0.2).</summary>
    public double LeakyReluSlope { get; set; } = 0.2;

    /// <summary>Gets or sets the kernel predictor's hidden channels (64).</summary>
    public int KernelPredictorHidden { get; set; } = 64;

    /// <summary>Gets or sets the kernel predictor's convolution size (3).</summary>
    public int KernelPredictorConvSize { get; set; } = 3;

    /// <summary>Gets or sets the period discriminators' periods (2, 3, 5, 7, 11).</summary>
    public int[] DiscriminatorPeriods { get; set; } = [2, 3, 5, 7, 11];

    /// <summary>Gets or sets the period discriminators' widths (64, 128, 256, 512, 1024).</summary>
    public int[] PeriodDiscriminatorChannels { get; set; } = [64, 128, 256, 512, 1024];

    /// <summary>Gets or sets the resolution discriminators' channels (32).</summary>
    public int ResolutionDiscriminatorChannels { get; set; } = 32;

    /// <summary>Gets or sets a divisor on the period discriminators' widths (1: the paper's).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>Gets or sets the STFT FFT sizes of the resolution discriminators and the auxiliary loss (1024, 2048, 512).</summary>
    public int[] StftFftSizes { get; set; } = [1024, 2048, 512];

    /// <summary>Gets or sets the STFT hops (120, 240, 50).</summary>
    public int[] StftHopSizes { get; set; } = [120, 240, 50];

    /// <summary>Gets or sets the STFT windows (600, 1200, 240).</summary>
    public int[] StftWindowSizes { get; set; } = [600, 1200, 240];

    /// <summary>Gets or sets λ, the auxiliary STFT loss's weight (2.5).</summary>
    public double StftLossWeight { get; set; } = 2.5;

    /// <summary>Gets or sets the generator-only steps (200,000).</summary>
    public long PretrainingSteps { get; set; } = 200_000;

    /// <summary>Gets or sets Adam's β₁ (0.5).</summary>
    public double Beta1 { get; set; } = 0.5;

    /// <summary>Gets or sets Adam's β₂ (0.9).</summary>
    public double Beta2 { get; set; } = 0.9;

    /// <summary>Gets or sets the training segment in samples (16,384).</summary>
    public int SegmentSize { get; set; } = 16384;

    /// <summary>Gets or sets the analysis window of the input features (1024).</summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>Gets or sets the input mel's highest frequency (12,000 Hz: full band at 24 kHz).</summary>
    public double MelMaxFrequency { get; set; } = 12000;

    /// <summary>Gets or sets the log(1e-5) frames appended at synthesis and trimmed from the waveform (10).</summary>
    public int InferencePaddingFrames { get; set; } = 10;

    /// <summary>Gets or sets the seed of the training draws and of the synthesis noise.</summary>
    public int SamplingSeed { get; set; }
}
