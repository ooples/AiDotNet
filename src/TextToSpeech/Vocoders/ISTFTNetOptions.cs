namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for iSTFTNet (Kaneko et al. 2022): a HiFi-GAN whose output-side upsampling is replaced by an inverse
/// STFT of a small predicted magnitude and phase spectrogram.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's V1-C8C8I model (§3.2–4.1): HiFi-GAN V1's generator (512 channels, kernels 3, 7, 11 with
/// dilations 1, 3, 5) with two ×8 upsamplings (kernels 16, 16), then an output convolution to the (16/2 + 1) × 2
/// magnitude and phase channels of iSTFT(16, 4, 16) — the 1024 / 256 / 1024 analysis divided by the 64× upsampling
/// (Eq. 1); HiFi-GAN's period and scale discriminators and its LSGAN + 2·feature-matching + 45·mel loss; Adam
/// (β = 0.5, 0.9) at 2e-4; 22.05 kHz audio, 80 mel bands from a 1024-point FFT with hop 256. HiFi-GAN's configuration
/// supplies what the paper leaves to it (§4.1): the 0.999 per-epoch learning-rate decay, the 8192-sample segments, batch
/// 16, the mel loss up to Nyquist with the input mel up to 8 kHz; the reference iSTFTNet code reflect-pads one frame
/// before the output convolution so the waveform has exactly frames × 256 samples.
/// </para>
/// <para><b>For Beginners:</b> These options configure the iSTFTNet model. Default values follow the original paper settings.</para>
/// </remarks>
public class ISTFTNetOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ISTFTNetOptions(ISTFTNetOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        ResblockDilationSizes = other.ResblockDilationSizes.Select(d => (int[])d.Clone()).ToArray();
        ResblockType = other.ResblockType;
        InverseFftSize = other.InverseFftSize;
        InverseHopSize = other.InverseHopSize;
        InverseWindowSize = other.InverseWindowSize;
        DiscriminatorPeriods = (int[])other.DiscriminatorPeriods.Clone();
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor;
        FeatureMatchingWeight = other.FeatureMatchingWeight;
        MelLossWeight = other.MelLossWeight;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        LearningRateDecay = other.LearningRateDecay;
        UpdatesPerEpoch = other.UpdatesPerEpoch;
        SegmentSize = other.SegmentSize;
        WindowSize = other.WindowSize;
        MelMaxFrequency = other.MelMaxFrequency;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public ISTFTNetOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        UpsampleRates = [8, 8];
        UpsampleKernelSizes = [16, 16];
        UpsampleInitialChannels = 512;
        ResblockKernelSizes = [3, 7, 11];
        LearningRate = 2e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the residual block dilations ((1, 3, 5) for each kernel).</summary>
    public int[][] ResblockDilationSizes { get; set; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    /// <summary>Gets or sets the residual block type (1, HiFi-GAN V1).</summary>
    public int ResblockType { get; set; } = 1;

    /// <summary>Gets or sets the inverse STFT's FFT size (16).</summary>
    public int InverseFftSize { get; set; } = 16;

    /// <summary>Gets or sets the inverse STFT's hop (4).</summary>
    public int InverseHopSize { get; set; } = 4;

    /// <summary>Gets or sets the inverse STFT's window (16).</summary>
    public int InverseWindowSize { get; set; } = 16;

    /// <summary>Gets or sets the period discriminators' periods (2, 3, 5, 7, 11).</summary>
    public int[] DiscriminatorPeriods { get; set; } = [2, 3, 5, 7, 11];

    /// <summary>Gets or sets a divisor on every discriminator width (1: the paper's widths).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>Gets or sets the feature-matching weight (2).</summary>
    public double FeatureMatchingWeight { get; set; } = 2.0;

    /// <summary>Gets or sets the mel loss weight (45).</summary>
    public double MelLossWeight { get; set; } = 45.0;

    /// <summary>Gets or sets Adam's β₁ (0.5).</summary>
    public double Beta1 { get; set; } = 0.5;

    /// <summary>Gets or sets Adam's β₂ (0.9).</summary>
    public double Beta2 { get; set; } = 0.9;

    /// <summary>Gets or sets the per-epoch learning-rate decay (0.999).</summary>
    public double LearningRateDecay { get; set; } = 0.999;

    /// <summary>Gets or sets the updates in one epoch (788: 12,600 training clips at a batch of 16).</summary>
    public int UpdatesPerEpoch { get; set; } = 788;

    /// <summary>Gets or sets the training segment in samples (8192).</summary>
    public int SegmentSize { get; set; } = 8192;

    /// <summary>Gets or sets the analysis window of the input features (1024).</summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>Gets or sets the input mel's highest frequency (8000 Hz).</summary>
    public double MelMaxFrequency { get; set; } = 8000;

    /// <summary>Gets or sets the seed of the training segment draws.</summary>
    public int SamplingSeed { get; set; }
}
