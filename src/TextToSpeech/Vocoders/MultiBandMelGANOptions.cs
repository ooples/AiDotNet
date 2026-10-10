namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for Multi-band MelGAN (Yang et al. 2021): a MelGAN generator predicting four PQMF sub-bands, trained
/// with the full- and sub-band multi-resolution STFT loss after a generator-only pre-training phase.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (§3.1–3.2, Tables 1–2): 16 kHz audio, 50 ms frames (800 samples) with a 12.5 ms shift (200)
/// and a 1024-point FFT, 80 mel bands standardized per band; a generator of 384 first channels upsampling ×2, ×5, ×5 to
/// 192, 96, 48 channels, residual stacks of four blocks (dilations 1, 3, 9, 27) and four output bands merged by a PQMF of
/// 63 coefficients; three discriminators with strided convolutions of 64, 256, 512 channels; LSGAN; the full-band STFT
/// resolutions (1024, 600, 120), (2048, 1200, 240), (512, 240, 50) and sub-band ones (384, 150, 30), (683, 300, 60),
/// (171, 60, 10); 200k generator-only steps; Adam at 1e-4 for both networks, halved every 100k steps down to 1e-6; one
/// second of audio per example, batch 128.
/// </para>
/// <para>What the paper leaves open follows kan-bayashi/ParallelWaveGAN (<c>multi_band_melgan.v2.yaml</c>,
/// <c>logmelfilterbank</c>): λ_adv = 2.5 on the adversarial term; mel bands from 80 to 7600 Hz with
/// <c>log10(max(x, 1e-10))</c>; PQMF cutoff 0.142 and Kaiser β 9.</para>
/// <para><b>For Beginners:</b> These options configure the Multi-band MelGAN model. Default values follow the original paper settings.</para>
/// </remarks>
public class MultiBandMelGANOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MultiBandMelGANOptions(MultiBandMelGANOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        NumBands = other.NumBands;
        PqmfTaps = other.PqmfTaps;
        PqmfCutoffRatio = other.PqmfCutoffRatio;
        PqmfBeta = other.PqmfBeta;
        ResidualLayers = other.ResidualLayers;
        NumDiscriminators = other.NumDiscriminators;
        DiscriminatorChannels = (int[])other.DiscriminatorChannels.Clone();
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor;
        AdversarialWeight = other.AdversarialWeight;
        FullBandFftSizes = (int[])other.FullBandFftSizes.Clone();
        FullBandWindowSizes = (int[])other.FullBandWindowSizes.Clone();
        FullBandHopSizes = (int[])other.FullBandHopSizes.Clone();
        SubBandFftSizes = (int[])other.SubBandFftSizes.Clone();
        SubBandWindowSizes = (int[])other.SubBandWindowSizes.Clone();
        SubBandHopSizes = (int[])other.SubBandHopSizes.Clone();
        PretrainingSteps = other.PretrainingSteps;
        LearningRateHalvingSteps = other.LearningRateHalvingSteps;
        MinimumLearningRate = other.MinimumLearningRate;
        SegmentSize = other.SegmentSize;
        WindowSize = other.WindowSize;
        MelMinFrequency = other.MelMinFrequency;
        MelMaxFrequency = other.MelMaxFrequency;
        MelMean = (double[]?)other.MelMean?.Clone();
        MelScale = (double[]?)other.MelScale?.Clone();
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public MultiBandMelGANOptions()
    {
        SampleRate = 16000;
        MelChannels = 80;
        HopSize = 200;
        FftSize = 1024;
        UpsampleRates = [2, 5, 5];
        UpsampleInitialChannels = 384;
        LearningRate = 1e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the number of sub-bands (4).</summary>
    public int NumBands { get; set; } = 4;

    /// <summary>Gets or sets the PQMF taps (62: filters of 63 coefficients).</summary>
    public int PqmfTaps { get; set; } = 62;

    /// <summary>Gets or sets the PQMF prototype's cutoff ratio (0.142).</summary>
    public double PqmfCutoffRatio { get; set; } = 0.142;

    /// <summary>Gets or sets the PQMF prototype's Kaiser β (9).</summary>
    public double PqmfBeta { get; set; } = 9.0;

    /// <summary>Gets or sets the residual blocks per stack (4: dilations 1, 3, 9, 27).</summary>
    public int ResidualLayers { get; set; } = 4;

    /// <summary>Gets or sets the number of discriminators (3).</summary>
    public int NumDiscriminators { get; set; } = 3;

    /// <summary>Gets or sets the discriminators' strided convolution widths (64, 256, 512).</summary>
    public int[] DiscriminatorChannels { get; set; } = [64, 256, 512];

    /// <summary>Gets or sets a divisor on every discriminator width (1: the paper's widths).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>Gets or sets λ_adv, the adversarial term's weight (2.5).</summary>
    public double AdversarialWeight { get; set; } = 2.5;

    /// <summary>Gets or sets the full-band STFT loss FFT sizes (1024, 2048, 512).</summary>
    public int[] FullBandFftSizes { get; set; } = [1024, 2048, 512];

    /// <summary>Gets or sets the full-band STFT loss windows (600, 1200, 240).</summary>
    public int[] FullBandWindowSizes { get; set; } = [600, 1200, 240];

    /// <summary>Gets or sets the full-band STFT loss hops (120, 240, 50).</summary>
    public int[] FullBandHopSizes { get; set; } = [120, 240, 50];

    /// <summary>Gets or sets the sub-band STFT loss FFT sizes (384, 683, 171).</summary>
    public int[] SubBandFftSizes { get; set; } = [384, 683, 171];

    /// <summary>Gets or sets the sub-band STFT loss windows (150, 300, 60).</summary>
    public int[] SubBandWindowSizes { get; set; } = [150, 300, 60];

    /// <summary>Gets or sets the sub-band STFT loss hops (30, 60, 10).</summary>
    public int[] SubBandHopSizes { get; set; } = [30, 60, 10];

    /// <summary>Gets or sets the generator-only pre-training steps (200,000).</summary>
    public long PretrainingSteps { get; set; } = 200_000;

    /// <summary>Gets or sets the steps between learning-rate halvings (100,000).</summary>
    public int LearningRateHalvingSteps { get; set; } = 100_000;

    /// <summary>Gets or sets the learning rate's floor (1e-6).</summary>
    public double MinimumLearningRate { get; set; } = 1e-6;

    /// <summary>Gets or sets the training segment in samples (16,000: one second).</summary>
    public int SegmentSize { get; set; } = 16000;

    /// <summary>Gets or sets the analysis window of the input features (800: 50 ms).</summary>
    public int WindowSize { get; set; } = 800;

    /// <summary>Gets or sets the lowest mel frequency (80 Hz).</summary>
    public double MelMinFrequency { get; set; } = 80;

    /// <summary>Gets or sets the highest mel frequency (7600 Hz).</summary>
    public double MelMaxFrequency { get; set; } = 7600;

    /// <summary>Gets or sets the per-band mean of the log-mel features over the training set, or null for none.</summary>
    public double[]? MelMean { get; set; }

    /// <summary>Gets or sets the per-band standard deviation of the log-mel features over the training set, or null for
    /// none.</summary>
    public double[]? MelScale { get; set; }

    /// <summary>Gets or sets the seed of the training segment draws.</summary>
    public int SamplingSeed { get; set; }
}
