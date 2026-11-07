namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for Vocos (Siuzdak 2024): a ConvNeXt backbone at frame rate whose head predicts STFT magnitude and
/// phase, inverted by the iSTFT, trained against multi-period and multi-resolution discriminators.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (§3.2–3.3, §4.1): 24 kHz LibriTTS audio with 100-band mel input from a 1024-point FFT with
/// hop 256; a 512-wide backbone of 8 ConvNeXt blocks with a 1536-wide bottleneck; the hinge loss, feature matching and
/// the mel L1; AdamW (β = 0.9, 0.999) at 2e-4 with a cosine decay over 1M steps per network; 16,384-sample segments,
/// batch 16; the multi-resolution discriminator of Jang et al. 2021 (UnivNet's resolutions (1024, 120, 600),
/// (2048, 240, 1200), (512, 50, 240)) and HiFi-GAN's multi-period discriminator.
/// </para>
/// <para>What the paper leaves open follows gemelo-ai/vocos (<c>configs/vocos.yaml</c>, <c>experiment.py</c>): λ_mel = 45;
/// the multi-resolution terms weighted 0.1; adversarial and feature-matching terms averaged per discriminator family;
/// mel features from torchaudio's <c>MelSpectrogram</c> (centred, magnitude, HTK scale) with <c>log(max(x, 1e-7))</c>;
/// the magnitude clipped at 100.</para>
/// <para><b>For Beginners:</b> These options configure the Vocos model. Default values follow the original paper settings.</para>
/// </remarks>
public class VocosOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VocosOptions(VocosOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        ConvNeXtDim = other.ConvNeXtDim;
        NumBackboneBlocks = other.NumBackboneBlocks;
        IntermediateDim = other.IntermediateDim;
        DiscriminatorPeriods = (int[])other.DiscriminatorPeriods.Clone();
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor;
        ResolutionDiscriminatorChannels = other.ResolutionDiscriminatorChannels;
        ResolutionFftSizes = (int[])other.ResolutionFftSizes.Clone();
        ResolutionHopSizes = (int[])other.ResolutionHopSizes.Clone();
        ResolutionWindowSizes = (int[])other.ResolutionWindowSizes.Clone();
        MelLossWeight = other.MelLossWeight;
        ResolutionLossWeight = other.ResolutionLossWeight;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        CosineSteps = other.CosineSteps;
        SegmentSize = other.SegmentSize;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public VocosOptions()
    {
        SampleRate = 24000;
        MelChannels = 100;
        HopSize = 256;
        FftSize = 1024;
        LearningRate = 2e-4;
        WeightDecay = 0.01;
    }

    /// <summary>Gets or sets the backbone width d (512).</summary>
    public int ConvNeXtDim { get; set; } = 512;

    /// <summary>Gets or sets the ConvNeXt blocks (8).</summary>
    public int NumBackboneBlocks { get; set; } = 8;

    /// <summary>Gets or sets the bottleneck width (1536).</summary>
    public int IntermediateDim { get; set; } = 1536;

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

    /// <summary>Gets or sets λ_mel (45).</summary>
    public double MelLossWeight { get; set; } = 45.0;

    /// <summary>Gets or sets the weight of the resolution discriminators' terms (0.1).</summary>
    public double ResolutionLossWeight { get; set; } = 0.1;

    /// <summary>Gets or sets AdamW's β₁ (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets AdamW's β₂ (0.999).</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>Gets or sets the steps of each network's cosine decay (1,000,000).</summary>
    public int CosineSteps { get; set; } = 1_000_000;

    /// <summary>Gets or sets the training segment in samples (16,384).</summary>
    public int SegmentSize { get; set; } = 16384;

    /// <summary>Gets or sets the seed of the training draws and the reference initialization.</summary>
    public int SamplingSeed { get; set; }
}
