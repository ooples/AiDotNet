namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for Parallel WaveGAN (Yamamoto et al. 2020): a non-autoregressive WaveNet generator over Gaussian noise,
/// trained with the multi-resolution STFT loss and an adversarial loss.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (§4.1): 24 kHz audio; 80 log-mel bands from 70 to 8000 Hz with 50 ms frames and a 12.5 ms
/// shift (1200 / 300 samples), standardized; a generator of 30 dilated residual blocks in three cycles with 64 residual
/// and skip channels and kernel 3; a discriminator of ten 64-channel convolutions; λ_adv = 4; STFT resolutions (1024,
/// 600, 120), (2048, 1200, 240), (512, 240, 50); 400k steps with the discriminator fixed for the first 100k; RAdam
/// (ε = 1e-6) at 1e-4 (generator) and 5e-5 (discriminator), halved every 200k steps; batch 8 of 1-second clips.
/// </para>
/// <para>What the paper leaves open follows kan-bayashi/ParallelWaveGAN (<c>parallel_wavegan.v1.yaml</c>): a 2048-point
/// FFT for the features; a 128-channel gate; upsampling scales 4, 5, 3, 5 with an auxiliary context window of 2;
/// gradient-norm clipping at 10 (generator) and 1 (discriminator).</para>
/// <para><b>For Beginners:</b> These options configure the Parallel WaveGAN model. Default values follow the original paper settings.</para>
/// </remarks>
public class ParallelWaveGANOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ParallelWaveGANOptions(ParallelWaveGANOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        AuxContextWindow = other.AuxContextWindow;
        NumLayers = other.NumLayers;
        NumStacks = other.NumStacks;
        ResidualChannels = other.ResidualChannels;
        GateChannels = other.GateChannels;
        SkipChannels = other.SkipChannels;
        KernelSize = other.KernelSize;
        DiscriminatorLayers = other.DiscriminatorLayers;
        DiscriminatorChannels = other.DiscriminatorChannels;
        AdversarialWeight = other.AdversarialWeight;
        StftFftSizes = (int[])other.StftFftSizes.Clone();
        StftWindowSizes = (int[])other.StftWindowSizes.Clone();
        StftHopSizes = (int[])other.StftHopSizes.Clone();
        DiscriminatorStartStep = other.DiscriminatorStartStep;
        DiscriminatorLearningRate = other.DiscriminatorLearningRate;
        Epsilon = other.Epsilon;
        LearningRateHalvingSteps = other.LearningRateHalvingSteps;
        GeneratorGradientClipNorm = other.GeneratorGradientClipNorm;
        DiscriminatorGradientClipNorm = other.DiscriminatorGradientClipNorm;
        SegmentSize = other.SegmentSize;
        WindowSize = other.WindowSize;
        MelMinFrequency = other.MelMinFrequency;
        MelMaxFrequency = other.MelMaxFrequency;
        MelMean = (double[]?)other.MelMean?.Clone();
        MelScale = (double[]?)other.MelScale?.Clone();
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public ParallelWaveGANOptions()
    {
        SampleRate = 24000;
        MelChannels = 80;
        HopSize = 300;
        FftSize = 2048;
        UpsampleRates = [4, 5, 3, 5];
        LearningRate = 1e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the mel frames of context on each side of the input convolution (2).</summary>
    public int AuxContextWindow { get; set; } = 2;

    /// <summary>Gets or sets the generator's residual blocks (30).</summary>
    public int NumLayers { get; set; } = 30;

    /// <summary>Gets or sets the dilation cycles (3).</summary>
    public int NumStacks { get; set; } = 3;

    /// <summary>Gets or sets the residual channels (64).</summary>
    public int ResidualChannels { get; set; } = 64;

    /// <summary>Gets or sets the gate channels (128).</summary>
    public int GateChannels { get; set; } = 128;

    /// <summary>Gets or sets the skip channels (64).</summary>
    public int SkipChannels { get; set; } = 64;

    /// <summary>Gets or sets the convolution kernel (3).</summary>
    public int KernelSize { get; set; } = 3;

    /// <summary>Gets or sets the discriminator's convolutions (10).</summary>
    public int DiscriminatorLayers { get; set; } = 10;

    /// <summary>Gets or sets the discriminator's channels (64).</summary>
    public int DiscriminatorChannels { get; set; } = 64;

    /// <summary>Gets or sets λ_adv (4).</summary>
    public double AdversarialWeight { get; set; } = 4.0;

    /// <summary>Gets or sets the STFT loss FFT sizes (1024, 2048, 512).</summary>
    public int[] StftFftSizes { get; set; } = [1024, 2048, 512];

    /// <summary>Gets or sets the STFT loss windows (600, 1200, 240).</summary>
    public int[] StftWindowSizes { get; set; } = [600, 1200, 240];

    /// <summary>Gets or sets the STFT loss hops (120, 240, 50).</summary>
    public int[] StftHopSizes { get; set; } = [120, 240, 50];

    /// <summary>Gets or sets the steps the discriminator stays fixed (100,000).</summary>
    public long DiscriminatorStartStep { get; set; } = 100_000;

    /// <summary>Gets or sets the discriminator's learning rate (5e-5).</summary>
    public double DiscriminatorLearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets RAdam's ε (1e-6).</summary>
    public double Epsilon { get; set; } = 1e-6;

    /// <summary>Gets or sets the steps between learning-rate halvings (200,000).</summary>
    public int LearningRateHalvingSteps { get; set; } = 200_000;

    /// <summary>Gets or sets the generator's gradient-norm clip (10).</summary>
    public double GeneratorGradientClipNorm { get; set; } = 10.0;

    /// <summary>Gets or sets the discriminator's gradient-norm clip (1).</summary>
    public double DiscriminatorGradientClipNorm { get; set; } = 1.0;

    /// <summary>Gets or sets the training segment in samples (24,000: one second).</summary>
    public int SegmentSize { get; set; } = 24000;

    /// <summary>Gets or sets the analysis window of the input features (1200: 50 ms).</summary>
    public int WindowSize { get; set; } = 1200;

    /// <summary>Gets or sets the lowest mel frequency (70 Hz).</summary>
    public double MelMinFrequency { get; set; } = 70;

    /// <summary>Gets or sets the highest mel frequency (8000 Hz).</summary>
    public double MelMaxFrequency { get; set; } = 8000;

    /// <summary>Gets or sets the per-band mean of the log-mel features over the training set, or null for none.</summary>
    public double[]? MelMean { get; set; }

    /// <summary>Gets or sets the per-band standard deviation of the log-mel features over the training set, or null for
    /// none.</summary>
    public double[]? MelScale { get; set; }

    /// <summary>Gets or sets the seed of the training draws and of the synthesis noise.</summary>
    public int SamplingSeed { get; set; }
}
