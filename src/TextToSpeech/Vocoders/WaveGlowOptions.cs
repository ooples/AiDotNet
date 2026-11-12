namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for WaveGlow (Prenger et al. 2019): a flow-based vocoder trained only by maximum likelihood.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (§2–3): groups of 8 samples; 12 steps of an invertible 1×1 convolution and an affine coupling;
/// 2 channels output early after every 4 couplings; WN networks of 8 weight-normalized dilated convolutions with 3 taps,
/// 512 residual and 256 skip channels; σ = √0.5 in training and 0.6 at inference; 80-band mel spectrograms (librosa's
/// Slaney filters) from a 1024-point FFT with a 1024-sample window and hop 256 at 22.05 kHz; Adam at 1e-4, batch 24,
/// 16,000-sample clips (the paper lowers the rate to 5e-5 by hand when training plateaus).
/// </para>
/// <para>What the paper leaves open follows NVIDIA/waveglow: the mel spectrogram upsampled by a transposed convolution
/// with a 1024-tap kernel and stride 256; Tacotron 2's features (centred STFT, <c>ln(max(x, 1e-5))</c>, 0–8 kHz); the
/// gated units as wide as the residual channels (the paper names only residual and skip widths); no gradient clipping;
/// the loss <c>(Σz² / 2σ² − Σ log s − Σ log|det W|) / samples</c>.</para>
/// <para><b>For Beginners:</b> These options configure the WaveGlow model. Default values follow the original paper settings.</para>
/// </remarks>
public class WaveGlowOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public WaveGlowOptions(WaveGlowOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        NumFlows = other.NumFlows;
        NumWaveNetLayers = other.NumWaveNetLayers;
        EarlyOutputChannels = other.EarlyOutputChannels;
        EarlyOutputEvery = other.EarlyOutputEvery;
        GroupSize = other.GroupSize;
        ResidualChannels = other.ResidualChannels;
        GateChannels = other.GateChannels;
        SkipChannels = other.SkipChannels;
        KernelSize = other.KernelSize;
        UpsampleKernel = other.UpsampleKernel;
        TrainingSigma = other.TrainingSigma;
        InferenceSigma = other.InferenceSigma;
        SegmentSamples = other.SegmentSamples;
        WindowSize = other.WindowSize;
        MelMinFrequency = other.MelMinFrequency;
        MelMaxFrequency = other.MelMaxFrequency;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's WaveGlow configuration.</summary>
    public WaveGlowOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        LearningRate = 1e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the steps of flow (12).</summary>
    public int NumFlows { get; set; } = 12;

    /// <summary>Gets or sets the WN layers per coupling (8).</summary>
    public int NumWaveNetLayers { get; set; } = 8;

    /// <summary>Gets or sets the channels output early each time (2).</summary>
    public int EarlyOutputChannels { get; set; } = 2;

    /// <summary>Gets or sets the couplings between early outputs (4).</summary>
    public int EarlyOutputEvery { get; set; } = 4;

    /// <summary>Gets or sets the samples per squeezed vector (8).</summary>
    public int GroupSize { get; set; } = 8;

    /// <summary>Gets or sets the WN residual channels (512).</summary>
    public int ResidualChannels { get; set; } = 512;

    /// <summary>Gets or sets the WN gated-unit channels (512).</summary>
    public int GateChannels { get; set; } = 512;

    /// <summary>Gets or sets the WN skip channels (256).</summary>
    public int SkipChannels { get; set; } = 256;

    /// <summary>Gets or sets the WN convolution taps (3).</summary>
    public int KernelSize { get; set; } = 3;

    /// <summary>Gets or sets the mel upsampler's kernel (1024).</summary>
    public int UpsampleKernel { get; set; } = 1024;

    /// <summary>Gets or sets σ of the latent Gaussian in training (√0.5).</summary>
    public double TrainingSigma { get; set; } = Math.Sqrt(0.5);

    /// <summary>Gets or sets σ of the latent Gaussian at inference (0.6).</summary>
    public double InferenceSigma { get; set; } = 0.6;

    /// <summary>Gets or sets the samples of a training clip (16,000; rounded down to whole frames).</summary>
    public int SegmentSamples { get; set; } = 16000;

    /// <summary>Gets or sets the analysis window of the input features (1024).</summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>Gets or sets the lowest mel frequency (0 Hz).</summary>
    public double MelMinFrequency { get; set; }

    /// <summary>Gets or sets the highest mel frequency (8 kHz).</summary>
    public double MelMaxFrequency { get; set; } = 8000;

    /// <summary>Gets or sets the seed of the training draws and of the synthesis latents.</summary>
    public int SamplingSeed { get; set; }
}
