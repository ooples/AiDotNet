namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options APNet and APNet2 share (Ai and Ling 2023; Du et al. 2023): the analysis, the multilevel loss weights,
/// the discriminators' shared settings, the training segment and the optimizer schedule.</summary>
/// <remarks>
/// <para>The loss weights are the papers' (APNet §IV; APNet2 keeps them): λ_A = λ_Mel = 45, λ_P = 100, λ_S = 20,
/// λ_RI = 2.25. Both train with AdamW (β = 0.8, 0.99, weight decay 0.01) at 2e-4, decayed by 0.999 per epoch, batch 16.</para>
/// <para><b>For Beginners:</b> These options configure the amplitude-and-phase vocoders.</para>
/// </remarks>
public abstract class AmplitudePhaseVocoderOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    protected AmplitudePhaseVocoderOptions(AmplitudePhaseVocoderOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        InputKernelSize = other.InputKernelSize;
        OutputKernelSize = other.OutputKernelSize;
        WindowSize = other.WindowSize;
        MelMaxFrequency = other.MelMaxFrequency;
        AmplitudeLossWeight = other.AmplitudeLossWeight;
        PhaseLossWeight = other.PhaseLossWeight;
        StftLossWeight = other.StftLossWeight;
        RealImaginaryLossWeight = other.RealImaginaryLossWeight;
        MelLossWeight = other.MelLossWeight;
        DiscriminatorPeriods = (int[])other.DiscriminatorPeriods.Clone();
        DiscriminatorWidthDivisor = other.DiscriminatorWidthDivisor;
        SegmentSize = other.SegmentSize;
        UpdatesPerEpoch = other.UpdatesPerEpoch;
        LearningRateDecay = other.LearningRateDecay;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Initializes the shared defaults.</summary>
    protected AmplitudePhaseVocoderOptions()
    {
        MelChannels = 80;
        FftSize = 1024;
        LearningRate = 2e-4;
        WeightDecay = 0.01;
    }

    /// <summary>Gets or sets the input convolutions' kernel (7).</summary>
    public int InputKernelSize { get; set; } = 7;

    /// <summary>Gets or sets the output convolutions' kernel (7).</summary>
    public int OutputKernelSize { get; set; } = 7;

    /// <summary>Gets or sets the analysis window in samples.</summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>Gets or sets the highest mel frequency (8 kHz).</summary>
    public double MelMaxFrequency { get; set; } = 8000;

    /// <summary>Gets or sets λ_A of the amplitude loss (45).</summary>
    public double AmplitudeLossWeight { get; set; } = 45;

    /// <summary>Gets or sets λ_P of the phase losses (100).</summary>
    public double PhaseLossWeight { get; set; } = 100;

    /// <summary>Gets or sets λ_S of the reconstructed STFT spectrum losses (20).</summary>
    public double StftLossWeight { get; set; } = 20;

    /// <summary>Gets or sets λ_RI of the real and imaginary part losses within the STFT losses (2.25).</summary>
    public double RealImaginaryLossWeight { get; set; } = 2.25;

    /// <summary>Gets or sets λ_Mel of the mel spectrogram loss (45).</summary>
    public double MelLossWeight { get; set; } = 45;

    /// <summary>Gets or sets the multi-period discriminator's periods (2, 3, 5, 7, 11).</summary>
    public int[] DiscriminatorPeriods { get; set; } = [2, 3, 5, 7, 11];

    /// <summary>Gets or sets a divisor of the discriminators' widths (1: the papers').</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    /// <summary>Gets or sets the training segment in samples.</summary>
    public int SegmentSize { get; set; } = 8192;

    /// <summary>Gets or sets the optimizer steps per epoch over which the learning rate decays once.</summary>
    public int UpdatesPerEpoch { get; set; } = 750;

    /// <summary>Gets or sets the per-epoch learning-rate decay (0.999).</summary>
    public double LearningRateDecay { get; set; } = 0.999;

    /// <summary>Gets or sets the seed of the training draws.</summary>
    public int SamplingSeed { get; set; }
}
