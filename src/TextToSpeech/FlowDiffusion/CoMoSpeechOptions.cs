using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for CoMoSpeech (Ye et al. 2023): Grad-TTS's encoder with an EDM-preconditioned U-Net teacher,
/// consistency-distilled into one-step synthesis.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's TTS configuration (§3, §4.1.2) with the values it leaves to its reference implementation
/// (zhenye234/CoMoSpeech, <c>params.py</c>, <c>model/como.py</c>): Grad-TTS's encoder (width 192, 6 relative-position
/// layers, 2 heads, window 4, filter 768, kernel 3, dropout 0.1, pre-net of 3 kernel-5 convolutions) and duration
/// predictor (filter 256); Grad-TTS's U-Net (base width 64, multipliers 1, 2, 4, time embedding scale 1000) as F_θ;
/// EDM with σ_data = 0.5, ε = σ_min = 0.002, σ_max = 80, ρ = 7, training σ from ln σ ~ N(−1.2, 1.2²); 50 teacher Euler
/// steps; consistency distillation over 50 discretization points with an EMA target at μ = 0.95; Adam at 1e-4 with
/// encoder and decoder gradients each clipped to norm 1; 2-second training segments; 22.05 kHz, 1024-point STFT,
/// hop 256, 80 mel bins.
/// </para>
/// <para><b>For Beginners:</b> These options configure the CoMoSpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class CoMoSpeechOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CoMoSpeechOptions(CoMoSpeechOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        FlowDim = other.FlowDim;
        DecoderDimMultipliers = (int[])other.DecoderDimMultipliers.Clone();
        TimePositionScale = other.TimePositionScale;
        EncoderKernelSize = other.EncoderKernelSize;
        RelativeWindow = other.RelativeWindow;
        PrenetLayers = other.PrenetLayers;
        PrenetKernelSize = other.PrenetKernelSize;
        PrenetDropout = other.PrenetDropout;
        DurationPredictorFilterChannels = other.DurationPredictorFilterChannels;
        SigmaData = other.SigmaData;
        SigmaMin = other.SigmaMin;
        SigmaMax = other.SigmaMax;
        Rho = other.Rho;
        LogSigmaMean = other.LogSigmaMean;
        LogSigmaStd = other.LogSigmaStd;
        TeacherSamplingSteps = other.TeacherSamplingSteps;
        DistillationSteps = other.DistillationSteps;
        EmaDecay = other.EmaDecay;
        SegmentFrames = other.SegmentFrames;
        LengthScale = other.LengthScale;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public CoMoSpeechOptions()
    {
        HiddenDim = 192;
        EncoderDim = 192;
        FilterChannels = 768;
        NumEncoderLayers = 6;
        NumHeads = 2;
        DropoutRate = 0.1;
        NumFlowSteps = 1;
        SampleRate = 22050;
        HopSize = 256;
        FftSize = 1024;
        MelChannels = 80;
        LearningRate = 1e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the U-Net's base width (64).</summary>
    public int FlowDim { get; set; } = 64;

    /// <summary>Gets or sets the U-Net's width multipliers per resolution (1, 2, 4).</summary>
    public int[] DecoderDimMultipliers { get; set; } = { 1, 2, 4 };

    /// <summary>Gets or sets the scale of the noise level c_noise = ln σ / 4 in the sinusoidal embedding (1000).</summary>
    public double TimePositionScale { get; set; } = 1000.0;

    /// <summary>Gets or sets the encoder feed-forward and duration predictor kernel (3).</summary>
    public int EncoderKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the encoder attention's maximum relative position (4).</summary>
    public int RelativeWindow { get; set; } = 4;

    /// <summary>Gets or sets the pre-net's convolutions (3).</summary>
    public int PrenetLayers { get; set; } = 3;

    /// <summary>Gets or sets the pre-net's kernel (5).</summary>
    public int PrenetKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the pre-net's dropout (0.5).</summary>
    public double PrenetDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the duration predictor's filter (256).</summary>
    public int DurationPredictorFilterChannels { get; set; } = 256;

    /// <summary>Gets or sets σ_data (0.5, Eq. 9).</summary>
    public double SigmaData { get; set; } = 0.5;

    /// <summary>Gets or sets ε = σ_min, the smallest noise level (0.002, Eq. 9).</summary>
    public double SigmaMin { get; set; } = 0.002;

    /// <summary>Gets or sets σ_max, the largest noise level (80).</summary>
    public double SigmaMax { get; set; } = 80.0;

    /// <summary>Gets or sets ρ of the noise-level discretization (7).</summary>
    public double Rho { get; set; } = 7.0;

    /// <summary>Gets or sets the mean of ln σ during teacher training (−1.2).</summary>
    public double LogSigmaMean { get; set; } = -1.2;

    /// <summary>Gets or sets the standard deviation of ln σ during teacher training (1.2).</summary>
    public double LogSigmaStd { get; set; } = 1.2;

    /// <summary>Gets or sets the teacher's Euler steps N at synthesis (50, Table 1).</summary>
    public int TeacherSamplingSteps { get; set; } = 50;

    /// <summary>Gets or sets the discretization points of consistency distillation (50).</summary>
    public int DistillationSteps { get; set; } = 50;

    /// <summary>Gets or sets the EMA momentum of the distillation target θ⁻ (0.95, Eq. 12).</summary>
    public double EmaDecay { get; set; } = 0.95;

    /// <summary>Gets or sets the training segment length in frames (2 seconds at 22.05 kHz / 256 = 172).</summary>
    public int SegmentFrames { get; set; } = 172;

    /// <summary>Gets or sets the duration multiplier at inference (1).</summary>
    public double LengthScale { get; set; } = 1.0;

    /// <summary>Gets or sets the seed of the training draws and the inference sample, for repeatable runs.</summary>
    public int SamplingSeed { get; set; }
}
