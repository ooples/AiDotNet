using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for VoiceFlow (Guo et al. 2024): Grad-TTS's encoder with ground-truth durations and a rectified
/// flow-matching U-Net decoder.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (§3–4.1) with the values it leaves to its reference implementation
/// (cantabile-kwok/VoiceFlow-TTS, <c>configs/lj_16k_gt_dur.yaml</c>, <c>model/cfm.py</c>): Grad-TTS's encoder (width
/// 192, 6 relative-position layers, 2 heads, window 4, filter 768, kernel 3, dropout 0.1, pre-net of 3 kernel-5
/// convolutions; output projected to the mel width) and duration predictor (filter 256); Grad-TTS's U-Net (base width
/// 128, multipliers 1, 2, 4, time embedding scale 1000) conditioned on the duplicated encoder output; σ = 0.1; t clamped
/// to [1e-5, 1 − 1e-5]; 2-second segments (160 frames); Adam at 5e-5 with encoder and decoder gradients each clipped
/// to norm 1; 16 kHz, 12.5 ms hop (200), 50 ms window (800), 80 mel bins; Euler sampling.
/// </para>
/// <para><b>For Beginners:</b> These options configure the VoiceFlow model. Default values follow the original paper settings.</para>
/// </remarks>
public class VoiceFlowOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VoiceFlowOptions(VoiceFlowOptions other)
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
        Sigma = other.Sigma;
        SegmentFrames = other.SegmentFrames;
        LengthScale = other.LengthScale;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public VoiceFlowOptions()
    {
        HiddenDim = 192;
        EncoderDim = 192;
        FilterChannels = 768;
        NumEncoderLayers = 6;
        NumHeads = 2;
        DropoutRate = 0.1;
        NumFlowSteps = 2;
        SampleRate = 16000;
        HopSize = 200;
        FftSize = 1024;
        MelChannels = 80;
        LearningRate = 5e-5;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the U-Net's base width (128).</summary>
    public int FlowDim { get; set; } = 128;

    /// <summary>Gets or sets the U-Net's width multipliers per resolution (1, 2, 4).</summary>
    public int[] DecoderDimMultipliers { get; set; } = { 1, 2, 4 };

    /// <summary>Gets or sets the scale of t in the U-Net's sinusoidal time embedding (1000).</summary>
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

    /// <summary>Gets or sets σ, the width of the conditional probability path (0.1, Eq. 4).</summary>
    public double Sigma { get; set; } = 0.1;

    /// <summary>Gets or sets the training segment length in frames (2 seconds at 16 kHz / 200 = 160).</summary>
    public int SegmentFrames { get; set; } = 160;

    /// <summary>Gets or sets the duration multiplier at inference (1).</summary>
    public double LengthScale { get; set; } = 1.0;

    /// <summary>Gets or sets the seed of the training draws and the inference noise, for repeatable runs.</summary>
    public int SamplingSeed { get; set; }
}
