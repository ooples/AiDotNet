namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Grad-TTS (score-based diffusion decoder on a Glow-TTS-style text encoder).</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (Popov et al. 2021, §3–4) with unstated values from the reference implementation
/// (huawei-noah/Speech-Backbones, <c>Grad-TTS/params.py</c>): the Glow-TTS encoder (width 192, 6 relative-position
/// layers, 2 heads, window 4, filter 768, kernel 3, dropout 0.1; pre-net of 3 kernel-5 convolutions; duration
/// predictor filter 256); a U-Net score network with base width 64 and multipliers (1, 2, 4), time embedding scaled by
/// 1000; noise schedule β_t = 0.05 + (20 − 0.05) t on t ∈ [0, 1]; 2-second training segments (172 frames);
/// 22.05 kHz audio, 1024-point STFT, hop 256, 80 mel bins; inference with τ = 1.5 and N = 10 reverse steps.
/// </para>
/// <para><b>For Beginners:</b> These options configure the GradTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class GradTTSOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GradTTSOptions(GradTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        NumDiffusionSteps = other.NumDiffusionSteps;
        BetaMin = other.BetaMin;
        BetaMax = other.BetaMax;
        FilterChannels = other.FilterChannels;
        EncoderKernelSize = other.EncoderKernelSize;
        RelativeWindow = other.RelativeWindow;
        PrenetLayers = other.PrenetLayers;
        PrenetKernelSize = other.PrenetKernelSize;
        PrenetDropout = other.PrenetDropout;
        DurationPredictorFilterChannels = other.DurationPredictorFilterChannels;
        DecoderDimMultipliers = (int[])other.DecoderDimMultipliers.Clone();
        TimePositionScale = other.TimePositionScale;
        SegmentFrames = other.SegmentFrames;
        Temperature = other.Temperature;
        LengthScale = other.LengthScale;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public GradTTSOptions()
    {
        EncoderDim = 192;
        HiddenDim = 192;
        DecoderDim = 64;
        NumEncoderLayers = 6;
        NumHeads = 2;
        DropoutRate = 0.1;
        SampleRate = 22050;
        HopSize = 256;
        FftSize = 1024;
        MelChannels = 80;
        UsePostnet = false;
        LearningRate = 1e-4;
    }

    /// <summary>Gets or sets the reverse-diffusion steps N at inference (10; the paper evaluates 4, 10, 100, 1000).</summary>
    public int NumDiffusionSteps { get; set; } = 10;

    /// <summary>Gets or sets β₀ of the linear noise schedule (0.05).</summary>
    public double BetaMin { get; set; } = 0.05;

    /// <summary>Gets or sets β₁ of the linear noise schedule (20).</summary>
    public double BetaMax { get; set; } = 20.0;

    /// <summary>Gets or sets the encoder feed-forward filter (768).</summary>
    public int FilterChannels { get; set; } = 768;

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

    /// <summary>Gets or sets the U-Net's width multipliers per resolution (1, 2, 4); the base width is
    /// <see cref="AcousticModelOptions.DecoderDim"/> (64).</summary>
    public int[] DecoderDimMultipliers { get; set; } = { 1, 2, 4 };

    /// <summary>Gets or sets the scale of the diffusion time in its sinusoidal embedding (1000).</summary>
    public double TimePositionScale { get; set; } = 1000.0;

    /// <summary>Gets or sets the training segment length in frames (2 seconds at 22.05 kHz / 256 = 172).</summary>
    public int SegmentFrames { get; set; } = 172;

    /// <summary>Gets or sets the terminal-distribution temperature τ at inference (1.5).</summary>
    public double Temperature { get; set; } = 1.5;

    /// <summary>Gets or sets the duration multiplier at inference (1).</summary>
    public double LengthScale { get; set; } = 1.0;

    /// <summary>Gets or sets the seed of the training draws and the inference sample, for repeatable runs.</summary>
    public int SamplingSeed { get; set; }
}
