using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for Matcha-TTS (Mehta et al. 2024): a RoPE text encoder with duration predictor and an OT-CFM
/// U-Net decoder.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (§3–4) with the values it leaves to its reference implementation
/// (shivammehta25/Matcha-TTS, <c>configs/model/*.yaml</c>, <c>configs/data/ljspeech.yaml</c>,
/// <c>configs/trainer/default.yaml</c>): encoder width 192, 6 layers, 2 heads, filter 768, kernel 3, dropout 0.1, a
/// 3-layer kernel-5 pre-net with dropout 0.5, rotary embeddings on half of each head's channels; duration predictor
/// filter 256; decoder channels (256, 256), 2 mid blocks, 2 heads of 64, dropout 0.05, SnakeBeta feed-forward;
/// σ_min = 1e-4; Euler solver; Adam at 1e-4 with gradients clipped to norm 5; LJSpeech mel statistics
/// (mean −5.536622, std 2.116101); 22.05 kHz, 1024-point STFT, hop 256, 80 mel bins. Inference uses 10 Euler steps at
/// temperature 0.667 (the reference CLI's defaults; the paper evaluates 2, 4 and 10 steps).
/// </para>
/// <para><b>For Beginners:</b> These options configure the MatchaTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class MatchaTTSOptions : EndToEndTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MatchaTTSOptions(MatchaTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        FlowDim = other.FlowDim;
        DecoderLevels = other.DecoderLevels;
        DecoderMidBlocks = other.DecoderMidBlocks;
        DecoderHeads = other.DecoderHeads;
        DecoderHeadDim = other.DecoderHeadDim;
        DecoderDropout = other.DecoderDropout;
        SigmaMin = other.SigmaMin;
        EncoderKernelSize = other.EncoderKernelSize;
        PrenetLayers = other.PrenetLayers;
        PrenetKernelSize = other.PrenetKernelSize;
        PrenetDropout = other.PrenetDropout;
        DurationPredictorFilterChannels = other.DurationPredictorFilterChannels;
        MelMean = other.MelMean;
        MelStd = other.MelStd;
        GradientClipNorm = other.GradientClipNorm;
        Temperature = other.Temperature;
        LengthScale = other.LengthScale;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public MatchaTTSOptions()
    {
        NumFlowSteps = 10;
        NumEncoderLayers = 6;
        NumHeads = 2;
        DropoutRate = 0.1;
        HiddenDim = 192;
        EncoderDim = 192;
        FilterChannels = 768;
        SampleRate = 22050;
        HopSize = 256;
        FftSize = 1024;
        MelChannels = 80;
        LearningRate = 1e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the width of every decoder U-Net level (256; the reference's channels are (256, 256)).</summary>
    public int FlowDim { get; set; } = 256;

    /// <summary>Gets or sets the decoder U-Net's levels (2: one down-sampling, then a same-resolution level).</summary>
    public int DecoderLevels { get; set; } = 2;

    /// <summary>Gets or sets the decoder's middle blocks (2).</summary>
    public int DecoderMidBlocks { get; set; } = 2;

    /// <summary>Gets or sets the decoder Transformer's heads (2).</summary>
    public int DecoderHeads { get; set; } = 2;

    /// <summary>Gets or sets the decoder Transformer's per-head width (64).</summary>
    public int DecoderHeadDim { get; set; } = 64;

    /// <summary>Gets or sets the decoder's dropout (0.05).</summary>
    public double DecoderDropout { get; set; } = 0.05;

    /// <summary>Gets or sets σ_min of the OT-CFM probability path (1e-4).</summary>
    public double SigmaMin { get; set; } = 1e-4;

    /// <summary>Gets or sets the encoder feed-forward and duration predictor kernel (3).</summary>
    public int EncoderKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the pre-net's convolutions (3).</summary>
    public int PrenetLayers { get; set; } = 3;

    /// <summary>Gets or sets the pre-net's kernel (5).</summary>
    public int PrenetKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the pre-net's dropout (0.5).</summary>
    public double PrenetDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the duration predictor's filter (256).</summary>
    public int DurationPredictorFilterChannels { get; set; } = 256;

    /// <summary>Gets or sets the mel mean the model normalizes targets with (LJSpeech: −5.536622); compute it over
    /// the training set for other data.</summary>
    public double MelMean { get; set; } = -5.536622;

    /// <summary>Gets or sets the mel standard deviation the model normalizes targets with (LJSpeech: 2.116101).</summary>
    public double MelStd { get; set; } = 2.116101;

    /// <summary>Gets or sets the global gradient-norm clip (5, the reference trainer's <c>gradient_clip_val</c>).</summary>
    public double GradientClipNorm { get; set; } = 5.0;

    /// <summary>Gets or sets the noise temperature at inference (0.667).</summary>
    public double Temperature { get; set; } = 0.667;

    /// <summary>Gets or sets the duration multiplier at inference (1).</summary>
    public double LengthScale { get; set; } = 1.0;

    /// <summary>Gets or sets the seed of the training draws and the inference noise, for repeatable runs.</summary>
    public int SamplingSeed { get; set; }
}
