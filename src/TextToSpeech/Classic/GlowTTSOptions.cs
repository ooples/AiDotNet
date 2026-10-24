namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Glow-TTS (flow-based TTS with monotonic alignment search).</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's LJSpeech configuration (Kim et al. 2020, §3.3, §4, App. A.1, Table 3) as released in the
/// reference implementation's <c>configs/base.json</c>: encoder width 192, 6 relative-position Transformer layers
/// with 2 heads, maximum relative position 4, feed-forward filter 768 with kernel 3, dropout 0.1; a pre-net of three
/// kernel-5 convolutions with dropout 0.5; a duration predictor with filter 256; a decoder of 12 flow blocks over pairs
/// of frames squeezed into 160 channels, each an ActNorm, an invertible 1×1 convolution mixing groups of 4 channels and
/// an affine coupling layer with a 4-layer WaveNet-like network (width 192, kernel 5, dilation 1, dropout 0.05);
/// 22.05 kHz audio, 1024-point STFT, hop 256, 80 mel bins; sampling temperature 0.333 (§5.1).
/// </para>
/// <para><b>For Beginners:</b> These options configure the GlowTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class GlowTTSOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GlowTTSOptions(GlowTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        FilterChannels = other.FilterChannels;
        EncoderKernelSize = other.EncoderKernelSize;
        RelativeWindow = other.RelativeWindow;
        PrenetLayers = other.PrenetLayers;
        PrenetKernelSize = other.PrenetKernelSize;
        PrenetDropout = other.PrenetDropout;
        DurationPredictorFilterChannels = other.DurationPredictorFilterChannels;
        NumFlowBlocks = other.NumFlowBlocks;
        SqueezeFactor = other.SqueezeFactor;
        InvertibleConvGroupSize = other.InvertibleConvGroupSize;
        DecoderHiddenChannels = other.DecoderHiddenChannels;
        DecoderKernelSize = other.DecoderKernelSize;
        DecoderDilationRate = other.DecoderDilationRate;
        CouplingLayers = other.CouplingLayers;
        DecoderDropout = other.DecoderDropout;
        Temperature = other.Temperature;
        LengthScale = other.LengthScale;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public GlowTTSOptions()
    {
        EncoderDim = 192;
        HiddenDim = 192;
        NumEncoderLayers = 6;
        NumHeads = 2;
        DropoutRate = 0.1;
        SampleRate = 22050;
        HopSize = 256;
        FftSize = 1024;
        MelChannels = 80;
        UsePostnet = false;
    }

    /// <summary>Gets or sets the encoder feed-forward filter (768).</summary>
    public int FilterChannels { get; set; } = 768;

    /// <summary>Gets or sets the encoder feed-forward and duration predictor kernel (3).</summary>
    public int EncoderKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the maximum relative position of the encoder's attention (4).</summary>
    public int RelativeWindow { get; set; } = 4;

    /// <summary>Gets or sets the pre-net's convolutions (3).</summary>
    public int PrenetLayers { get; set; } = 3;

    /// <summary>Gets or sets the pre-net's kernel (5).</summary>
    public int PrenetKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the pre-net's dropout (0.5).</summary>
    public double PrenetDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the duration predictor's filter (256).</summary>
    public int DurationPredictorFilterChannels { get; set; } = 256;

    /// <summary>Gets or sets the decoder's flow blocks (12).</summary>
    public int NumFlowBlocks { get; set; } = 12;

    /// <summary>Gets or sets how many frames the decoder squeezes into channels (2).</summary>
    public int SqueezeFactor { get; set; } = 2;

    /// <summary>Gets or sets the channel group size of the invertible 1×1 convolutions (4).</summary>
    public int InvertibleConvGroupSize { get; set; } = 4;

    /// <summary>Gets or sets the width of the coupling layers' WaveNet-like networks (192).</summary>
    public int DecoderHiddenChannels { get; set; } = 192;

    /// <summary>Gets or sets the coupling layers' convolution kernel (5).</summary>
    public int DecoderKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the coupling layers' dilation base (1).</summary>
    public int DecoderDilationRate { get; set; } = 1;

    /// <summary>Gets or sets the gated layers per coupling network (4).</summary>
    public int CouplingLayers { get; set; } = 4;

    /// <summary>Gets or sets the coupling networks' dropout (0.05).</summary>
    public double DecoderDropout { get; set; } = 0.05;

    /// <summary>Gets or sets the prior's sampling temperature at inference (0.333, the paper's best).</summary>
    public double Temperature { get; set; } = 0.333;

    /// <summary>Gets or sets the duration multiplier at inference (1).</summary>
    public double LengthScale { get; set; } = 1.0;

    /// <summary>Gets or sets the seed of the prior sample drawn at inference, so synthesis is repeatable.</summary>
    public int SamplingSeed { get; set; }
}
