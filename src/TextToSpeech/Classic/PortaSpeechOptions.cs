namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for PortaSpeech (Ren et al. 2021): a mixture-alignment linguistic encoder, a VAE variational
/// generator with a flow prior, and a flow post-net with grouped parameter sharing.</summary>
/// <remarks>
/// <para>
/// Defaults are PortaSpeech (normal), Table 6, with values the paper leaves to its reference implementation
/// (NATSpeech, <c>egs/egs_bases/tts/{fs,ps,ps_flow}.yaml</c>): linguistic encoder width 192, 4 phoneme and 4 word
/// relative-position layers (2 heads, window 4, kernel 5, filter 768, dropout 0); duration predictor kernel 5, dropout
/// 0.2, input gradient scale 0.1; variational generator 192 channels, latent 16, kernel 5, 8 encoder and 4 decoder
/// WaveNet layers, stride 4; VP-flow prior of 4 steps × 4 layers, 64 channels, kernel 3; post-net of 12 Glow steps,
/// 3 WaveNet layers, kernel 3, 192 channels, sharing in groups of 4 (3 groups), temperature 0.8; KL counted after
/// 10 000 updates; post-net learning rate 1e-3; gradient-norm clip 1; frames a multiple of 4; 22.05 kHz, 1024-point
/// STFT, hop 256, 80 mel bins.
/// </para>
/// <para><b>For Beginners:</b> These options configure the PortaSpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class PortaSpeechOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PortaSpeechOptions(PortaSpeechOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        NumFlowLayers = other.NumFlowLayers;
        ProsodyDim = other.ProsodyDim;
        DurationScale = other.DurationScale;
        NumWordEncoderLayers = other.NumWordEncoderLayers;
        FilterChannels = other.FilterChannels;
        EncoderKernelSize = other.EncoderKernelSize;
        RelativeWindow = other.RelativeWindow;
        PrenetLayers = other.PrenetLayers;
        PrenetKernelSize = other.PrenetKernelSize;
        DurationPredictorKernelSize = other.DurationPredictorKernelSize;
        DurationPredictorDropout = other.DurationPredictorDropout;
        DurationPredictorGradientScale = other.DurationPredictorGradientScale;
        AttentionHeads = other.AttentionHeads;
        GeneratorChannels = other.GeneratorChannels;
        GeneratorKernelSize = other.GeneratorKernelSize;
        GeneratorEncoderLayers = other.GeneratorEncoderLayers;
        GeneratorDecoderLayers = other.GeneratorDecoderLayers;
        GeneratorStride = other.GeneratorStride;
        PriorFlowSteps = other.PriorFlowSteps;
        PriorFlowLayers = other.PriorFlowLayers;
        PriorFlowChannels = other.PriorFlowChannels;
        PriorFlowKernelSize = other.PriorFlowKernelSize;
        PostNetChannels = other.PostNetChannels;
        PostNetKernelSize = other.PostNetKernelSize;
        PostNetLayers = other.PostNetLayers;
        PostNetShareGroupSize = other.PostNetShareGroupSize;
        PostNetTemperature = other.PostNetTemperature;
        PostNetLearningRate = other.PostNetLearningRate;
        KlStartUpdates = other.KlStartUpdates;
        GradientClipNorm = other.GradientClipNorm;
        FramesMultiple = other.FramesMultiple;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates PortaSpeech (normal).</summary>
    public PortaSpeechOptions()
    {
        EncoderDim = 192;
        DecoderDim = 192;
        HiddenDim = 192;
        NumEncoderLayers = 4;
        NumDecoderLayers = 4;
        NumHeads = 2;
        DropoutRate = 0.0;
        SampleRate = 22050;
        HopSize = 256;
        FftSize = 1024;
        MelChannels = 80;
        UsePostnet = false;
        LearningRate = 2e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the post-net's flow steps (12).</summary>
    public int NumFlowLayers { get; set; } = 12;

    /// <summary>Gets or sets the variational generator's latent channels (16).</summary>
    public int ProsodyDim { get; set; } = 16;

    /// <summary>Gets or sets the multiplier on predicted word durations at inference (1).</summary>
    public double DurationScale { get; set; } = 1.0;

    /// <summary>Gets or sets the word encoder's layers (4).</summary>
    public int NumWordEncoderLayers { get; set; } = 4;

    /// <summary>Gets or sets the encoders' feed-forward filter (768).</summary>
    public int FilterChannels { get; set; } = 768;

    /// <summary>Gets or sets the encoders' feed-forward kernel (5).</summary>
    public int EncoderKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the encoders' maximum relative position (4).</summary>
    public int RelativeWindow { get; set; } = 4;

    /// <summary>Gets or sets the encoders' pre-net convolutions (3).</summary>
    public int PrenetLayers { get; set; } = 3;

    /// <summary>Gets or sets the encoders' pre-net kernel (5).</summary>
    public int PrenetKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the duration predictor's kernel (5).</summary>
    public int DurationPredictorKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the duration predictor's dropout (0.2).</summary>
    public double DurationPredictorDropout { get; set; } = 0.2;

    /// <summary>Gets or sets the scale on the gradient the duration loss sends into the encoder (0.1).</summary>
    public double DurationPredictorGradientScale { get; set; } = 0.1;

    /// <summary>Gets or sets the word-to-phoneme attention heads (2).</summary>
    public int AttentionHeads { get; set; } = 2;

    /// <summary>Gets or sets the variational generator's channels (192).</summary>
    public int GeneratorChannels { get; set; } = 192;

    /// <summary>Gets or sets the variational generator's WaveNet kernel (5).</summary>
    public int GeneratorKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the variational generator's encoder WaveNet layers (8).</summary>
    public int GeneratorEncoderLayers { get; set; } = 8;

    /// <summary>Gets or sets the variational generator's decoder WaveNet layers (4).</summary>
    public int GeneratorDecoderLayers { get; set; } = 4;

    /// <summary>Gets or sets the variational generator's time stride (4).</summary>
    public int GeneratorStride { get; set; } = 4;

    /// <summary>Gets or sets the VP-flow prior's coupling steps (4).</summary>
    public int PriorFlowSteps { get; set; } = 4;

    /// <summary>Gets or sets the VP-flow prior's WaveNet layers per step (4).</summary>
    public int PriorFlowLayers { get; set; } = 4;

    /// <summary>Gets or sets the VP-flow prior's channels (64).</summary>
    public int PriorFlowChannels { get; set; } = 64;

    /// <summary>Gets or sets the VP-flow prior's kernel (3).</summary>
    public int PriorFlowKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the post-net's WaveNet channels (192).</summary>
    public int PostNetChannels { get; set; } = 192;

    /// <summary>Gets or sets the post-net's WaveNet kernel (3).</summary>
    public int PostNetKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the post-net's WaveNet layers per step (3).</summary>
    public int PostNetLayers { get; set; } = 3;

    /// <summary>Gets or sets how many consecutive post-net steps share their WaveNet (4: 12 steps in 3 groups).</summary>
    public int PostNetShareGroupSize { get; set; } = 4;

    /// <summary>Gets or sets the post-net's sampling temperature (0.8).</summary>
    public double PostNetTemperature { get; set; } = 0.8;

    /// <summary>Gets or sets the post-net phase's learning rate (1e-3).</summary>
    public double PostNetLearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the generator updates before the KL term is optimized (10 000).</summary>
    public int KlStartUpdates { get; set; } = 10000;

    /// <summary>Gets or sets the global gradient-norm clip (1).</summary>
    public double GradientClipNorm { get; set; } = 1.0;

    /// <summary>Gets or sets the frame-count multiple (4, the generator stride).</summary>
    public int FramesMultiple { get; set; } = 4;

    /// <summary>Gets or sets the seed of the training noise and the inference samples, for repeatable runs.</summary>
    public int SamplingSeed { get; set; }
}
