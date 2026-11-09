using AiDotNet.TextToSpeech.EndToEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for Gao et al.'s raw-waveform E3 TTS diffusion model.</summary>
/// <remarks>
/// The block arrays are Table 1 of the paper, flattened from block groups 0 through 3. E3 operates at 24 kHz and
/// reduces a 10.92-second waveform by 4096x before the lowest-resolution attention blocks.
/// </remarks>
public class E3TTSOptions : EndToEndTtsOptions
{
    private bool _channelWidthReady;

    /// <summary>Number of layers in the BERT text encoder. Shares the setting with <see cref="BertLayers"/>.</summary>
    public override int NumEncoderLayers { get => BertLayers; set => BertLayers = value; }

    /// <summary>Attention heads in the BERT text encoder. Shares the setting with <see cref="BertHeads"/>.</summary>
    public override int NumHeads { get => BertHeads; set => BertHeads = value; }

    /// <summary>Dropout probability in the BERT text encoder. Shares the setting with <see cref="BertDropout"/>.</summary>
    public override double DropoutRate { get => BertDropout; set => BertDropout = value; }

    /// <summary>Creates an instance with the default settings.</summary>
    public E3TTSOptions()
    {
        SampleRate = 24_000;
        HiddenDim = 1_024;
        EncoderDim = 768;
        DecoderDim = 1_024;
        NumDiffusionSteps = 1_000;
        MaxTextLength = 512;
        _channelWidthReady = true;
    }

    /// <summary>Initializes an independent copy.</summary>
    public E3TTSOptions(E3TTSOptions other) : base(other)
    {
        Validation.Guard.NotNull(other);
        BertVocabularySize = other.BertVocabularySize;
        BertHiddenDim = other.BertHiddenDim;
        BertLayers = other.BertLayers;
        BertHeads = other.BertHeads;
        BertIntermediateDim = other.BertIntermediateDim;
        BertMaxPositions = other.BertMaxPositions;
        BertDropout = other.BertDropout;
        FreezeBert = other.FreezeBert;
        BlockChannels = (int[])other.BlockChannels.Clone();
        BlockKernelSizes = (int[])other.BlockKernelSizes.Clone();
        BlockStrides = (int[])other.BlockStrides.Clone();
        BlockRepetitions = (int[])other.BlockRepetitions.Clone();
        AdaptiveKernelCounts = (int[])other.AdaptiveKernelCounts.Clone();
        AttentionHeads = (int[])other.AttentionHeads.Clone();
        GroupNormGroups = other.GroupNormGroups;
        GroupNormEpsilon = other.GroupNormEpsilon;
        TimeEmbeddingDim = other.TimeEmbeddingDim;
        ConditionDim = other.ConditionDim;
        SpeakerCount = other.SpeakerCount;
        DefaultSpeakerId = other.DefaultSpeakerId;
        TrainingWaveformSamples = other.TrainingWaveformSamples;
        InferenceWaveformSamples = other.InferenceWaveformSamples;
        MinimumAlphaBar = other.MinimumAlphaBar;
        TrainingBetaStart = other.TrainingBetaStart;
        TrainingBetaEnd = other.TrainingBetaEnd;
        PaddingLossWeight = other.PaddingLossWeight;
        SilenceBlockSize = other.SilenceBlockSize;
        SilenceMagnitudeThreshold = other.SilenceMagnitudeThreshold;
        SamplingSeed = other.SamplingSeed;
        _channelWidthReady = true;
    }

    /// <summary>
    /// Gets the peak U-Net channel width. Setting it scales every stage by the same ratio;
    /// the resulting widths must be positive integers. Use <see cref="BlockChannels"/> to configure stages independently.
    /// The default retains the 128/256/512/1024 widths in paper Table 1.
    /// </summary>
    public override int HiddenDim
    {
        get => _channelWidthReady && BlockChannels is { Length: > 0 } ? BlockChannels.Max() : base.HiddenDim;
        set
        {
            // Base copy constructors assign shared aliases before the stage arrays are copied.
            if (!_channelWidthReady) { base.HiddenDim = value; return; }
            if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "The U-Net width must be positive.");
            if (BlockChannels is null || BlockChannels.Length == 0 || BlockChannels.Any(width => width <= 0))
                throw new InvalidOperationException("Configure positive BlockChannels before scaling the U-Net width.");
            int peak = BlockChannels.Max();
            var scaled = new int[BlockChannels.Length];
            for (int i = 0; i < scaled.Length; i++)
            {
                long numerator = (long)BlockChannels[i] * value;
                if (numerator % peak != 0 || numerator / peak <= 0)
                    throw new ArgumentException("The width must preserve integral stage proportions; configure BlockChannels for other layouts.", nameof(value));
                scaled[i] = checked((int)(numerator / peak));
            }
            BlockChannels = scaled;
            base.HiddenDim = value;
        }
    }

    /// <summary>Gets or sets the BERT vocabulary size. Defaults to <c>30_522</c>.</summary>
    public int BertVocabularySize { get; set; } = 30_522;
    /// <summary>Gets or sets the feature width of the BERT hidden. Defaults to <c>768</c>.</summary>
    public int BertHiddenDim { get; set; } = 768;
    /// <summary>Gets or sets the number of layers in the BERT. Defaults to <c>12</c>.</summary>
    public int BertLayers { get; set; } = 12;
    /// <summary>Gets or sets the number of attention heads in the BERT. Defaults to <c>12</c>.</summary>
    public int BertHeads { get; set; } = 12;
    /// <summary>Gets or sets the feature width of the BERT intermediate. Defaults to <c>3_072</c>.</summary>
    public int BertIntermediateDim { get; set; } = 3_072;
    /// <summary>Gets or sets the BERT max positions. Defaults to <c>512</c>.</summary>
    public int BertMaxPositions { get; set; } = 512;
    /// <summary>Gets or sets the dropout probability for the BERT. Defaults to <c>0.1</c>.</summary>
    public double BertDropout { get; set; } = 0.1;

    /// <summary>The paper does not say that the English BERT checkpoint is fine-tuned; by default it is a frozen feature extractor.</summary>
    public bool FreezeBert { get; set; } = true;

    /// <summary>Output width of each DBlock in Table 1.</summary>
    public int[] BlockChannels { get; set; } = [128, 128, 256, 256, 512, 1_024, 1_024, 1_024, 1_024, 1_024];

    /// <summary>Convolution kernel of each DBlock/UBlock pair in Table 1.</summary>
    public int[] BlockKernelSizes { get; set; } = [5, 5, 5, 5, 5, 3, 3, 3, 3, 3];

    /// <summary>Temporal stride of each DBlock/UBlock pair in Table 1; the product is 4096.</summary>
    public int[] BlockStrides { get; set; } = [2, 2, 2, 2, 4, 4, 2, 2, 2, 2];

    /// <summary>Residual blocks at each resolution, from Table 1's Blocks row.</summary>
    public int[] BlockRepetitions { get; set; } = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1];

    /// <summary>Number of softmax kernel experts; zero means the block uses FiLM instead.</summary>
    public int[] AdaptiveKernelCounts { get; set; } = [8, 8, 4, 4, 2, 0, 0, 0, 0, 0];

    /// <summary>Self/cross-attention heads per block; zero means no attention.</summary>
    public int[] AttentionHeads { get; set; } = [0, 0, 0, 0, 0, 8, 8, 8, 8, 8];

    /// <summary>Gets or sets the group norm groups. Defaults to <c>32</c>.</summary>
    public int GroupNormGroups { get; set; } = 32;
    /// <summary>Gets or sets the numerical stability constant for group norm. Defaults to <c>1e-5</c>.</summary>
    public double GroupNormEpsilon { get; set; } = 1e-5;
    /// <summary>Gets or sets the feature width of the time embedding. Defaults to <c>256</c>.</summary>
    public int TimeEmbeddingDim { get; set; } = 256;
    /// <summary>Gets or sets the feature width of the condition. Defaults to <c>1_024</c>.</summary>
    public int ConditionDim { get; set; } = 1_024;
    /// <summary>Gets or sets the speaker count. Defaults to <c>1</c>.</summary>
    public int SpeakerCount { get; set; } = 1;
    /// <summary>Gets or sets the token ID for default speaker.</summary>
    public int DefaultSpeakerId { get; set; }

    /// <summary>10.92 seconds at 24 kHz, as used during training.</summary>
    public int TrainingWaveformSamples { get; set; } = 262_080;

    /// <summary>Fixed inference canvas; trailing low-magnitude blocks are removed after sampling.</summary>
    public int InferenceWaveformSamples { get; set; } = 262_080;

    /// <summary>Terminal cumulative alpha in equation 7.</summary>
    public double MinimumAlphaBar { get; set; } = 1e-7;

    /// <summary>First beta of the linear training schedule (Eq. 4). The paper does not disclose endpoints;
    /// 1e-4 and 0.02 are configurable DDPM defaults, not verified E3 author hyperparameters.</summary>
    public double TrainingBetaStart { get; set; } = 1e-4;

    /// <summary>Last beta of the linear training schedule, separate from the Eq. 7 inference schedule.</summary>
    public double TrainingBetaEnd { get; set; } = 0.02;

    /// <summary>Paper section 3.3 weights padded samples one tenth as much as real samples.</summary>
    public double PaddingLossWeight { get; set; } = 0.1;
    /// <summary>Gets or sets the silence block size. Defaults to <c>1_024</c>.</summary>
    public int SilenceBlockSize { get; set; } = 1_024;
    /// <summary>Gets or sets the silence magnitude threshold. Defaults to <c>0.02</c>.</summary>
    public double SilenceMagnitudeThreshold { get; set; } = 0.02;
    /// <summary>Gets or sets the random seed used during sampling. Defaults to <c>17</c>.</summary>
    public int SamplingSeed { get; set; } = 17;

    internal void Validate()
    {
        if (double.IsNaN(TrainingBetaStart) || double.IsInfinity(TrainingBetaStart)
            || double.IsNaN(TrainingBetaEnd) || double.IsInfinity(TrainingBetaEnd)
            || TrainingBetaStart <= 0 || TrainingBetaEnd >= 1 || TrainingBetaEnd < TrainingBetaStart)
            throw new ArgumentException("E3's linear training betas must satisfy 0 < start <= end < 1.");
        int count = BlockChannels.Length;
        if (count == 0 || BlockKernelSizes.Length != count || BlockStrides.Length != count ||
            AdaptiveKernelCounts.Length != count || AttentionHeads.Length != count || BlockRepetitions.Length != count)
            throw new ArgumentException("E3 block configuration arrays must be non-empty and have equal lengths.");
        if (BlockChannels.Any(x => x <= 0) || BlockKernelSizes.Any(x => x <= 0 || x % 2 == 0) || BlockStrides.Any(x => x <= 0) || BlockRepetitions.Any(x => x <= 0))
            throw new ArgumentException("E3 channels, odd kernels and strides must be positive.");
        if (AttentionHeads.Where((h, i) => h > 0 && BlockChannels[i] % h != 0).Any())
            throw new ArgumentException("Every attention block width must divide by its head count.");
        if (BertHiddenDim % BertHeads != 0) throw new ArgumentException("BERT width must divide by its head count.");
        int factor = BlockStrides.Aggregate(1, (a, b) => checked(a * b));
        if (TrainingWaveformSamples <= 0 || InferenceWaveformSamples <= 0)
            throw new ArgumentException("E3 waveform lengths must be positive; the U-Net pads its internal canvas to its stride.");
        if (NumDiffusionSteps <= 0 || MinimumAlphaBar <= 0 || MinimumAlphaBar >= 1)
            throw new ArgumentException("E3 needs positive diffusion steps and a terminal alpha in (0, 1).");
        if (PaddingLossWeight <= 0 || PaddingLossWeight > 1) throw new ArgumentException("PaddingLossWeight must be in (0, 1].");
        if (SpeakerCount <= 0 || DefaultSpeakerId < 0 || DefaultSpeakerId >= SpeakerCount)
            throw new ArgumentException("The default speaker id must address the speaker table.");
    }
}
