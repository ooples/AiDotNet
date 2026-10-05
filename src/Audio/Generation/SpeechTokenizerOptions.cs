using AiDotNet.Audio.Codecs;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// Options for SpeechTokenizer ("SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models", Zhang et al.,
/// ICLR 2024). The defaults are the paper's: EnCodec's SEANet at C = 32 with a two-layer BiLSTM in the encoder, strides
/// (2, 4, 5, 8) at 16 kHz (50 frames per second), eight 1024-entry codebooks whose first is distilled toward HuBERT
/// representations. <see cref="OfficialCheckpoint"/> is the released <c>speechtokenizer_hubert_avg</c> configuration.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SpeechTokenizer turns speech into eight streams of codes: the first carries what is said,
/// the other seven add who said it and how. Speech language models generate the first stream from text and the rest from
/// the first.</para>
/// </remarks>
public class SpeechTokenizerOptions : NeuralAudioCodecOptions
{
    /// <summary>Creates the paper's configuration.</summary>
    public SpeechTokenizerOptions()
    {
        SampleRate = 16000;
        Channels = 1;
        NumQuantizers = 8;
        CodebookSize = 1024;
        TargetBandwidthKbps = 4.0;          // 8 codebooks × 50 frames × 10 bits
        SegmentSize = 48000;                // 3 s random crops (§4.1)
    }

    /// <summary>Gets or sets the base channel count C (32; App. D). The released checkpoint uses 64.</summary>
    public int Filters { get; set; } = 32;

    /// <summary>Gets or sets the strides in decoder order (8, 5, 4, 2); the encoder applies them reversed (2, 4, 5, 8).</summary>
    public int[] Ratios { get; set; } = [8, 5, 4, 2];

    /// <summary>Gets or sets the latent dimension D (the paper leaves it unstated; 1024, the reference's).</summary>
    public int Dimension { get; set; } = 1024;

    /// <summary>Gets or sets the semantic teacher's feature dimension (768: HuBERT base).</summary>
    public int SemanticDimension { get; set; } = 768;

    /// <summary>Gets or sets the encoder's BiLSTM layers (2).</summary>
    public int LstmLayers { get; set; } = 2;

    /// <summary>Gets or sets whether the codebook commitment loss is the reference's mean squared error (true: the λ below
    /// were tuned for it) rather than the sum of squared norms of §3.3.</summary>
    public bool CommitmentAsMean { get; set; } = true;

    /// <summary>Gets or sets the codebooks' EMA decay (0.99).</summary>
    public double CodebookDecay { get; set; } = 0.99;

    /// <summary>Gets or sets the k-means iterations of the first-batch codebook initialization (50).</summary>
    public int KMeansIterations { get; set; } = 50;

    /// <summary>Gets or sets the EMA usage below which a code is replaced by a batch vector (2).</summary>
    public int DeadCodeThreshold { get; set; } = 2;

    // ---------------------------------------------------------------- losses (§3.2–3.3; weights from the reference)

    /// <summary>Gets or sets λ_t on the mean absolute time-domain error (500).</summary>
    public double TimeLossWeight { get; set; } = 500.0;

    /// <summary>Gets or sets λ_f on the multi-scale mel loss (45).</summary>
    public double FrequencyLossWeight { get; set; } = 45.0;

    /// <summary>Gets or sets λ_g (1).</summary>
    public double AdversarialLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets λ_feat (1).</summary>
    public double FeatureLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets λ_w (10).</summary>
    public double CommitmentLossWeight { get; set; } = 10.0;

    /// <summary>Gets or sets λ_distill (120).</summary>
    public double DistillationLossWeight { get; set; } = 120.0;

    /// <summary>Gets or sets the mel losses' scales i (window 2^i, hop 2^i / 4): 5 … 11.</summary>
    public int[] MelScales { get; set; } = [5, 6, 7, 8, 9, 10, 11];

    /// <summary>Gets or sets the mel bins (64).</summary>
    public int MelBins { get; set; } = 64;

    // ---------------------------------------------------------------- discriminators (§3.3, App. D)

    /// <summary>Gets or sets the MS-STFT discriminator's windows (1024, 2048, 512, 256, 128; hop a quarter; reference).</summary>
    public int[] StftDiscriminatorWindows { get; set; } = [1024, 2048, 512, 256, 128];

    /// <summary>Gets or sets the MS-STFT discriminator's channels (32).</summary>
    public int StftDiscriminatorFilters { get; set; } = 32;

    /// <summary>Gets or sets the multi-period discriminator's periods (2, 3, 5, 7, 11).</summary>
    public int[] DiscriminatorPeriods { get; set; } = [2, 3, 5, 7, 11];

    /// <summary>Gets or sets a divisor on the period and scale discriminators' widths (1).</summary>
    public int DiscriminatorWidthDivisor { get; set; } = 1;

    // ---------------------------------------------------------------- optimizer (§4.1; betas and decay from the reference)

    /// <summary>Gets or sets Adam's learning rate (4e-4: the paper's maximum).</summary>
    public double LearningRate { get; set; } = 4e-4;

    /// <summary>Gets or sets Adam's β1 (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets Adam's β2 (0.99).</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>Gets or sets the learning rate's decay per epoch (0.98).</summary>
    public double LearningRateDecay { get; set; } = 0.98;

    /// <summary>Gets or sets the optimizer steps per epoch over which the decay is applied (1000).</summary>
    public int UpdatesPerEpoch { get; set; } = 1000;

    /// <summary>The released <c>speechtokenizer_hubert_avg</c> configuration: C = 64.</summary>
    public static SpeechTokenizerOptions OfficialCheckpoint() => new() { Filters = 64 };
}
