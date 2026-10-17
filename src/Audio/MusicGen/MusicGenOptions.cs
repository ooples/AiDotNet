using AiDotNet.Audio.Generation;

namespace AiDotNet.Audio.MusicGen;

/// <summary>
/// Options for MusicGen (Copet et al. 2023, "Simple and Controllable Music Generation"): a single-stage transformer
/// language model over the four codebooks of a 32 kHz EnCodec, interleaved by the "delay" pattern and conditioned on
/// T5 text embeddings through cross-attention.
/// </summary>
/// <remarks>
/// <para>
/// From the paper (§2–3): EnCodec at 32 kHz with stride 640 (50 Hz), four RVQ codebooks of 2048; the delay pattern;
/// pre-norm layers of causal self-attention, cross-attention and a D → 4D → D ReLU block; a frozen T5 text encoder;
/// AdamW (β = 0.9, 0.95), decoupled weight decay 0.1, gradient clipping 1.0, a cosine schedule with 4000 warmup steps,
/// 1M steps, D-Adaptation for the 300M model, an EMA of decay 0.99; condition merging 0.25, description dropout 0.5
/// and word dropout 0.3; condition dropout 0.2 and classifier-free guidance 3.0; top-k 250 at temperature 1.0.
/// </para>
/// <para>What the paper leaves open follows audiocraft (<c>config/model/lm/musicgen_lm.yaml</c>, its model scales,
/// <c>conditioner/text2music.yaml</c>, <c>solver/musicgen</c>): the widths and depths of the three sizes, t5-base as
/// the text encoder, bias-free linear layers, the sinusoidal step embedding, the learning rate 1e-4 of the
/// AdamW-trained sizes, the EMA taken every 10 updates, the dropout of other metadata fields 0.5 and 30-second
/// generation.</para>
/// <para><b>For Beginners:</b> MusicGen writes music the way a language model writes text, one short frame of
/// compressed audio at a time, guided by a description such as "an upbeat 80s synth-pop track".</para>
/// </remarks>
public class MusicGenOptions : AiDotNet.Models.Options.ModelOptions
{
    /// <summary>Creates the paper's 1.5B ("medium") configuration.</summary>
    public MusicGenOptions()
    {
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    public MusicGenOptions(MusicGenOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        ModelSize = other.ModelSize;
        ModelDim = other.ModelDim;
        NumHeads = other.NumHeads;
        NumLayers = other.NumLayers;
        FeedForwardScale = other.FeedForwardScale;
        DropoutRate = other.DropoutRate;
        Delays = other.Delays is null ? null : (int[])other.Delays.Clone();
        Stereo = other.Stereo;
        StereoPattern = other.StereoPattern;
        Codec = (EnCodecOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(other.Codec);
        TextModelDim = other.TextModelDim;
        TextLayers = other.TextLayers;
        TextHeads = other.TextHeads;
        TextKeyValueDim = other.TextKeyValueDim;
        TextFeedForwardDim = other.TextFeedForwardDim;
        TextVocabularySize = other.TextVocabularySize;
        FreezeTextEncoder = other.FreezeTextEncoder;
        MaxTextLength = other.MaxTextLength;
        DurationSeconds = other.DurationSeconds;
        Temperature = other.Temperature;
        TopK = other.TopK;
        TopP = other.TopP;
        GuidanceScale = other.GuidanceScale;
        UseSampling = other.UseSampling;
        ConditionDropout = other.ConditionDropout;
        WordDropout = other.WordDropout;
        MergeMetadataProbability = other.MergeMetadataProbability;
        DropDescriptionProbability = other.DropDescriptionProbability;
        KeepMetadataFieldProbability = other.KeepMetadataFieldProbability;
        LearningRate = other.LearningRate;
        UseDAdaptation = other.UseDAdaptation;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        WeightDecay = other.WeightDecay;
        GradientClipNorm = other.GradientClipNorm;
        WarmupSteps = other.WarmupSteps;
        TrainingSteps = other.TrainingSteps;
        EmaDecay = other.EmaDecay;
        EmaEveryUpdates = other.EmaEveryUpdates;
        MelodyConditioning = other.MelodyConditioning;
        MelodyDropout = other.MelodyDropout;
        ChromaWindow = other.ChromaWindow;
        ChromaHop = other.ChromaHop;
        Seed = other.Seed;
    }

    /// <summary>Gets or sets the size preset (Medium: 1.5B parameters).</summary>
    public MusicGenModelSize ModelSize { get; set; } = MusicGenModelSize.Medium;

    /// <summary>Gets or sets the model width D; 0 takes the preset's (1024, 1536, 2048).</summary>
    public int ModelDim { get; set; }

    /// <summary>Gets or sets the attention heads; 0 takes the preset's (16, 24, 32).</summary>
    public int NumHeads { get; set; }

    /// <summary>Gets or sets the transformer layers L; 0 takes the preset's (24, 48, 48).</summary>
    public int NumLayers { get; set; }

    /// <summary>Gets or sets the feed-forward width as a multiple of D (4, §2.4).</summary>
    public int FeedForwardScale { get; set; } = 4;

    /// <summary>Gets or sets the dropout inside the transformer (0, the reference's).</summary>
    public double DropoutRate { get; set; }

    /// <summary>Gets or sets each modelled codebook's delay; null takes the pattern's ([0, 1, 2, 3], Eq. 6; for stereo,
    /// the <see cref="StereoPattern"/>'s over the interleaved left and right codebooks).</summary>
    public int[]? Delays { get; set; }

    /// <summary>Gets or sets whether the model generates stereo (§4.3): EnCodec codes each channel separately, giving
    /// twice the codebooks, interleaved left, right per level (audiocraft <c>InterleaveStereoCompressionModel</c>).</summary>
    public bool Stereo { get; set; }

    /// <summary>Gets or sets the stereo codebook pattern (partial delay: the better rated, Table 3).</summary>
    public MusicGenStereoPattern StereoPattern { get; set; } = MusicGenStereoPattern.PartialDelay;

    /// <summary>The delays in effect: <see cref="Delays"/>, or the pattern's.</summary>
    public int[] ResolveDelays()
    {
        if (Delays is not null) return Delays;
        int k = Codec.NumQuantizers;
        if (!Stereo) return Enumerable.Range(0, k).ToArray();
        // Interleaved [L0, R0, L1, R1, ...]: the partial delay gives both channels of a level the same delay; the
        // stereo delay also delays the right channel by one step behind the left (Figure 2a).
        return Enumerable.Range(0, 2 * k).Select(i => StereoPattern == MusicGenStereoPattern.PartialDelay ? i / 2 : i).ToArray();
    }

    /// <summary>Gets or sets the audio tokenizer (EnCodec 32 kHz, §3.1).</summary>
    public EnCodecOptions Codec { get; set; } = EnCodec32kHz();

    /// <summary>Gets or sets the T5 encoder width (768, t5-base).</summary>
    public int TextModelDim { get; set; } = 768;

    /// <summary>Gets or sets the T5 encoder layers (12, t5-base).</summary>
    public int TextLayers { get; set; } = 12;

    /// <summary>Gets or sets the T5 attention heads (12, t5-base).</summary>
    public int TextHeads { get; set; } = 12;

    /// <summary>Gets or sets the T5 key/value width per head (64, t5-base).</summary>
    public int TextKeyValueDim { get; set; } = 64;

    /// <summary>Gets or sets the T5 feed-forward width (3072, t5-base).</summary>
    public int TextFeedForwardDim { get; set; } = 3072;

    /// <summary>Gets or sets the T5 embedding rows (32128, t5-base; its tokenizer uses the first 32100).</summary>
    public int TextVocabularySize { get; set; } = 32128;

    /// <summary>Gets or sets whether the T5 encoder stays fixed during training (true: "pretrained", §2.3, and the
    /// reference's <c>finetune: false</c>).</summary>
    public bool FreezeTextEncoder { get; set; } = true;

    /// <summary>Gets or sets the most text tokens read (512, T5's).</summary>
    public int MaxTextLength { get; set; } = 512;

    /// <summary>Gets or sets the default generation length in seconds (30, the reference's).</summary>
    public double DurationSeconds { get; set; } = 30.0;

    /// <summary>Gets or sets the sampling temperature (1.0, §3.1).</summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>Gets or sets top-k sampling's k (250, §3.1).</summary>
    public int TopK { get; set; } = 250;

    /// <summary>Gets or sets nucleus sampling's p; 0 uses top-k (0).</summary>
    public double TopP { get; set; }

    /// <summary>Gets or sets the classifier-free guidance scale (3.0, §3.1).</summary>
    public double GuidanceScale { get; set; } = 3.0;

    /// <summary>Gets or sets whether generation samples (true) or takes the most likely code.</summary>
    public bool UseSampling { get; set; } = true;

    /// <summary>Gets or sets the probability of training a step without the condition (0.2, §3.1).</summary>
    public double ConditionDropout { get; set; } = 0.2;

    /// <summary>Gets or sets the probability of dropping each word of the description (0.3, §3.1).</summary>
    public double WordDropout { get; set; } = 0.3;

    /// <summary>Gets or sets the probability of merging the metadata into the description (0.25, §3.1).</summary>
    public double MergeMetadataProbability { get; set; } = 0.25;

    /// <summary>Gets or sets the probability of dropping the description when metadata is merged (0.5, §3.1).</summary>
    public double DropDescriptionProbability { get; set; } = 0.5;

    /// <summary>Gets or sets the probability of keeping each metadata field in a merge (0.5, the reference's
    /// <c>drop_other_p</c>).</summary>
    public double KeepMetadataFieldProbability { get; set; } = 0.5;

    /// <summary>Gets or sets the learning rate; null takes the size's (1 for the D-Adaptation-trained Small, the
    /// D-Adaptation multiplier; 1e-4 otherwise).</summary>
    public double? LearningRate { get; set; }

    /// <summary>Gets or sets whether training uses D-Adaptation; null uses it for Small only (§3.1).</summary>
    public bool? UseDAdaptation { get; set; }

    /// <summary>Gets or sets AdamW's β₁ (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets AdamW's β₂ (0.95).</summary>
    public double Beta2 { get; set; } = 0.95;

    /// <summary>Gets or sets the decoupled weight decay (0.1).</summary>
    public double WeightDecay { get; set; } = 0.1;

    /// <summary>Gets or sets the gradient-norm clip (1.0).</summary>
    public double GradientClipNorm { get; set; } = 1.0;

    /// <summary>Gets or sets the learning-rate warmup steps (4000).</summary>
    public int WarmupSteps { get; set; } = 4000;

    /// <summary>Gets or sets the training steps the cosine schedule spans (1,000,000).</summary>
    public int TrainingSteps { get; set; } = 1_000_000;

    /// <summary>Gets or sets the EMA decay (0.99, §3.1).</summary>
    public double EmaDecay { get; set; } = 0.99;

    /// <summary>Gets or sets how many optimizer updates pass between EMA updates (10, the reference's).</summary>
    public int EmaEveryUpdates { get; set; } = 10;

    /// <summary>Gets or sets whether the model is conditioned on a melody as well as text (§2.3, the melody model):
    /// the chromagram of the conditioning audio, quantized to its dominant pitch class, and the text are given as a
    /// prefix of the sequence instead of through cross-attention (§2.4, audiocraft's chroma2music fuser).</summary>
    public bool MelodyConditioning { get; set; }

    /// <summary>Gets or sets the probability of training a step without the melody (0.5, audiocraft's
    /// <c>attribute_dropout.wav.self_wav</c>; the paper does not say).</summary>
    public double MelodyDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the chromagram window and FFT size (2¹⁴ = 16384, §3.1).</summary>
    public int ChromaWindow { get; set; } = 1 << 14;

    /// <summary>Gets or sets the chromagram hop (2¹² = 4096, §3.1).</summary>
    public int ChromaHop { get; set; } = 1 << 12;

    /// <summary>The paper's 32 kHz EnCodec (§3.1; audiocraft <c>encodec_large_nq4_s640</c>): stride 640 (ratios 8, 5,
    /// 4, 4), 64 initial filters doubling over the encoder, non-causal with constant padding, four quantizers of 2048
    /// codes.</summary>
    public static EnCodecOptions EnCodec32kHz() => new()
    {
        SampleRate = 32000,
        Filters = 64,
        Ratios = [8, 5, 4, 4],
        Dimension = 128,
        ResidualKernelSizes = [3, 1],
        TrueSkip = true,
        LstmLayers = 2,
        Causal = false,
        ReflectPadding = false,
        NumQuantizers = 4,
        CodebookSize = 2048,
        TargetBandwidths = [2.2],
        TargetBandwidthKbps = 2.2,
        IncludeDiscriminators = false,
    };
}

/// <summary>MusicGen's stereo codebook patterns (Copet et al. 2023, §4.3, Figure 2a).</summary>
public enum MusicGenStereoPattern
{
    /// <summary>The left and right codebooks of a level share its delay (predicted in parallel).</summary>
    PartialDelay,

    /// <summary>The right codebook of a level is one step behind the left.</summary>
    Delay,
}
