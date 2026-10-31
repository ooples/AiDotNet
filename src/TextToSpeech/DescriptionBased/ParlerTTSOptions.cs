using AiDotNet.Audio.Effects;

namespace AiDotNet.TextToSpeech.DescriptionBased;

/// <summary>Options for Parler-TTS (Lyth &amp; King 2024, "Natural language guidance of high-fidelity text-to-speech with
/// synthetic annotations").</summary>
/// <remarks>
/// <para>
/// The paper fixes the design (§3): the transcript prepended to the codes of DAC at 44.1 kHz (nine codebooks, 86 frames
/// a second) in MusicGen's delay pattern, and the description read by cross-attention from a frozen Flan-T5 encoder. It
/// leaves sizes and training settings open; the defaults take them from the open reproduction the paper's code release
/// points to (huggingface/parler-tts), as its released parler-tts-mini-v1 configures them: a 24-layer, 1024-wide decoder
/// with 16 heads and feed-forward 4096, sinusoidal positions, dropout 0.1, Flan-T5-large as the description encoder,
/// and training with AdamW (9.5e-4, betas 0.9 and 0.99, weight decay 0.01, gradients clipped to norm 1) held constant
/// after 20,000 warm-up updates, the encoder frozen. Generation samples (temperature 1, top 50) for at most 2580 steps
/// and at least 10.
/// </para>
/// <para><b>For Beginners:</b> These options set the sizes of Parler-TTS's networks, its codec and how it is trained and
/// sampled. The defaults reproduce the released model.</para>
/// </remarks>
public class ParlerTTSOptions : TtsModelOptions
{
    /// <summary>Creates the released parler-tts-mini-v1 configuration.</summary>
    public ParlerTTSOptions()
    {
        SampleRate = 44100;
        HopSize = 512;
        HiddenDim = 1024;
        NumHeads = 16;
        NumDecoderLayers = 24;
        DropoutRate = 0.1;
        MaxTextLength = 512;
        LearningRate = 9.5e-4;
        WeightDecay = 0.01;
        Codec = new DACOptions();
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public ParlerTTSOptions(ParlerTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        FeedForwardDim = other.FeedForwardDim;
        KeyValueHeads = other.KeyValueHeads;
        CrossAttentionKeyValueHeads = other.CrossAttentionKeyValueHeads;
        AttentionDropout = other.AttentionDropout;
        ActivationDropout = other.ActivationDropout;
        LayerDrop = other.LayerDrop;
        RotaryEmbeddings = other.RotaryEmbeddings;
        RopeTheta = other.RopeTheta;
        PromptCrossAttention = other.PromptCrossAttention;
        NumCodebooks = other.NumCodebooks;
        CodebookSize = other.CodebookSize;
        AudioVocabulary = other.AudioVocabulary;
        PromptVocabulary = other.PromptVocabulary;
        CodebookWeights = other.CodebookWeights is null ? null : (double[])other.CodebookWeights.Clone();
        TextEncoderDim = other.TextEncoderDim;
        TextEncoderFeedForwardDim = other.TextEncoderFeedForwardDim;
        TextEncoderKeyValueDim = other.TextEncoderKeyValueDim;
        TextEncoderHeads = other.TextEncoderHeads;
        TextEncoderLayers = other.TextEncoderLayers;
        TextEncoderVocabulary = other.TextEncoderVocabulary;
        TextEncoderDropout = other.TextEncoderDropout;
        FreezeTextEncoder = other.FreezeTextEncoder;
        InitializerFactor = other.InitializerFactor;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        MaxGradientNorm = other.MaxGradientNorm;
        WarmupSteps = other.WarmupSteps;
        DoSample = other.DoSample;
        Temperature = other.Temperature;
        TopK = other.TopK;
        MinNewTokens = other.MinNewTokens;
        MaxLength = other.MaxLength;
        Codec = (DACOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(other.Codec);
        SamplingSeed = other.SamplingSeed;
    }

    // ---------------------------------------------------------------- decoder (width HiddenDim, heads NumHeads, layers NumDecoderLayers)

    /// <summary>Feed-forward width of the decoder (4096).</summary>
    public int FeedForwardDim { get; set; } = 4096;

    /// <summary>Key/value heads of the decoder's self-attention (0: one per head, as released; fewer is grouped-query
    /// attention).</summary>
    public int KeyValueHeads { get; set; }

    /// <summary>Key/value heads of the decoder's cross-attention (0: as many as <see cref="KeyValueHeads"/>).</summary>
    public int CrossAttentionKeyValueHeads { get; set; }

    /// <summary>Dropout on the attention weights (0).</summary>
    public double AttentionDropout { get; set; }

    /// <summary>Dropout after the feed-forward activation (0).</summary>
    public double ActivationDropout { get; set; }

    /// <summary>Probability of skipping a decoder layer in training (0).</summary>
    public double LayerDrop { get; set; }

    /// <summary>Rotary position embeddings in place of the added sinusoidal ones (off in the released model; the
    /// reproduction's <c>rope_embeddings</c>).</summary>
    public bool RotaryEmbeddings { get; set; }

    /// <summary>The rotary embeddings' base (10,000).</summary>
    public double RopeTheta { get; set; } = 10_000.0;

    /// <summary>Whether the transcript is cross-attended next to the description instead of prepended to the codes (off,
    /// as the paper prepends it; the reproduction's <c>prompt_cross_attention</c>).</summary>
    public bool PromptCrossAttention { get; set; }

    /// <summary>Codebooks (9: DAC at 8 kbps).</summary>
    public int NumCodebooks { get; set; } = 9;

    /// <summary>Codes per codebook (1024, the codec's).</summary>
    [AiDotNet.Attributes.NotScaled("the codec's codes per codebook, which it must equal")]
    public int CodebookSize { get; set; } = 1024;

    /// <summary>Logits per codebook (1088): the codes, the end token (the codebook size) and the start token (one more),
    /// rounded up as released.</summary>
    public int AudioVocabulary { get; set; } = 1088;

    /// <summary>Rows of the transcript embedding (32,128: Flan-T5's vocabulary, which tokenizes the transcript).</summary>
    [AiDotNet.Attributes.NotScaled("holds every id of the T5 tokenizer that tokenizes the transcript")]
    public int PromptVocabulary { get; set; } = 32_128;

    /// <summary>Per-codebook loss weights (none: the mean over the codebooks).</summary>
    public double[]? CodebookWeights { get; set; }

    // ---------------------------------------------------------------- description encoder (Flan-T5-large)

    /// <summary>Width of the description encoder (1024); a width other than the decoder's adds a projection.</summary>
    public int TextEncoderDim { get; set; } = 1024;

    /// <summary>Feed-forward width of the description encoder (2816, gated GELU).</summary>
    public int TextEncoderFeedForwardDim { get; set; } = 2816;

    /// <summary>Width of each attention head of the description encoder (64).</summary>
    public int TextEncoderKeyValueDim { get; set; } = 64;

    /// <summary>Attention heads of the description encoder (16).</summary>
    public int TextEncoderHeads { get; set; } = 16;

    /// <summary>Layers of the description encoder (24).</summary>
    public int TextEncoderLayers { get; set; } = 24;

    /// <summary>Vocabulary of the description encoder (32,128).</summary>
    [AiDotNet.Attributes.NotScaled("holds every id of the T5 tokenizer that tokenizes the description")]
    public int TextEncoderVocabulary { get; set; } = 32_128;

    /// <summary>Dropout inside the description encoder (0.1, Flan-T5's; active in training even when frozen, as the
    /// reproduction leaves the frozen encoder in training mode).</summary>
    public double TextEncoderDropout { get; set; } = 0.1;

    /// <summary>Whether the description encoder is frozen in training (on, §3).</summary>
    public bool FreezeTextEncoder { get; set; } = true;

    // ---------------------------------------------------------------- training

    /// <summary>Standard deviation of the decoder's initial weights (0.02, <c>initializer_factor</c>).</summary>
    public double InitializerFactor { get; set; } = 0.02;

    /// <summary>AdamW β₁ (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>AdamW β₂ (0.99).</summary>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>AdamW ε (1e-8).</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>Gradient-norm clipping (1; 0 turns it off).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>Linear warm-up of the learning rate, in updates (20,000), after which it stays constant.</summary>
    public int WarmupSteps { get; set; } = 20_000;

    // ---------------------------------------------------------------- generation

    /// <summary>Whether generation samples (on) or takes the most likely code.</summary>
    public bool DoSample { get; set; } = true;

    /// <summary>Sampling temperature (1).</summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>Sampling top-k cut (50; 0 keeps every code).</summary>
    public int TopK { get; set; } = 50;

    /// <summary>Steps before the end token is allowed (10).</summary>
    public int MinNewTokens { get; set; } = 10;

    /// <summary>Most decoder steps, the start column included (2580: 30 seconds of codes and the delay).</summary>
    public int MaxLength { get; set; } = 2580;

    /// <summary>The codec (DAC 44.1 kHz at 8 kbps: nine codebooks).</summary>
    public DACOptions Codec { get; set; }

    /// <summary>Seed of the training draws (dropout) and of sampling.</summary>
    public int SamplingSeed { get; set; }
}
