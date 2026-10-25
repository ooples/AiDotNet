namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Options for <see cref="KaniTTS{T}"/>.</summary>
/// <remarks>
/// <para>KaniTTS has no paper: the defaults are its release (nineninesix-ai/kani-tts and the kani-tts 0.0.4 package;
/// nineninesix/kani-tts-370m's <c>config.json</c>, which every Kani 1 checkpoint shares). The language model is LFM2-350M
/// (width 1024, 16 layers of which 6 attend, 16 heads over 8 key-value heads, a short-convolution kernel of 3, SwiGLU
/// 4608 after LFM2's adjustment of 6656, rotary base 10⁶, a head tied to its 80,539-token table: the text tokenizer's
/// 64,400, ten control tokens and four codebooks of 4032 audio codes). It speaks NVIDIA's NanoCodec
/// (<c>nemo-nano-codec-22khz-0.6kbps-12.5fps</c>: 22.05 kHz, 1764 samples a frame, four FSQ codebooks of 9·8·8·7 codes),
/// sampled at temperature 1, top-k 50, top-p 0.95 and a repetition penalty of 1.1 for up to 1200 tokens. Training follows
/// the release's fine-tuning pipeline (KaniTTS-Finetune-pipeline): rank-stabilized LoRA of rank 16 (alpha 16, dropout 0.1)
/// on q_proj, w1 and w2, AdamW at 5e-5 with weight decay 0.02, a cosine schedule after a 10% warm-up, the gradient norm
/// clipped to 1 and four samples accumulated per step.</para>
/// <para><b>For Beginners:</b> These settings size KaniTTS's language model and its audio codec, how it picks each next
/// sound token, and how it fine-tunes; the defaults reproduce the released models.</para>
/// </remarks>
public class KaniTTSOptions : TtsModelOptions
{
    /// <summary>Creates the released configuration.</summary>
    public KaniTTSOptions()
    {
        SampleRate = 22050;
        HopSize = 1764;
        HiddenDim = 1024;
        VocabSize = 80539;
        MaxTextLength = 1000;
        LearningRate = 5e-5;
        WeightDecay = 0.02;
        DropoutRate = 0.0;
    }

    /// <summary>Copies another instance.</summary>
    public KaniTTSOptions(KaniTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        AttentionLayers = (bool[])other.AttentionLayers.Clone();
        Heads = other.Heads;
        KeyValueHeads = other.KeyValueHeads;
        Intermediate = other.Intermediate;
        AutoAdjustFeedForward = other.AutoAdjustFeedForward;
        FeedForwardMultiplier = other.FeedForwardMultiplier;
        FeedForwardMultipleOf = other.FeedForwardMultipleOf;
        NormEpsilon = other.NormEpsilon;
        RopeTheta = other.RopeTheta;
        ConvKernel = other.ConvKernel;
        TextVocabulary = other.TextVocabulary;
        StartOfText = other.StartOfText;
        EndOfText = other.EndOfText;
        Codebooks = other.Codebooks;
        CodecEncoderChannels = other.CodecEncoderChannels;
        CodecDownsampleRates = (int[])other.CodecDownsampleRates.Clone();
        CodecEncodedDim = other.CodecEncodedDim;
        CodecDecoderChannels = other.CodecDecoderChannels;
        CodecUpsampleRates = (int[])other.CodecUpsampleRates.Clone();
        CodecLevels = (int[])other.CodecLevels.Clone();
        SpeakerNames = (string[])other.SpeakerNames.Clone();
        TokenizerJson = other.TokenizerJson;
        Temperature = other.Temperature;
        TopK = other.TopK;
        TopP = other.TopP;
        RepetitionPenalty = other.RepetitionPenalty;
        MaxNewTokens = other.MaxNewTokens;
        LenientSpeechSpans = other.LenientSpeechSpans;
        StreamingChunkFrames = other.StreamingChunkFrames;
        StreamingLookbackFrames = other.StreamingLookbackFrames;
        UseLora = other.UseLora;
        LoraRank = other.LoraRank;
        LoraAlpha = other.LoraAlpha;
        LoraDropout = other.LoraDropout;
        LoraTargets = (string[])other.LoraTargets.Clone();
        RankStabilizedLora = other.RankStabilizedLora;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        MaxGradNorm = other.MaxGradNorm;
        WarmupRatio = other.WarmupRatio;
        TrainingSteps = other.TrainingSteps;
        GradientAccumulationSteps = other.GradientAccumulationSteps;
        RemoveRepeatedFrames = other.RemoveRepeatedFrames;
        FrameLevelPositions = other.FrameLevelPositions;
        AudioStep = other.AudioStep;
        LearnableRope = other.LearnableRope;
        RopeAlphaMin = other.RopeAlphaMin;
        RopeAlphaMax = other.RopeAlphaMax;
        RopeAlphaLearningRateRatio = other.RopeAlphaLearningRateRatio;
        SpeakerEmbeddingDim = other.SpeakerEmbeddingDim;
        WarmupSteps = other.WarmupSteps;
        ReleaseSpeakerCachePosition = other.ReleaseSpeakerCachePosition;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Which of the language model's layers attend (the rest are short convolutions): LFM2-350M's 16 layers attend
    /// at 2, 5, 8, 10, 12 and 14.</summary>
    public bool[] AttentionLayers { get; set; } =
        { false, false, true, false, false, true, false, false, true, false, true, false, true, false, true, false };

    /// <summary>Attention heads (16).</summary>
    public int Heads { get; set; } = 16;

    /// <summary>Key-value heads (8).</summary>
    public int KeyValueHeads { get; set; } = 8;

    /// <summary>The feed-forward width as configured (6656; LFM2 builds 4608 from it).</summary>
    public int Intermediate { get; set; } = 6656;

    /// <summary>LFM2's feed-forward adjustment: 2/3 of <see cref="Intermediate"/>, times
    /// <see cref="FeedForwardMultiplier"/>, rounded up to a multiple of <see cref="FeedForwardMultipleOf"/> (on).</summary>
    public bool AutoAdjustFeedForward { get; set; } = true;

    /// <summary><c>block_ffn_dim_multiplier</c> (1).</summary>
    public double FeedForwardMultiplier { get; set; } = 1.0;

    /// <summary><c>block_multiple_of</c> (256).</summary>
    public int FeedForwardMultipleOf { get; set; } = 256;

    /// <summary>RMSNorm ε (1e-5).</summary>
    public double NormEpsilon { get; set; } = 1e-5;

    /// <summary>Rotary base (10⁶).</summary>
    public double RopeTheta { get; set; } = 1000000;

    /// <summary>The short convolution's kernel (3).</summary>
    public int ConvKernel { get; set; } = 3;

    /// <summary>The text tokenizer's size (64,400); the control tokens and the audio codes follow it.</summary>
    public int TextVocabulary { get; set; } = 64400;

    /// <summary>The text tokenizer's start token (1, <c>&lt;|startoftext|&gt;</c>).</summary>
    public int StartOfText { get; set; } = 1;

    /// <summary>The text tokenizer's end token (2, <c>&lt;|endoftext|&gt;</c>).</summary>
    public int EndOfText { get; set; } = 2;

    /// <summary>NanoCodec's codebooks (4).</summary>
    public int Codebooks { get; set; } = 4;

    /// <summary>The codec encoder's first width (24).</summary>
    public int CodecEncoderChannels { get; set; } = 24;

    /// <summary>The codec encoder's downsampling (2, 3, 6, 7, 7: 1764 samples a frame).</summary>
    public int[] CodecDownsampleRates { get; set; } = { 2, 3, 6, 7, 7 };

    /// <summary>The codec's encoded width (16: four groups of four FSQ dimensions).</summary>
    public int CodecEncodedDim { get; set; } = 16;

    /// <summary>The codec decoder's first width (864).</summary>
    public int CodecDecoderChannels { get; set; } = 864;

    /// <summary>The codec decoder's upsampling (7, 7, 6, 3, 2).</summary>
    public int[] CodecUpsampleRates { get; set; } = { 7, 7, 6, 3, 2 };

    /// <summary>The levels of each FSQ dimension of a codebook (9, 8, 8, 7: 4032 codes).</summary>
    public int[] CodecLevels { get; set; } = { 9, 8, 8, 7 };

    /// <summary>A multi-speaker checkpoint's speakers (<c>config.json</c>'s <c>speaker_settings.speaker_list</c>, which
    /// <c>LoadReferenceWeights</c> reads), indexed by a voice's or a training sample's <c>SpeakerId</c>.</summary>
    public string[] SpeakerNames { get; set; } = Array.Empty<string>();

    /// <summary>The text tokenizer's <c>tokenizer.json</c>, which <c>LoadReferenceWeights</c> or <c>LoadTokenizer</c>
    /// fills (LFM2's tokenizer comes under its own license, so none is built in); null until then.</summary>
    public string? TokenizerJson { get; set; }

    /// <summary>Sampling temperature (1; 0 samples greedily).</summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>Top-k (50, Hugging Face generate()'s default, which the release keeps).</summary>
    public int TopK { get; set; } = 50;

    /// <summary>Top-p (0.95).</summary>
    public double TopP { get; set; } = 0.95;

    /// <summary>Repetition penalty over every token read so far (1.1).</summary>
    public double RepetitionPenalty { get; set; } = 1.1;

    /// <summary>Tokens generated at most (1200).</summary>
    public int MaxNewTokens { get; set; } = 1200;

    /// <summary>
    /// How generated tokens become audio. Off (the kani-tts package): the codes between the first start-of-speech and
    /// end-of-speech tokens, which must both be present and hold whole frames of valid codes, else synthesis throws. On
    /// (the release's streaming example): the tokens after start-of-speech up to end-of-speech or the end, decoded in
    /// windows of <see cref="StreamingChunkFrames"/> new frames with <see cref="StreamingLookbackFrames"/> of context,
    /// skipping windows with invalid codes; empty audio when none decode.
    /// </summary>
    public bool LenientSpeechSpans { get; set; }

    /// <summary>New frames per streaming window (25).</summary>
    public int StreamingChunkFrames { get; set; } = 25;

    /// <summary>Earlier frames each streaming window re-reads for context (15).</summary>
    public int StreamingLookbackFrames { get; set; } = 15;

    /// <summary>Whether training adapts LoRA weights (the release's fine-tuning) rather than every language-model weight.</summary>
    public bool UseLora { get; set; } = true;

    /// <summary>LoRA rank (16).</summary>
    public int LoraRank { get; set; } = 16;

    /// <summary>LoRA alpha (16).</summary>
    public double LoraAlpha { get; set; } = 16;

    /// <summary>LoRA dropout (0.1).</summary>
    public double LoraDropout { get; set; } = 0.1;

    /// <summary>The modules LoRA adapts (q_proj, w1, w2), by peft's names: q_proj, k_proj, v_proj, out_proj, in_proj, w1,
    /// w2, w3.</summary>
    public string[] LoraTargets { get; set; } = { "q_proj", "w1", "w2" };

    /// <summary>peft's rsLoRA: updates scaled by alpha / √rank rather than alpha / rank (on).</summary>
    public bool RankStabilizedLora { get; set; } = true;

    /// <summary>AdamW β₁ (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>AdamW β₂ (0.999).</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>AdamW ε (1e-8).</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>The gradient norm clipped to (1).</summary>
    public double MaxGradNorm { get; set; } = 1.0;

    /// <summary>The share of <see cref="TrainingSteps"/> the learning rate warms up over linearly (0.1).</summary>
    public double WarmupRatio { get; set; } = 0.1;

    /// <summary>Optimizer steps the cosine schedule spans (0: a constant rate). Set it to your run's length.</summary>
    public int TrainingSteps { get; set; }

    /// <summary>Samples whose gradients make one optimizer step (4).</summary>
    public int GradientAccumulationSteps { get; set; } = 4;

    /// <summary>Whether a training recording's consecutive identical frames are kept once (the release's
    /// <c>remove_consecutive_duplicates</c>, on).</summary>
    public bool RemoveRepeatedFrames { get; set; } = true;

    /// <summary>KaniTTS 2's frame-level positions: the four tokens of an audio frame share one rotary position, each
    /// frame <see cref="AudioStep"/> after the last (off for Kani 1).</summary>
    public bool FrameLevelPositions { get; set; }

    /// <summary>How far each audio frame's position advances (<c>audio_step</c>, 1).</summary>
    public double AudioStep { get; set; } = 1.0;

    /// <summary>KaniTTS 2's learnable rotary scale per attention layer (off for Kani 1).</summary>
    public bool LearnableRope { get; set; }

    /// <summary>The learnable rotary scale's least value (<c>alpha_min</c>, 0.1).</summary>
    public double RopeAlphaMin { get; set; } = 0.1;

    /// <summary>The learnable rotary scale's greatest value (<c>alpha_max</c>, 2).</summary>
    public double RopeAlphaMax { get; set; } = 2.0;

    /// <summary>The learnable rotary scales' learning rate as a multiple of <see cref="ModelOptions"/>'s
    /// (<c>alpha_lr_ratio</c>, 1.5), with no weight decay.</summary>
    public double RopeAlphaLearningRateRatio { get; set; } = 1.5;

    /// <summary>KaniTTS 2's speaker embedding width (128), projected and inserted after the first token; 0 for none
    /// (Kani 1).</summary>
    public int SpeakerEmbeddingDim { get; set; }

    /// <summary>
    /// Whether generation with a speaker embedding keeps the release's cache position (on): KaniTTS2ForCausalLM inserts
    /// the speaker after the first token, but transformers' generate() keeps counting cache positions without it, so from
    /// the second step on the causal mask it builds hides each new token from its own key. Off attends as the training
    /// forward does.
    /// </summary>
    public bool ReleaseSpeakerCachePosition { get; set; } = true;

    /// <summary>Linear warm-up steps (KaniTTS 2's 2000); 0 to warm up over <see cref="WarmupRatio"/> of the steps.</summary>
    public int WarmupSteps { get; set; }

    /// <summary>Seed of sampling and of LoRA's dropout.</summary>
    public int SamplingSeed { get; set; }
}
