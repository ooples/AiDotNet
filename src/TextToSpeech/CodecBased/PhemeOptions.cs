using AiDotNet.Audio.Generation;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for Pheme (Budzianowski et al. 2024, "Pheme: Efficient and Conversational Speech Generation").</summary>
/// <remarks>
/// <para>
/// Pheme is a two-stage model over SpeechTokenizer codes: a T5 encoder–decoder turns phonemes into semantic tokens (the
/// first SpeechTokenizer codebook), and a SoundStorm-style Conformer fills in the seven acoustic codebooks
/// non-autoregressively, conditioned on the semantic tokens, a prompt and a speaker embedding.
/// </para>
/// <para>
/// The defaults are the paper's small model (Table 5: a 6 + 6-layer T5 of width 512, a 3-layer acoustic Conformer of
/// width 1024) and its training recipe (§4.1: both networks trained with AdamW, β = 0.9 / 0.98, at 5e-4 with 10,000
/// warm-up steps and linear decay to zero over 800,000 steps; the paper also writes "decayed from 2×10⁻⁴", which
/// contradicts its stated rate, so the stated 5e-4 is the peak). <see cref="Large"/> is the paper's large model. The
/// released checkpoints differ from the paper's text: <see cref="OfficialSmallCheckpoint"/> (PolyAI/pheme_small) has a
/// 768-wide acoustic model, and <see cref="OfficialLargeCheckpoint"/> (PolyAI/pheme) no speaker embedding.
/// </para>
/// <para>
/// What the paper leaves to its code follows the reference implementation (PolyAI-LDN/pheme): the text-to-semantic
/// model is Hugging Face's <c>T5ForConditionalGeneration</c> v1.0 (d_ff 2048, d_kv 64, 8 heads, ReLU, dropout 0.1,
/// 32 relative buckets up to distance 128), with the Trainer's defaults where the paper is silent (no weight decay,
/// gradient norm clipped to 1); the acoustic model is lucidrains' SoundStorm Conformer (8 heads of 64, feed-forward ×4,
/// convolution expansion 2, kernel 5, dropout 0.1) with a 512 → width projection of the L2-normalized pyannote speaker
/// embedding (dropout 0.05) and PyTorch's default AdamW weight decay 0.01; codes are SpeechTokenizer's (1024 semantic
/// and 1024 acoustic codes, 7 acoustic codebooks, 16 kHz, 50 frames per second). Synthesis samples semantic tokens at
/// temperature 0.7 from the top 210 (§4.1), at most 750 new tokens, resampled while any token repeats more than 100
/// times in a row; the acoustic codebooks are decoded as <see cref="AcousticDecoding"/> says.
/// </para>
/// <para><b>For Beginners:</b> These options set the sizes and training settings of Pheme's two networks. The defaults
/// reproduce the released small model.</para>
/// </remarks>
public class PhemeOptions : TtsModelOptions
{
    /// <summary>Creates the paper's small model's options (Budzianowski et al. 2024, Table 5 and §4.1).</summary>
    public PhemeOptions()
    {
        SampleRate = 16000;
        HopSize = 320;
        HiddenDim = 1024;
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
        LearningRate = 5e-4;
        WeightDecay = 0.01;
        MaxTextLength = 512;
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public PhemeOptions(PhemeOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        TextModelDim = other.TextModelDim;
        TextFeedForwardDim = other.TextFeedForwardDim;
        TextKeyValueDim = other.TextKeyValueDim;
        AcousticLayers = other.AcousticLayers;
        AcousticHeads = other.AcousticHeads;
        AcousticHeadDim = other.AcousticHeadDim;
        AcousticFeedForwardMultiplier = other.AcousticFeedForwardMultiplier;
        AcousticConvExpansion = other.AcousticConvExpansion;
        AcousticConvKernel = other.AcousticConvKernel;
        AcousticCodebooks = other.AcousticCodebooks;
        CodebookSize = other.CodebookSize;
        SemanticCodes = other.SemanticCodes;
        UseSpeakerEmbedding = other.UseSpeakerEmbedding;
        SpeakerEmbeddingDropout = other.SpeakerEmbeddingDropout;
        TextWarmupSteps = other.TextWarmupSteps;
        TextTrainingSteps = other.TextTrainingSteps;
        AcousticWarmupSteps = other.AcousticWarmupSteps;
        AcousticTrainingSteps = other.AcousticTrainingSteps;
        Temperature = other.Temperature;
        TopK = other.TopK;
        MaxNewSemanticTokens = other.MaxNewSemanticTokens;
        MaxConsecutiveRepeats = other.MaxConsecutiveRepeats;
        MaskGitSteps = other.MaskGitSteps;
        AcousticDecoding = other.AcousticDecoding;
        SpeechTokenizer = (SpeechTokenizerOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(other.SpeechTokenizer);
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>The paper's large model (Table 5): 14 + 14 T5 layers and an 8-layer acoustic Conformer.</summary>
    public static PhemeOptions Large() => new()
    {
        NumEncoderLayers = 14,
        NumDecoderLayers = 14,
        AcousticLayers = 8,
    };

    /// <summary>The released small checkpoint (PolyAI/pheme_small, <c>config_s2a.json</c>): the small model with a
    /// 768-wide acoustic Conformer.</summary>
    public static PhemeOptions OfficialSmallCheckpoint() => new() { HiddenDim = 768 };

    /// <summary>The released large checkpoint (PolyAI/pheme, <c>config_s2a.json</c>): the large model trained without
    /// the speaker embedding (<c>use_spkr_emb: false</c>).</summary>
    public static PhemeOptions OfficialLargeCheckpoint()
    {
        var options = Large();
        options.UseSpeakerEmbedding = false;
        return options;
    }

    /// <summary>T5 model width d_model (512).</summary>
    public int TextModelDim { get; set; } = 512;

    /// <summary>T5 feed-forward width d_ff (2048).</summary>
    public int TextFeedForwardDim { get; set; } = 2048;

    /// <summary>T5 per-head width d_kv (64).</summary>
    public int TextKeyValueDim { get; set; } = 64;

    /// <summary>Acoustic Conformer layers (3; 8 in the large model). Its width is <see cref="TtsModelOptions.HiddenDim"/>.</summary>
    public int AcousticLayers { get; set; } = 3;

    /// <summary>Acoustic Conformer attention heads (8).</summary>
    public int AcousticHeads { get; set; } = 8;

    /// <summary>Acoustic Conformer per-head width (64).</summary>
    public int AcousticHeadDim { get; set; } = 64;

    /// <summary>Acoustic Conformer feed-forward multiplier (4).</summary>
    public int AcousticFeedForwardMultiplier { get; set; } = 4;

    /// <summary>Acoustic Conformer convolution expansion factor (2).</summary>
    public int AcousticConvExpansion { get; set; } = 2;

    /// <summary>Acoustic Conformer depthwise convolution kernel (5).</summary>
    public int AcousticConvKernel { get; set; } = 5;

    /// <summary>Acoustic codebooks after the semantic one (7).</summary>
    public int AcousticCodebooks { get; set; } = 7;

    /// <summary>Codes per acoustic codebook (1024).</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>Semantic codes (1024).</summary>
    public int SemanticCodes { get; set; } = 1024;

    /// <summary>Whether the acoustic model reads a pyannote speaker embedding (true; false in the released large
    /// checkpoint).</summary>
    public bool UseSpeakerEmbedding { get; set; } = true;

    /// <summary>Dropout on the speaker embedding before its projection (0.05).</summary>
    public double SpeakerEmbeddingDropout { get; set; } = 0.05;

    /// <summary>Linear warm-up of the text-to-semantic learning rate, in updates (10,000, §4.1).</summary>
    public int TextWarmupSteps { get; set; } = 10_000;

    /// <summary>Updates after which the text-to-semantic learning rate reaches zero (800,000, §4.1).</summary>
    public int TextTrainingSteps { get; set; } = 800_000;

    /// <summary>Linear warm-up of the acoustic learning rate, in updates (10,000, §4.1).</summary>
    public int AcousticWarmupSteps { get; set; } = 10_000;

    /// <summary>Updates after which the acoustic learning rate reaches zero (800,000, §4.1).</summary>
    public int AcousticTrainingSteps { get; set; } = 800_000;

    /// <summary>Semantic sampling temperature (0.7).</summary>
    public double Temperature { get; set; } = 0.7;

    /// <summary>Semantic top-k sampling (210).</summary>
    public int TopK { get; set; } = 210;

    /// <summary>Most semantic tokens generated after the prompt (750).</summary>
    public int MaxNewSemanticTokens { get; set; } = 750;

    /// <summary>Semantic tokens are resampled while one token repeats more than this many times in a row (100).</summary>
    public int MaxConsecutiveRepeats { get; set; } = 100;

    /// <summary>MaskGIT (confidence-based) steps for each codebook <see cref="AcousticDecoding"/> decodes that way (16).</summary>
    public int MaskGitSteps { get; set; } = 16;

    /// <summary>
    /// Which acoustic codebooks are decoded with MaskGIT and which greedily. The default is the reference code's order,
    /// which produced the released models' outputs; the paper's text (§4.1) describes the reverse.
    /// </summary>
    public PhemeAcousticDecoding AcousticDecoding { get; set; } = PhemeAcousticDecoding.MaskGitFirstLevel;

    /// <summary>The SpeechTokenizer whose codes Pheme models (its released configuration by default).</summary>
    public SpeechTokenizerOptions SpeechTokenizer { get; set; } = SpeechTokenizerOptions.OfficialCheckpoint();

    /// <summary>Seed of the training draws (prompt split, mask ratio, codebook level) and of sampling.</summary>
    public int SamplingSeed { get; set; }
}

/// <summary>How Pheme's acoustic model decodes its codebooks at synthesis.</summary>
public enum PhemeAcousticDecoding
{
    /// <summary>
    /// The reference code's order (<c>TTSConformer.inference</c>): MaskGIT steps on the first acoustic codebook, then
    /// one greedy step on each of the others.
    /// </summary>
    MaskGitFirstLevel,

    /// <summary>
    /// The paper's text (§4.1: "greedy sampling in the first level of acoustic tokens (q2) and ... 16 steps of
    /// confidence-based sampling for all the remaining levels").
    /// </summary>
    MaskGitRemainingLevels,
}
