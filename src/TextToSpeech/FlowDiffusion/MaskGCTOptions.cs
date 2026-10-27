namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Which of MaskGCT's networks <see cref="MaskGCT{T}"/> trains.</summary>
public enum MaskGctStage
{
    /// <summary>The text-to-semantic model (§3.2.2).</summary>
    TextToSemantic,

    /// <summary>The semantic-to-acoustic model that predicts the first acoustic layer (the release's <c>s2a_1layer</c>).</summary>
    FirstAcousticLayer,

    /// <summary>The semantic-to-acoustic model that predicts layers 2–12 given the first (the release's <c>s2a_full</c>).</summary>
    AcousticLayers,

    /// <summary>The total-duration predictor (App. A.5); it trains on phone durations.</summary>
    Duration,
}

/// <summary>How <see cref="MaskGCT{T}"/> chooses the target's length.</summary>
public enum MaskGctDurationMode
{
    /// <summary>The release's rule: the prompt's speaking rate (its 16 kHz samples per phone) times the target's phones,
    /// at 50 semantic frames per second.</summary>
    Rule,

    /// <summary>The paper's flow-matching duration predictor (App. A.5), which must be trained
    /// (<see cref="MaskGctStage.Duration"/>).</summary>
    Predictor,
}

/// <summary>Options for <see cref="MaskGCT{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are the released MaskGCT (Wang et al. 2024; <c>amphion/MaskGCT</c>, Amphion
/// <c>models/tts/maskgct/config/maskgct.json</c>): semantic tokens from w2v-BERT 2.0's 17th hidden layer, normalized and
/// quantized by a RepCodec of 8,192 codes; acoustic tokens from a 24 kHz codec of twelve 1,024-code layers at 50 Hz; a
/// text-to-semantic transformer of width 1536 and a pair of semantic-to-acoustic transformers of width 1024, all 16
/// layers with 16 heads. Decoding follows the paper where it is explicit (§4.1: 50 text-to-semantic steps,
/// semantic-to-acoustic steps 40, 16 and then 1 per layer, guidance 2.5 rescaled by 0.75) and the release otherwise
/// (temperatures 0.9 and 1.5, top-k keeping the best 2%).</para>
/// <para><b>For Beginners:</b> MaskGCT clones a voice from a short recording and its transcript: these settings size its
/// networks and control how many refinement passes it makes; the defaults reproduce the released model.</para>
/// </remarks>
public class MaskGCTOptions : TtsModelOptions
{
    /// <summary>Creates the released configuration.</summary>
    public MaskGCTOptions()
    {
        SampleRate = 24000;
        HopSize = 480;
        HiddenDim = 1024;
        VocabSize = 1024;
        LearningRate = 1e-4;
        DropoutRate = 0.0;
    }

    // ---------------------------------------------------------------- text

    /// <summary>The language of the text to synthesize: "en", "zh", "ja", "ko", "fr" or "de".</summary>
    public string Language { get; set; } = "en";

    /// <summary>
    /// The release's Chinese polyphone model (<c>g2p/sources/g2p_chinese_model/poly_bert_model.onnx</c>), which Mandarin
    /// text needs.
    /// </summary>
    public string? PolyphoneModelPath { get; set; }

    /// <summary>
    /// The Open JTalk dictionary directory (<c>open_jtalk_dic_utf_8-1.11</c>, as pyopenjtalk downloads it), which Japanese
    /// text needs.
    /// </summary>
    public string? OpenJTalkDictionaryPath { get; set; }

    // ---------------------------------------------------------------- semantic tokens

    /// <summary>The rate the semantic model reads (16 kHz).</summary>
    public int SemanticSampleRate { get; set; } = 16000;

    /// <summary>w2v-BERT 2.0's width.</summary>
    public int SemanticHiddenSize { get; set; } = 1024;

    /// <summary>w2v-BERT 2.0's attention heads.</summary>
    public int SemanticHeads { get; set; } = 16;

    /// <summary>w2v-BERT 2.0's feed-forward width.</summary>
    public int SemanticIntermediateSize { get; set; } = 4096;

    /// <summary>The w2v-BERT 2.0 hidden state the semantic codec reads (17; only that many layers are built).</summary>
    public int SemanticLayer { get; set; } = 17;

    /// <summary>The semantic codebook's size (8,192).</summary>
    public int SemanticCodebookSize { get; set; } = 8192;

    /// <summary>The semantic codebook's dimension (8).</summary>
    public int SemanticCodebookDim { get; set; } = 8;

    /// <summary>The semantic codec's Vocos width (384).</summary>
    public int SemanticCodecDim { get; set; } = 384;

    /// <summary>The semantic codec's Vocos feed-forward width (2048).</summary>
    public int SemanticCodecIntermediate { get; set; } = 2048;

    /// <summary>The semantic codec's ConvNeXt blocks (12).</summary>
    public int SemanticCodecLayers { get; set; } = 12;

    // ---------------------------------------------------------------- acoustic tokens

    /// <summary>The acoustic codec's encoder width (96).</summary>
    public int CodecEncoderDim { get; set; } = 96;

    /// <summary>The acoustic codec's encoder strides (3, 4, 5, 8: 480 samples a frame).</summary>
    public int[] CodecEncoderRates { get; set; } = { 3, 4, 5, 8 };

    /// <summary>The acoustic codec's latent width (256).</summary>
    public int CodecLatentDim { get; set; } = 256;

    /// <summary>The acoustic layers (12).</summary>
    public int NumQuantizers { get; set; } = 12;

    /// <summary>Each acoustic layer's codes (1,024).</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>The acoustic codebooks' dimension (8).</summary>
    public int CodebookDim { get; set; } = 8;

    /// <summary>The acoustic decoder's Vocos width (512).</summary>
    public int VocosDim { get; set; } = 512;

    /// <summary>The acoustic decoder's Vocos feed-forward width (4096).</summary>
    public int VocosIntermediate { get; set; } = 4096;

    /// <summary>The acoustic decoder's ConvNeXt blocks (30).</summary>
    public int VocosLayers { get; set; } = 30;

    /// <summary>The acoustic decoder's ISTFT size (1920).</summary>
    public int VocosFftSize { get; set; } = 1920;

    // ---------------------------------------------------------------- the generators

    /// <summary>The text-to-semantic model's width (1536).</summary>
    public int TextToSemanticDim { get; set; } = 1536;

    /// <summary>The text-to-semantic model's layers (16).</summary>
    public int TextToSemanticLayers { get; set; } = 16;

    /// <summary>The text-to-semantic model's attention heads (16).</summary>
    public int TextToSemanticHeads { get; set; } = 16;

    /// <summary>The semantic-to-acoustic models' layers (16; their width is <see cref="TtsModelOptions.HiddenDim"/>).</summary>
    public int SemanticToAcousticLayers { get; set; } = 16;

    /// <summary>The semantic-to-acoustic models' attention heads (16).</summary>
    public int SemanticToAcousticHeads { get; set; } = 16;

    /// <summary>The generators' gated activation: SiLU, as the released models were trained, or GELU, as the paper writes.</summary>
    public MaskGctActivation Activation { get; set; } = MaskGctActivation.SiLU;

    // ---------------------------------------------------------------- decoding

    /// <summary>Text-to-semantic decoding steps (50, §4.1; the release's pipeline defaults to 25).</summary>
    public int TextToSemanticSteps { get; set; } = 50;

    /// <summary>Semantic-to-acoustic decoding steps per layer (40, 16, then 1, §4.1; the release's pipeline uses 25, 10,
    /// then 1).</summary>
    public int[] SemanticToAcousticSteps { get; set; } = { 40, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    /// <summary>The text-to-semantic sampling temperature (0.9, the release's).</summary>
    public double TextToSemanticTemperature { get; set; } = 0.9;

    /// <summary>The semantic-to-acoustic sampling temperature (1.5, the release's).</summary>
    public double SemanticToAcousticTemperature { get; set; } = 1.5;

    /// <summary>The top-k filter: logits below the best (1 − this) of the vocabulary are dropped (0.98).</summary>
    public double FilterThreshold { get; set; } = 0.98;

    /// <summary>The classifier-free guidance scale (2.5; 0 turns guidance off).</summary>
    public double Guidance { get; set; } = 2.5;

    /// <summary>The guidance rescale (0.75).</summary>
    public double GuidanceRescale { get; set; } = 0.75;

    /// <summary>The seed of decoding's random draws.</summary>
    public int SamplingSeed { get; set; } = 1234;

    // ---------------------------------------------------------------- duration

    /// <summary>How the target's length is chosen (the release's rule by default).</summary>
    public MaskGctDurationMode DurationMode { get; set; } = MaskGctDurationMode.Rule;

    /// <summary>
    /// The target's duration in seconds, overriding <see cref="DurationMode"/> (the release's <c>target_len</c>); null to
    /// choose it.
    /// </summary>
    public double? TargetDurationSeconds { get; set; }

    /// <summary>A factor on the chosen duration (§4.2.3's total-duration multiplier; 1 keeps it).</summary>
    public double DurationMultiplier { get; set; } = 1.0;

    /// <summary>The duration predictor's width (768, App. A.5).</summary>
    public int DurationDim { get; set; } = 768;

    /// <summary>The duration predictor's layers (12).</summary>
    public int DurationLayers { get; set; } = 12;

    /// <summary>The duration predictor's attention heads (12).</summary>
    public int DurationHeads { get; set; } = 12;

    /// <summary>The midpoint ODE solver's steps (4).</summary>
    public int DurationSteps { get; set; } = 4;

    /// <summary>The duration predictor's guidance toward the prompted velocity (1: none; the paper uses guidance but gives
    /// no scale).</summary>
    public double DurationGuidance { get; set; } = 1.0;

    // ---------------------------------------------------------------- training

    /// <summary>The network training updates.</summary>
    public MaskGctStage CurrentStage { get; set; } = MaskGctStage.TextToSemantic;

    /// <summary>The probability a training example has no prompt (0.15, the release's <c>cfg_scale</c> and App. A.5).</summary>
    public double PromptDropProbability { get; set; } = 0.15;

    /// <summary>
    /// The learning rate's warmup steps (32,000, §4.1), after which it decays with the inverse square root of the step.
    /// </summary>
    /// <remarks>A run far shorter than the warmup barely trains; lower it for short fine-tuning runs.</remarks>
    public int WarmupSteps { get; set; } = 32000;

    /// <summary>AdamW's β₁ (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>AdamW's β₂ (0.999).</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>AdamW's ε (1e-8).</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>The seed of training's random choices (the step, the prompt, the masks).</summary>
    public int TrainingSeed { get; set; } = 1;
}
