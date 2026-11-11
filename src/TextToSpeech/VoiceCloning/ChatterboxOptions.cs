using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>The released Chatterbox checkpoint a model follows.</summary>
public enum ChatterboxVariant
{
    /// <summary>The original English model (<c>t3_cfg</c>, a 704-symbol text vocabulary).</summary>
    English,

    /// <summary>The 23-language model, version 2 (<c>t3_mtl23ls_v2</c>, the release's default).</summary>
    MultilingualV2,

    /// <summary>The 23-language model, version 3 (<c>t3_mtl23ls_v3</c>).</summary>
    MultilingualV3,

    /// <summary>Chatterbox Turbo (<c>ResembleAI/chatterbox-turbo</c>): a GPT-2 medium T3 over GPT-2 text tokens and a two-step
    /// mean-flow S3Gen.</summary>
    Turbo,

    /// <summary>Chatterbox Nano (<c>ResembleAI/chatterbox-nano</c>): Turbo with a GPT-2 small T3.</summary>
    Nano,
}

/// <summary>Options for Resemble AI's Chatterbox releases (the original English model by default; see
/// <see cref="Multilingual"/>).</summary>
/// <remarks>
/// <para>The defaults reproduce the released 2025 model: the 520M T3 decoder (30 Llama layers, 1,024 channels),
/// the 25 Hz S3Tokenizer-v2 vocabulary, the causal S3Gen flow, CAM++ and the 24 kHz HiFT vocoder.</para>
/// </remarks>
public class ChatterboxOptions : CosyVoiceConformerOptions
{
    /// <summary>S3Tokenizer v2's fixed eight-dimensional ternary FSQ vocabulary.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("S3Tokenizer v2 requires exactly 3^8 = 6561 codes. ChatterboxOptionContextTests rejects unpaired cardinalities and exercises valid native codec configurations.")]
    public override int CodebookSize { get => base.CodebookSize; set => base.CodebookSize = value; }

    /// <summary>S3Tokenizer v2's state width, equal to 64 times its head count.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("S3Tokenizer v2 uses 64-wide rotary heads. ChatterboxOptionContextTests verifies native encoding with paired 64/1 and 128/2 widths and rejects unpaired changes.")]
    public override int SpeechTokenizerState { get => base.SpeechTokenizerState; set => base.SpeechTokenizerState = value; }

    /// <summary>S3Tokenizer v2's count of 64-wide rotary heads.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("S3Tokenizer v2 uses 64-wide rotary heads. ChatterboxOptionContextTests verifies native encoding with paired 64/1 and 128/2 widths and rejects unpaired changes.")]
    public override int SpeechTokenizerHeads { get => base.SpeechTokenizerHeads; set => base.SpeechTokenizerHeads = value; }

    /// <summary>Initializes a new instance with default values.</summary>
    public ChatterboxOptions()
    {
        SampleRate = 24000;
        MelChannels = 80;
        HopSize = 480;
        FftSize = 1920;

        NumCodebooks = 1;
        CodebookSize = 6561;
        CodecFrameRate = 25;

        VocabSize = 704;
        MaxTextLength = 2048;
        LLMDim = 1024;
        NumLLMLayers = 30;
        NumHeads = 16;
        DropoutRate = 0.0;

        SpeakerEmbeddingDim = 192;
        DecoderChannels = new[] { 256 };
        HiftUpsampleRates = new[] { 8, 5, 3 };
        HiftUpsampleKernels = new[] { 16, 11, 7 };
        HiftSourceKernels = new[] { 7, 7, 11 };
        WarmupSteps = 0;
        VocoderSegmentLength = 24480;
    }

    /// <summary>Initializes a copy of every Chatterbox setting.</summary>
    public ChatterboxOptions(ChatterboxOptions other)
        : base(other)
    {
        TokenizerPath = other.TokenizerPath;
        TextStartToken = other.TextStartToken;
        TextStopToken = other.TextStopToken;
        SpeechStartToken = other.SpeechStartToken;
        SpeechStopToken = other.SpeechStopToken;
        SpeechVocabularySize = other.SpeechVocabularySize;
        MaxSpeechTokens = other.MaxSpeechTokens;
        MaxGeneratedSpeechTokens = other.MaxGeneratedSpeechTokens;
        LmFeedForwardDim = other.LmFeedForwardDim;
        KeyValueHeads = other.KeyValueHeads;
        RopeTheta = other.RopeTheta;
        RmsEpsilon = other.RmsEpsilon;
        RopeScalingFactor = other.RopeScalingFactor;
        RopeLowFrequencyFactor = other.RopeLowFrequencyFactor;
        RopeHighFrequencyFactor = other.RopeHighFrequencyFactor;
        RopeOriginalMaxPositions = other.RopeOriginalMaxPositions;
        PerceiverQueries = other.PerceiverQueries;
        PerceiverHeads = other.PerceiverHeads;
        PerceiverDropout = other.PerceiverDropout;
        SpeechConditionPromptTokens = other.SpeechConditionPromptTokens;
        T3SpeakerEmbeddingDim = other.T3SpeakerEmbeddingDim;
        VoiceEncoderMelChannels = other.VoiceEncoderMelChannels;
        VoiceEncoderHiddenDim = other.VoiceEncoderHiddenDim;
        VoiceEncoderLayers = other.VoiceEncoderLayers;
        VoiceEncoderPartialFrames = other.VoiceEncoderPartialFrames;
        VoiceEncoderPartialRate = other.VoiceEncoderPartialRate;
        EncoderReferenceSeconds = other.EncoderReferenceSeconds;
        DecoderReferenceSeconds = other.DecoderReferenceSeconds;
        TokenMelRatio = other.TokenMelRatio;
        Lookahead = other.Lookahead;
        StaticChunkSize = other.StaticChunkSize;
        Exaggeration = other.Exaggeration;
        ClassifierFreeGuidanceWeight = other.ClassifierFreeGuidanceWeight;
        Temperature = other.Temperature;
        MinP = other.MinP;
        GenerationTopP = other.GenerationTopP;
        RepetitionPenalty = other.RepetitionPenalty;
        TextLossWeight = other.TextLossWeight;
        SpeechLossWeight = other.SpeechLossWeight;
        Watermark = other.Watermark;
        WatermarkModelPath = other.WatermarkModelPath;
        Variant = other.Variant;
        LanguageId = other.LanguageId;
        CangjieMappingPath = other.CangjieMappingPath;
        ChineseSegmenterPath = other.ChineseSegmenterPath;
        HebrewDiacritizerPath = other.HebrewDiacritizerPath;
        RussianStresserPath = other.RussianStresserPath;
        GenerationTopK = other.GenerationTopK;
        ReferenceLoudness = other.ReferenceLoudness;
        DistillationTeacherSteps = other.DistillationTeacherSteps;
    }

    /// <summary>Options for Chatterbox Turbo, or Nano (<c>ChatterboxTurboTTS.from_local</c>'s T3Config and
    /// <c>generate()</c>'s defaults): GPT-2 medium (24 layers, 1,024 wide) or small (12, 768) over the 50,276 GPT-2 text
    /// tokens, 6,563 speech tokens, 375 raw prompt tokens from 15 s of reference; sampling at temperature 0.8 with top-k
    /// 1,000, top-p 0.95 and repetition penalty 1.2, no guidance; the reference normalized to −27 LUFS; a two-step mean-flow
    /// S3Gen.</summary>
    public static ChatterboxOptions Turbo(bool nano = false)
    {
        int width = nano ? 768 : 1024;
        return new ChatterboxOptions
        {
            Variant = nano ? ChatterboxVariant.Nano : ChatterboxVariant.Turbo,
            VocabSize = 50276,
            SpeechVocabularySize = 6563,
            LLMDim = width,
            NumLLMLayers = nano ? 12 : 24,
            NumHeads = nano ? 12 : 16,
            KeyValueHeads = nano ? 12 : 16,
            LmFeedForwardDim = 4 * width,
            SpeechConditionPromptTokens = 375,
            EncoderReferenceSeconds = 15,
            Exaggeration = 0,
            ClassifierFreeGuidanceWeight = 0,
            MinP = 0,
            GenerationTopP = 0.95,
            GenerationTopK = 1000,
            ReferenceLoudness = -27,
            FlowTimesteps = 2,
        };
    }

    /// <summary>Top-k sampling of speech tokens (Hugging Face's <c>TopKLogitsWarper</c>); 0 keeps every token. The original
    /// and multilingual releases do not use it (0); Turbo samples from the top 1,000.</summary>
    public int GenerationTopK { get; set; }

    /// <summary>The integrated loudness (LUFS, ITU-R BS.1770 as pyloudnorm measures it) a reference recording is normalized
    /// to before conditioning, or null to use it as recorded (the original and multilingual releases). Turbo uses −27.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("The loudness applies to a reference recording when a voice is created, which token-id inputs never reach. ChatterboxTurboTests checks that CreateVoice normalizes the reference to it.")]
    public double? ReferenceLoudness { get; set; }

    /// <summary>Euler steps of the frozen teacher S3Gen per distillation target when training a mean-flow (Turbo, Nano)
    /// decoder: the teacher's average velocity over [t, r] (IntMeanFlow, Eq. 5–6; the paper's default 16).</summary>
    public int DistillationTeacherSteps { get; set; } = 16;

    /// <summary>Options for the 23-language release (<c>ChatterboxMultilingualTTS</c>): its 2,454-symbol grapheme vocabulary,
    /// otherwise the English model's sizes.</summary>
    /// <param name="variant">The multilingual checkpoint (version 2 by default, as the release).</param>
    public static ChatterboxOptions Multilingual(ChatterboxVariant variant = ChatterboxVariant.MultilingualV2)
    {
        if (variant == ChatterboxVariant.English) throw new ArgumentException("Choose a multilingual variant.", nameof(variant));
        return new ChatterboxOptions { Variant = variant, VocabSize = 2454 };
    }

    /// <summary>The released checkpoint this model follows: the text front end and the files
    /// <c>LoadPretrainedWeights</c> reads. Defaults to the original English model.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("The variant selects the text front end and the checkpoint files, which token-id inputs never reach. ChatterboxVariantTests checks that multilingual tokenization follows the release's MTLTokenizer and that the English model's does not.")]
    public ChatterboxVariant Variant { get; set; } = ChatterboxVariant.English;

    /// <summary>The language a multilingual model speaks (one of the release's 23 codes: ar, da, de, el, en, es, fi, fr, he,
    /// hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh). Defaults to <c>en</c>; the English model ignores it.</summary>
    public string LanguageId { get; set; } = "en";

    /// <summary>The release's <c>Cangjie5_TC.json</c>, which writes Chinese characters as Cangjie codes. Loading a
    /// multilingual model directory sets this automatically.</summary>
    public string? CangjieMappingPath { get; set; }

    /// <summary>The pkuseg model directory (<c>spacy_ontonotes</c>: <c>features.msgpack</c>, <c>weights.npz</c>) Chinese
    /// text is segmented with. When unset, pkuseg's own cache is used if it holds the model, else the model is downloaded
    /// once from the release's URL into <c>~/.aidotnet/models/pkuseg</c> and checked against its SHA-256.</summary>
    public string? ChineseSegmenterPath { get; set; }

    /// <summary>The Dicta diacritizer directory (<c>dicta-il/dictabert-large-char-menaked</c>: <c>config.json</c>,
    /// <c>tokenizer.json</c>, <c>model.safetensors</c>) Hebrew text gets its niqqud from. When unset, it is downloaded once
    /// from the Hugging Face Hub into its cache.</summary>
    public string? HebrewDiacritizerPath { get; set; }

    /// <summary>The Russian stresser directory (russian_text_stresser's <c>russian_dict.db</c>, <c>simple_cases.pkl</c>
    /// and its spaCy <c>ru_core_news_sm-3.6.0</c> files: <c>tokenizer</c>, <c>tok2vec/model</c>, <c>morphologizer/model</c>,
    /// <c>morphologizer/cfg</c>) Russian text gets its stress marks from. When unset, they are downloaded once from the
    /// stresser's release into <c>~/.aidotnet/models/russian-text-stresser</c>.</summary>
    public string? RussianStresserPath { get; set; }

    /// <summary>Path to the release's <c>tokenizer.json</c>. Loading a model directory sets this automatically.</summary>
    public string? TokenizerPath { get; set; }

    /// <summary>Gets or sets the token ID for text start. Defaults to <c>255</c>.</summary>
    public int TextStartToken { get; set; } = 255;
    /// <summary>Gets or sets the token ID for text stop. Defaults to <c>0</c>.</summary>
    public int TextStopToken { get; set; } = 0;
    /// <summary>Gets or sets the token ID for speech start. Defaults to <c>6561</c>.</summary>
    public int SpeechStartToken { get; set; } = 6561;
    /// <summary>Gets or sets the token ID for speech stop. Defaults to <c>6562</c>.</summary>
    public int SpeechStopToken { get; set; } = 6562;
    /// <summary>Gets or sets the speech vocabulary size. Defaults to <c>8194</c>.</summary>
    public int SpeechVocabularySize { get; set; } = 8194;
    /// <summary>Gets or sets the max speech tokens. Defaults to <c>4096</c>.</summary>
    public int MaxSpeechTokens { get; set; } = 4096;

    /// <summary>The original synchronous wrapper caps a generation at 1,000 new speech tokens.</summary>
    public int MaxGeneratedSpeechTokens { get; set; } = 1000;

    /// <summary>Gets or sets the feature width of the language-model feed forward. Defaults to <c>4096</c>.</summary>
    public int LmFeedForwardDim { get; set; } = 4096;
    /// <summary>Gets or sets the number of key/value attention heads. Defaults to <c>16</c>.</summary>
    public int KeyValueHeads { get; set; } = 16;
    /// <summary>Gets or sets the rotary positional-embedding frequency base for the model. Defaults to <c>500000.0</c>.</summary>
    public double RopeTheta { get; set; } = 500000.0;
    /// <summary>Gets or sets the RMS-normalization stability constant for the model. Defaults to <c>1e-5</c>.</summary>
    public double RmsEpsilon { get; set; } = 1e-5;
    /// <summary>Gets or sets the rotary positional-embedding scaling factor. Defaults to <c>8.0</c>.</summary>
    public double RopeScalingFactor { get; set; } = 8.0;
    /// <summary>Gets or sets the rotary positional embedding low frequency factor. Defaults to <c>1.0</c>.</summary>
    public double RopeLowFrequencyFactor { get; set; } = 1.0;
    /// <summary>Gets or sets the rotary positional embedding high frequency factor. Defaults to <c>4.0</c>.</summary>
    public double RopeHighFrequencyFactor { get; set; } = 4.0;
    /// <summary>Gets or sets the rotary positional embedding original max positions. Defaults to <c>8192</c>.</summary>
    public int RopeOriginalMaxPositions { get; set; } = 8192;

    /// <summary>Gets or sets the perceiver queries. Defaults to <c>32</c>.</summary>
    public int PerceiverQueries { get; set; } = 32;
    /// <summary>Gets or sets the number of attention heads in the perceiver. Defaults to <c>4</c>.</summary>
    public int PerceiverHeads { get; set; } = 4;
    /// <summary>Gets or sets the dropout probability for the perceiver. Defaults to <c>0.2</c>.</summary>
    public double PerceiverDropout { get; set; } = 0.2;
    /// <summary>Gets or sets the speech condition prompt tokens. Defaults to <c>150</c>.</summary>
    public int SpeechConditionPromptTokens { get; set; } = 150;

    /// <summary>Gets or sets the feature width of the t 3 speaker embedding. Defaults to <c>256</c>.</summary>
    public int T3SpeakerEmbeddingDim { get; set; } = 256;
    /// <summary>Gets or sets the voice encoder mel channels. Defaults to <c>40</c>.</summary>
    public int VoiceEncoderMelChannels { get; set; } = 40;
    /// <summary>Gets or sets the feature width of the voice encoder hidden. Defaults to <c>256</c>.</summary>
    public int VoiceEncoderHiddenDim { get; set; } = 256;
    /// <summary>Gets or sets the number of layers in the voice encoder. Defaults to <c>3</c>.</summary>
    public int VoiceEncoderLayers { get; set; } = 3;
    /// <summary>Gets or sets the voice encoder partial frames. Defaults to <c>160</c>.</summary>
    public int VoiceEncoderPartialFrames { get; set; } = 160;

    /// <summary>Partials per second setting used by <c>embeds_from_wavs</c>; 1.3 gives the release's 77-frame step.</summary>
    public double VoiceEncoderPartialRate { get; set; } = 1.3;

    /// <summary>Gets or sets the encoder reference duration, in seconds. Defaults to <c>6</c>.</summary>
    public int EncoderReferenceSeconds { get; set; } = 6;
    /// <summary>Gets or sets the decoder reference duration, in seconds. Defaults to <c>10</c>.</summary>
    public int DecoderReferenceSeconds { get; set; } = 10;

    /// <summary>Gets or sets the static chunk size. Defaults to <c>0</c>.</summary>
    public int StaticChunkSize { get; set; } = 0;
    /// <summary>Gets or sets the token mel ratio. Defaults to <c>2</c>.</summary>
    public int TokenMelRatio { get; set; } = 2;
    /// <summary>Gets or sets the lookahead. Defaults to <c>3</c>.</summary>
    public int Lookahead { get; set; } = 3;

    /// <summary>Gets or sets the exaggeration. Defaults to <c>0.5</c>.</summary>
    public double Exaggeration { get; set; } = 0.5;
    /// <summary>Gets or sets the weight of the classifier free guidance objective. Defaults to <c>0.5</c>.</summary>
    public double ClassifierFreeGuidanceWeight { get; set; } = 0.5;
    /// <summary>Gets or sets the logit temperature used during sampling. Defaults to <c>0.8</c>.</summary>
    public double Temperature { get; set; } = 0.8;
    /// <summary>Gets or sets the minimum token probability relative to the most likely token during sampling. Defaults to <c>0.05</c>.</summary>
    public double MinP { get; set; } = 0.05;
    /// <summary>Gets or sets the cumulative probability threshold for generated-token nucleus sampling. Defaults to <c>1.0</c>.</summary>
    public double GenerationTopP { get; set; } = 1.0;
    /// <summary>Gets or sets the penalty applied to repeated tokens during generation. Defaults to <c>1.2</c>.</summary>
    public double RepetitionPenalty { get; set; } = 1.2;

    /// <summary>Gets or sets the weight of the text loss. Defaults to <c>1.0</c>.</summary>
    public double TextLossWeight { get; set; } = 1.0;
    /// <summary>Gets or sets the weight of the speech loss. Defaults to <c>1.0</c>.</summary>
    public double SpeechLossWeight { get; set; } = 1.0;

    /// <summary>Whether synthesis puts Resemble's Perth implicit watermark on its output, as the release does (true). The
    /// watermark model is read from <see cref="WatermarkModelPath"/>; without it synthesis refuses rather than skip the
    /// watermark.</summary>
    public bool Watermark { get; set; } = true;

    /// <summary>The Perth model the watermark is read from: <c>perth_net_250000.pth.tar</c>, which ships with the
    /// resemble-perth package (<c>perth/perth_net/pretrained/implicit/</c>). Loading a model directory that holds that file
    /// sets this automatically.</summary>
    public string? WatermarkModelPath { get; set; }
}
