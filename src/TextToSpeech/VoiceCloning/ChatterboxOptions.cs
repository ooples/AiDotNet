using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for Resemble AI's original English Chatterbox release.</summary>
/// <remarks>
/// <para>The defaults reproduce the released 2025 model: the 520M T3 decoder (30 Llama layers, 1,024 channels),
/// the 25 Hz S3Tokenizer-v2 vocabulary, the causal S3Gen flow, CAM++ and the 24 kHz HiFT vocoder.</para>
/// </remarks>
public class ChatterboxOptions : CosyVoiceFamilyOptions
{
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
    }

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
}
