using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Options for Resemble AI's original English Chatterbox release.</summary>
/// <remarks>
/// <para>The defaults reproduce the released 2025 model: the 520M T3 decoder (30 Llama layers, 1,024 channels),
/// the 25 Hz S3Tokenizer-v2 vocabulary, the causal S3Gen flow, CAM++ and the 24 kHz HiFT vocoder.</para>
/// </remarks>
public class ChatterboxOptions : CosyVoiceFamilyOptions
{
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

    public int TextStartToken { get; set; } = 255;
    public int TextStopToken { get; set; } = 0;
    public int SpeechStartToken { get; set; } = 6561;
    public int SpeechStopToken { get; set; } = 6562;
    public int SpeechVocabularySize { get; set; } = 8194;
    public int MaxSpeechTokens { get; set; } = 4096;

    /// <summary>The original synchronous wrapper caps a generation at 1,000 new speech tokens.</summary>
    public int MaxGeneratedSpeechTokens { get; set; } = 1000;

    public int LmFeedForwardDim { get; set; } = 4096;
    public int KeyValueHeads { get; set; } = 16;
    public double RopeTheta { get; set; } = 500000.0;
    public double RmsEpsilon { get; set; } = 1e-5;
    public double RopeScalingFactor { get; set; } = 8.0;
    public double RopeLowFrequencyFactor { get; set; } = 1.0;
    public double RopeHighFrequencyFactor { get; set; } = 4.0;
    public int RopeOriginalMaxPositions { get; set; } = 8192;

    public int PerceiverQueries { get; set; } = 32;
    public int PerceiverHeads { get; set; } = 4;
    public double PerceiverDropout { get; set; } = 0.2;
    public int SpeechConditionPromptTokens { get; set; } = 150;

    public int T3SpeakerEmbeddingDim { get; set; } = 256;
    public int VoiceEncoderMelChannels { get; set; } = 40;
    public int VoiceEncoderHiddenDim { get; set; } = 256;
    public int VoiceEncoderLayers { get; set; } = 3;
    public int VoiceEncoderPartialFrames { get; set; } = 160;

    /// <summary>Partials per second setting used by <c>embeds_from_wavs</c>; 1.3 gives the release's 77-frame step.</summary>
    public double VoiceEncoderPartialRate { get; set; } = 1.3;

    public int EncoderReferenceSeconds { get; set; } = 6;
    public int DecoderReferenceSeconds { get; set; } = 10;

    public int StaticChunkSize { get; set; } = 0;
    public int TokenMelRatio { get; set; } = 2;
    public int Lookahead { get; set; } = 3;

    public double Exaggeration { get; set; } = 0.5;
    public double ClassifierFreeGuidanceWeight { get; set; } = 0.5;
    public double Temperature { get; set; } = 0.8;
    public double MinP { get; set; } = 0.05;
    public double GenerationTopP { get; set; } = 1.0;
    public double RepetitionPenalty { get; set; } = 1.2;

    public double TextLossWeight { get; set; } = 1.0;
    public double SpeechLossWeight { get; set; } = 1.0;
}
