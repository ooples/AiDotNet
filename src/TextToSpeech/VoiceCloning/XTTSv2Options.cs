using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Configuration of Casanova et al.'s XTTS-v2 and Coqui's released checkpoint.</summary>
/// <remarks>
/// Defaults describe <c>coqui/XTTS-v2</c>: a 30-layer, 1024-wide GPT-2, a 1024-entry single-codebook
/// dVAE, the six-block conditioning encoder and 32-latent Perceiver, H/ASP speaker encoder, and the
/// speaker-conditioned 24 kHz HiFi-GAN decoder. The paper reports 6,681 text symbols; the released
/// <c>vocab.json</c> and checkpoint both have 6,681 embedding rows.
/// </remarks>
public class XTTSv2Options : CodecTtsOptions
{
    /// <summary>The language ids in the released checkpoint, in model-card order.</summary>
    public static readonly string[] ReleasedLanguages =
        ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"];

    /// <summary>Creates the released XTTS-v2 configuration and the paper's full-training optimizer recipe.</summary>
    public XTTSv2Options()
    {
        SampleRate = 24000;
        InputSampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        NumCodebooks = 1;
        CodebookSize = 1024;
        CodecFrameRate = 22;
        TextEncoderDim = 1024;
        NumEncoderLayers = 0;
        LLMDim = 1024;
        NumLLMLayers = 30;
        NumHeads = 16;
        SpeakerEmbeddingDim = 512;
        VocabSize = 6681;
        MaxTextLength = 402;
        MaxCodecFrames = 605;
        LearningRate = 5e-5;
        WeightDecay = 1e-2;
        DropoutRate = 0.1;
        LanguageModelName = "GPT-2";
    }

    /// <summary>Creates a deep copy.</summary>
    public XTTSv2Options(XTTSv2Options other) : base(other)
    {
        Validation.Guard.NotNull(other);
        InputSampleRate = other.InputSampleRate;
        PaperTextVocabularySize = other.PaperTextVocabularySize;
        AudioVocabularySize = other.AudioVocabularySize;
        StartTextToken = other.StartTextToken;
        StopTextToken = other.StopTextToken;
        StartAudioToken = other.StartAudioToken;
        StopAudioToken = other.StopAudioToken;
        MaxPromptTokens = other.MaxPromptTokens;
        ConditioningBlocks = other.ConditioningBlocks;
        ConditioningHeads = other.ConditioningHeads;
        PerceiverDepth = other.PerceiverDepth;
        PerceiverLatents = other.PerceiverLatents;
        PerceiverHeads = other.PerceiverHeads;
        PerceiverHeadDim = other.PerceiverHeadDim;
        PerceiverFeedForwardMultiple = other.PerceiverFeedForwardMultiple;
        DvaeHiddenDim = other.DvaeHiddenDim;
        DvaeCodebookDim = other.DvaeCodebookDim;
        DvaeLayers = other.DvaeLayers;
        DvaeResidualBlocks = other.DvaeResidualBlocks;
        GptCodeStride = other.GptCodeStride;
        ConditioningFftSize = other.ConditioningFftSize;
        ConditioningWindowSize = other.ConditioningWindowSize;
        ConditioningHopSize = other.ConditioningHopSize;
        DvaeFftSize = other.DvaeFftSize;
        DvaeWindowSize = other.DvaeWindowSize;
        DvaeHopSize = other.DvaeHopSize;
        MelMaxFrequency = other.MelMaxFrequency;
        MaximumReferenceSeconds = other.MaximumReferenceSeconds;
        ConditioningSeconds = other.ConditioningSeconds;
        ConditioningChunkSeconds = other.ConditioningChunkSeconds;
        MinimumConditioningChunkSeconds = other.MinimumConditioningChunkSeconds;
        SpeakerEncoderFilters = (int[])other.SpeakerEncoderFilters.Clone();
        DecoderInitialChannels = other.DecoderInitialChannels;
        DecoderUpsampleRates = (int[])other.DecoderUpsampleRates.Clone();
        DecoderUpsampleKernels = (int[])other.DecoderUpsampleKernels.Clone();
        DecoderResblockKernels = (int[])other.DecoderResblockKernels.Clone();
        DecoderResblockDilations = other.DecoderResblockDilations.Select(x => (int[])x.Clone()).ToArray();
        Temperature = other.Temperature;
        RepetitionPenalty = other.RepetitionPenalty;
        TopK = other.TopK;
        TopP = other.TopP;
        LengthPenalty = other.LengthPenalty;
        Speed = other.Speed;
        Language = other.Language;
        SamplingSeed = other.SamplingSeed;
        AdamBeta1 = other.AdamBeta1;
        AdamBeta2 = other.AdamBeta2;
        AdamEpsilon = other.AdamEpsilon;
        TextLossWeight = other.TextLossWeight;
        AudioLossWeight = other.AudioLossWeight;
        GradientAccumulationSteps = other.GradientAccumulationSteps;
        LearningRateMilestones = (int[])other.LearningRateMilestones.Clone();
        LearningRateGamma = other.LearningRateGamma;
        TokenizerPath = other.TokenizerPath;
        MelStatsPath = other.MelStatsPath;
    }

    /// <summary>The public GPT fine-tuning recipe (the frozen dVAE and decoder are not updated).</summary>
    public static XTTSv2Options OfficialGptFineTuning() => new() { LearningRate = 5e-6 };

    /// <summary>Input/reference and dVAE sample rate (22.05 kHz); generated audio is 24 kHz.</summary>
    public int InputSampleRate { get; set; }

    /// <summary>Embedding rows in the released checkpoint.</summary>
    public int TextVocabularySize { get => VocabSize; set => VocabSize = value; }
    /// <summary>Vocabulary size stated by the paper and present in the pinned official checkpoint.</summary>
    public int PaperTextVocabularySize { get; set; } = 6681;
    /// <summary>1024 dVAE codes plus start and stop.</summary>
    public int AudioVocabularySize { get; set; } = 1026;
    /// <summary>Resolved from <c>[START]</c> in the released tokenizer when it is loaded.</summary>
    public int StartTextToken { get; set; } = -1;
    /// <summary>Resolved from <c>[STOP]</c> in the released tokenizer when it is loaded.</summary>
    public int StopTextToken { get; set; } = -1;
    public int StartAudioToken { get; set; } = 1024;
    public int StopAudioToken { get; set; } = 1025;
    public int MaxPromptTokens { get; set; } = 70;

    public int ConditioningBlocks { get; set; } = 6;
    public int ConditioningHeads { get; set; } = 16;
    public int PerceiverDepth { get; set; } = 2;
    public int PerceiverLatents { get; set; } = 32;
    public int PerceiverHeads { get; set; } = 8;
    public int PerceiverHeadDim { get; set; } = 64;
    public int PerceiverFeedForwardMultiple { get; set; } = 4;

    public int DvaeHiddenDim { get; set; } = 512;
    public int DvaeCodebookDim { get; set; } = 512;
    public int DvaeLayers { get; set; } = 2;
    public int DvaeResidualBlocks { get; set; } = 3;
    /// <summary>Samples represented by one GPT/dVAE code at 22.05 kHz (21.5332 codes/s).</summary>
    public int GptCodeStride { get; set; } = 1024;
    public double AcousticTokenRate => (double)InputSampleRate / GptCodeStride;

    /// <summary>Perceiver conditioning mel parameters from the released inference path.</summary>
    public int ConditioningFftSize { get; set; } = 2048;
    public int ConditioningWindowSize { get; set; } = 1024;
    public int ConditioningHopSize { get; set; } = 256;
    public int DvaeFftSize { get; set; } = 1024;
    public int DvaeWindowSize { get; set; } = 1024;
    public int DvaeHopSize { get; set; } = 256;
    public double MelMaxFrequency { get; set; } = 8000;
    public double MaximumReferenceSeconds { get; set; } = 30;
    public double ConditioningSeconds { get; set; } = 6;
    public double ConditioningChunkSeconds { get; set; } = 6;
    public double MinimumConditioningChunkSeconds { get; set; } = 0.33;
    /// <summary>H/ASP SE-ResNet stage widths from the released <c>config_se.json</c>.</summary>
    public int[] SpeakerEncoderFilters { get; set; } = [32, 64, 128, 256];

    public int DecoderInitialChannels { get; set; } = 512;
    public int[] DecoderUpsampleRates { get; set; } = [8, 8, 2, 2];
    public int[] DecoderUpsampleKernels { get; set; } = [16, 16, 4, 4];
    public int[] DecoderResblockKernels { get; set; } = [3, 7, 11];
    public int[][] DecoderResblockDilations { get; set; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    public double Temperature { get; set; } = 0.75;
    public double RepetitionPenalty { get; set; } = 10;
    public int TopK { get; set; } = 50;
    public double TopP { get; set; } = 0.85;
    public double LengthPenalty { get; set; } = 1;
    public double Speed { get; set; } = 1;
    public string Language { get; set; } = "en";
    public int SamplingSeed { get; set; } = 1;

    public double AdamBeta1 { get; set; } = 0.9;
    public double AdamBeta2 { get; set; } = 0.96;
    public double AdamEpsilon { get; set; } = 1e-8;
    public double TextLossWeight { get; set; } = 0.01;
    public double AudioLossWeight { get; set; } = 1;
    public int GradientAccumulationSteps { get; set; } = 16;
    public int[] LearningRateMilestones { get; set; } = [5000, 150000, 300000];
    public double LearningRateGamma { get; set; } = 0.5;

    /// <summary>Optional Hugging Face/tokenizers tokenizer JSON shipped as <c>vocab.json</c>.</summary>
    public string? TokenizerPath { get; set; }
    /// <summary>Optional released <c>mel_stats.pth</c>.</summary>
    public string? MelStatsPath { get; set; }
}
