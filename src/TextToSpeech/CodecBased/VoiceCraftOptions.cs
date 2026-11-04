using AiDotNet.Audio.Generation;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Configuration for VoiceCraft (Peng et al. 2024).</summary>
/// <remarks>
/// Defaults follow the released 830M model. Table 9 says 12 attention heads, which cannot divide its 2048-wide
/// hidden states; released 830M checkpoints use 16. <see cref="Official830M"/> selects that released configuration.
/// The released 330M model is not
/// Table 9's 120M ablation: it is 1024-wide, 24 layers and 16 heads, exposed by <see cref="Official330M"/>.
/// </remarks>
public class VoiceCraftOptions : CodecTtsOptions
{
    /// <summary>Optional externally precomputed IPA phones using the checkpoint's inventory. Null runs the released eSpeak NG front end.</summary>
    public Func<string, IReadOnlyList<string>>? TextPhonemizer { get; set; }
    /// <summary>Creates the paper's main 830M configuration.</summary>
    public VoiceCraftOptions()
    {
        SampleRate = 16000;
        HopSize = 320;
        CodecFrameRate = 50;
        NumCodebooks = 4;
        CodebookSize = 2048;
        LLMDim = 2048;
        TextEncoderDim = 2048;
        NumLLMLayers = 16;
        NumHeads = 16;
        FeedForwardDim = 8192;
        MaxTextLength = 400;
        MaxCodecFrames = 1000;
        LearningRate = 0.05;
        Codec = VoiceCraftCodec();
    }

    /// <summary>Copies a configuration.</summary>
    public VoiceCraftOptions(VoiceCraftOptions other) : base(other)
    {
        Validation.Guard.NotNull(other);
        TextPhonemizer = other.TextPhonemizer;
        FeedForwardDim = other.FeedForwardDim;
        TextVocabularySize = other.TextVocabularySize;
        MaxMaskSpans = other.MaxMaskSpans;
        MinimumMaskFrames = other.MinimumMaskFrames;
        MaximumMaskFrames = other.MaximumMaskFrames;
        MinimumGapFrames = other.MinimumGapFrames;
        CodebookLossWeights = (double[])other.CodebookLossWeights.Clone();
        TextEmbeddingDropout = other.TextEmbeddingDropout;
        AudioEmbeddingDropout = other.AudioEmbeddingDropout;
        TextPositionDropout = other.TextPositionDropout;
        AudioPositionDropout = other.AudioPositionDropout;
        TransformerDropout = other.TransformerDropout;
        TopK = other.TopK;
        TopP = other.TopP;
        Temperature = other.Temperature;
        SampleBatchSize = other.SampleBatchSize;
        MinimumGeneratedFrames = other.MinimumGeneratedFrames;
        MaximumFramesPerPhone = other.MaximumFramesPerPhone;
        StopSilenceRepetition = other.StopSilenceRepetition;
        SilenceTokens = (int[])other.SilenceTokens.Clone();
        SamplingSeed = other.SamplingSeed;
        TotalTrainingSteps = other.TotalTrainingSteps;
        EdenWarmupSteps = other.EdenWarmupSteps;
        EdenStep = other.EdenStep;
        EdenEpoch = other.EdenEpoch;
        PseudoEpochSteps = other.PseudoEpochSteps;
        UseScaledAdam = other.UseScaledAdam;
        Codec = (EnCodecOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(other.Codec);
    }

    public int FeedForwardDim { get; set; }
    public int TextVocabularySize { get; set; } = 100;
    public int MaxMaskSpans { get; set; } = 3;
    public int MinimumMaskFrames { get; set; } = 1;
    public int MaximumMaskFrames { get; set; } = 600;
    public int MinimumGapFrames { get; set; } = 5;
    public double[] CodebookLossWeights { get; set; } = [5, 1, 0.5, 0.1];

    public double TextEmbeddingDropout { get; set; } = 0.1;
    public double AudioEmbeddingDropout { get; set; }
    public double TextPositionDropout { get; set; } = 0.1;
    public double AudioPositionDropout { get; set; } = 0.1;
    public double TransformerDropout { get; set; } = 0.1;

    /// <summary>Top-k sampling. Zero selects the paper's nucleus-only rule; the current release recommends 40.</summary>
    public int TopK { get; set; }
    public double TopP { get; set; } = 0.8;
    public double Temperature { get; set; } = 1;
    public int SampleBatchSize { get; set; } = 3;
    public int MinimumGeneratedFrames { get; set; } = 10;
    public int MaximumFramesPerPhone { get; set; } = 10;
    public int StopSilenceRepetition { get; set; } = 3;
    public int[] SilenceTokens { get; set; } = [1388, 1898, 131];
    public int SamplingSeed { get; set; } = 1;

    public int TotalTrainingSteps { get; set; } = 50000;
    public int EdenWarmupSteps { get; set; } = 500;
    public int EdenStep { get; set; } = 3000;
    public int EdenEpoch { get; set; } = 4;
    public int PseudoEpochSteps { get; set; } = 3000;
    public bool UseScaledAdam { get; set; } = true;

    /// <summary>The frozen 16 kHz, 50 Hz, four-by-2048 EnCodec trained by the authors on GigaSpeech.</summary>
    public EnCodecOptions Codec { get; set; }

    public int EmptyToken => CodebookSize;
    public int EndOfGenerationToken => CodebookSize + 1;
    public int AudioPaddingToken => CodebookSize + 2;
    public int EndOfUtteranceToken => CodebookSize + 3;
    public int AudioVocabularySize => CodebookSize + 4;

    /// <summary>Released 830M layout, including the release's 16 heads and current top-40 sampling.</summary>
    public static VoiceCraftOptions Official830M() => new() { NumHeads = 16, TopK = 40, TopP = 1 };

    /// <summary>Released 330M layout.</summary>
    public static VoiceCraftOptions Official330M() => new()
    {
        LLMDim = 1024,
        TextEncoderDim = 1024,
        FeedForwardDim = 4096,
        NumLLMLayers = 24,
        NumHeads = 16,
        TopK = 40,
        TopP = 1
    };

    /// <summary>The authors' TTS-enhanced 330M layout.</summary>
    public static VoiceCraftOptions Official330MTtsEnhanced() => new(Official330M())
    {
        TextVocabularySize = 120,
        TextEmbeddingDropout = 0,
        TextPositionDropout = 0,
        AudioPositionDropout = 0,
        TransformerDropout = 0,
        CodebookLossWeights = [2, 1, 1, 1],
        LearningRate = 1e-5,
        TotalTrainingSteps = 500000,
        UseScaledAdam = false,
        MaxCodecFrames = 800
    };

    private static EnCodecOptions VoiceCraftCodec() => new()
    {
        SampleRate = 16000,
        Filters = 64,
        Dimension = 128,
        ResidualKernelSizes = [3, 1],
        TrueSkip = true,
        Ratios = [8, 5, 4, 2],
        NumQuantizers = 4,
        CodebookSize = 2048,
        TargetBandwidthKbps = 2.2,
        TargetBandwidths = [2.2],
        SegmentSize = 16000,
        IncludeDiscriminators = false,
        Causal = false,
        TimeGroupNormalization = false,
        ReflectPadding = false
    };
}
