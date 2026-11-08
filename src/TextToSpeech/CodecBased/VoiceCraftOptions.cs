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
    /// <summary>The causal Transformer width, forwarded to <see cref="CodecTtsOptions.LLMDim"/>.</summary>
    public new int HiddenDim { get => LLMDim; set => LLMDim = value; }

    /// <summary>The causal Transformer stack depth, forwarded to <see cref="CodecTtsOptions.NumLLMLayers"/>.</summary>
    public new int NumEncoderLayers { get => NumLLMLayers; set => NumLLMLayers = value; }

    /// <summary>The causal Transformer stack depth, forwarded to <see cref="CodecTtsOptions.NumLLMLayers"/>.</summary>
    public new int NumDecoderLayers { get => NumLLMLayers; set => NumLLMLayers = value; }

    /// <summary>The text embedding vocabulary, forwarded to <see cref="TextVocabularySize"/>.</summary>
    public new int VocabSize { get => TextVocabularySize; set => TextVocabularySize = value; }

    /// <summary>The causal Transformer's training dropout, forwarded to <see cref="TransformerDropout"/>.</summary>
    public new double DropoutRate { get => TransformerDropout; set => TransformerDropout = value; }

    /// <summary>The GigaSpeech codec's fixed 16000 Hz sample rate.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("VoiceCraft's GigaSpeech codec is fixed at 16000 Hz; native geometry rejection and paired codec round-trip tests cover this contract.")]
    public new int SampleRate { get => base.SampleRate; set => base.SampleRate = value; }

    /// <summary>The GigaSpeech codec's fixed 320-sample hop.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("VoiceCraft's GigaSpeech codec uses a fixed 320-sample hop; native geometry tests reject other hops and verify decoded lengths.")]
    public new int HopSize { get => base.HopSize; set => base.HopSize = value; }

    /// <summary>The GigaSpeech codec's fixed 50 Hz frame rate.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("VoiceCraft's codec frame rate is fixed to 16000/320=50 Hz; native geometry and codec round-trip tests verify it.")]
    public new int CodecFrameRate { get => base.CodecFrameRate; set => base.CodecFrameRate = value; }

    /// <summary>The codebook count, paired with the codec bandwidth, quantizer count and loss weights.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("NumCodebooks must match Codec.NumQuantizers, its bandwidth and CodebookLossWeights; native tests reject unpaired values and round-trip paired two/four-codebook models.")]
    public new int NumCodebooks { get => base.NumCodebooks; set => base.NumCodebooks = value; }

    /// <summary>The codec vocabulary, paired with the nested codec configuration and bandwidth.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("CodebookSize must match Codec.CodebookSize and its bandwidth; native tests reject unpaired values and round-trip paired vocabularies of eight/sixteen codes.")]
    public new int CodebookSize { get => base.CodebookSize; set => base.CodebookSize = value; }

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

    /// <summary>Gets or sets the hidden width of the feed-forward network.</summary>
    public int FeedForwardDim { get; set; }
    /// <summary>Gets or sets the text vocabulary size. Defaults to <c>100</c>.</summary>
    public int TextVocabularySize { get; set; } = 100;
    /// <summary>Gets or sets the max mask spans. Defaults to <c>3</c>.</summary>
    public int MaxMaskSpans { get; set; } = 3;
    /// <summary>Gets or sets the minimum mask frames. Defaults to <c>1</c>.</summary>
    public int MinimumMaskFrames { get; set; } = 1;
    /// <summary>Gets or sets the maximum mask frames. Defaults to <c>600</c>.</summary>
    public int MaximumMaskFrames { get; set; } = 600;
    /// <summary>Gets or sets the minimum gap frames. Defaults to <c>5</c>.</summary>
    public int MinimumGapFrames { get; set; } = 5;
    /// <summary>Gets or sets the codebook loss weights.</summary>
    public double[] CodebookLossWeights { get; set; } = [5, 1, 0.5, 0.1];

    /// <summary>Gets or sets the dropout probability for the text embedding. Defaults to <c>0.1</c>.</summary>
    public double TextEmbeddingDropout { get; set; } = 0.1;
    /// <summary>Gets or sets the dropout probability for the audio embedding.</summary>
    public double AudioEmbeddingDropout { get; set; }
    /// <summary>Gets or sets the dropout probability for the text position. Defaults to <c>0.1</c>.</summary>
    public double TextPositionDropout { get; set; } = 0.1;
    /// <summary>Gets or sets the dropout probability for the audio position. Defaults to <c>0.1</c>.</summary>
    public double AudioPositionDropout { get; set; } = 0.1;
    /// <summary>Gets or sets the dropout probability for the transformer. Defaults to <c>0.1</c>.</summary>
    public double TransformerDropout { get; set; } = 0.1;

    /// <summary>Top-k sampling. Zero selects the paper's nucleus-only rule; the current release recommends 40.</summary>
    public int TopK { get; set; }
    /// <summary>Gets or sets the cumulative probability threshold for nucleus sampling. Defaults to <c>0.8</c>.</summary>
    public double TopP { get; set; } = 0.8;
    /// <summary>Gets or sets the logit temperature used during sampling. Defaults to <c>1</c>.</summary>
    public double Temperature { get; set; } = 1;
    /// <summary>Gets or sets the sample batch size. Defaults to <c>3</c>.</summary>
    public int SampleBatchSize { get; set; } = 3;
    /// <summary>Gets or sets the minimum generated frames. Defaults to <c>10</c>.</summary>
    public int MinimumGeneratedFrames { get; set; } = 10;
    /// <summary>Gets or sets the maximum frames per phone. Defaults to <c>10</c>.</summary>
    public int MaximumFramesPerPhone { get; set; } = 10;
    /// <summary>Gets or sets the stop silence repetition. Defaults to <c>3</c>.</summary>
    public int StopSilenceRepetition { get; set; } = 3;
    /// <summary>Gets or sets the silence tokens.</summary>
    public int[] SilenceTokens { get; set; } = [1388, 1898, 131];
    /// <summary>Gets or sets the random seed used during sampling. Defaults to <c>1</c>.</summary>
    public int SamplingSeed { get; set; } = 1;

    /// <summary>Gets or sets the total training steps. Defaults to <c>50000</c>.</summary>
    public int TotalTrainingSteps { get; set; } = 50000;
    /// <summary>Gets or sets the eden warmup steps. Defaults to <c>500</c>.</summary>
    public int EdenWarmupSteps { get; set; } = 500;
    /// <summary>Gets or sets the eden step. Defaults to <c>3000</c>.</summary>
    public int EdenStep { get; set; } = 3000;
    /// <summary>Gets or sets the eden epoch. Defaults to <c>4</c>.</summary>
    public int EdenEpoch { get; set; } = 4;
    /// <summary>Gets or sets the pseudo epoch steps. Defaults to <c>3000</c>.</summary>
    public int PseudoEpochSteps { get; set; } = 3000;
    /// <summary>Gets or sets whether training uses the ScaledAdam optimizer. Defaults to <c>true</c>.</summary>
    public bool UseScaledAdam { get; set; } = true;

    /// <summary>The frozen 16 kHz, 50 Hz, four-by-2048 EnCodec trained by the authors on GigaSpeech.</summary>
    public EnCodecOptions Codec { get; set; }

    /// <summary>Gets or sets the token ID for empty.</summary>
    public int EmptyToken => CodebookSize;
    /// <summary>Gets or sets the token ID for end of generation.</summary>
    public int EndOfGenerationToken => CodebookSize + 1;
    /// <summary>Gets or sets the token ID for audio padding.</summary>
    public int AudioPaddingToken => CodebookSize + 2;
    /// <summary>Gets or sets the token ID for end of utterance.</summary>
    public int EndOfUtteranceToken => CodebookSize + 3;
    /// <summary>Gets or sets the audio vocabulary size.</summary>
    public int AudioVocabularySize => CodebookSize + 4;

    /// <summary>Released 830M layout, including the release's 16 heads and current top-40 sampling.</summary>
    public static VoiceCraftOptions Official830M() => new() { NumHeads = 16, TopK = 40, TopP = 1 };

    /// <summary>Released 330M layout.</summary>
    public static VoiceCraftOptions Official330M() => new()
    {
        LLMDim = 1024,
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
