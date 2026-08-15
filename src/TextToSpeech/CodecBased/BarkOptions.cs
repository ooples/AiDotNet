using AiDotNet.Validation;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Describes one of Bark's transformer stages.
/// </summary>
/// <remarks>
/// Bark is not one language model. It uses a causal semantic model, a causal coarse-acoustic
/// model, and a bidirectional fine-acoustic model. Keeping their contracts separate prevents a
/// convenient-looking shared setting from silently producing a checkpoint-incompatible network.
/// </remarks>
public sealed class BarkStageOptions
{
    /// <summary>Input vocabulary width.</summary>
    public int InputVocabularySize { get; set; }

    /// <summary>Output vocabulary width.</summary>
    public int OutputVocabularySize { get; set; }

    /// <summary>Transformer hidden width.</summary>
    public int HiddenSize { get; set; }

    /// <summary>Transformer depth.</summary>
    public int NumberOfLayers { get; set; }

    /// <summary>Number of self-attention heads.</summary>
    public int NumberOfHeads { get; set; }

    /// <summary>Maximum sequence length accepted by the stage.</summary>
    public int BlockSize { get; set; }

    /// <summary>Feed-forward expansion width. Zero selects four times <see cref="HiddenSize"/>.</summary>
    public int FeedForwardSize { get; set; }

    /// <summary>Dropout probability used while training.</summary>
    public double Dropout { get; set; }

    /// <summary>Whether the stage is causal.</summary>
    public bool IsCausal { get; set; }

    internal BarkStageOptions Copy() => new()
    {
        InputVocabularySize = InputVocabularySize,
        OutputVocabularySize = OutputVocabularySize,
        HiddenSize = HiddenSize,
        NumberOfLayers = NumberOfLayers,
        NumberOfHeads = NumberOfHeads,
        BlockSize = BlockSize,
        FeedForwardSize = FeedForwardSize,
        Dropout = Dropout,
        IsCausal = IsCausal,
    };

    internal void Validate(string name)
    {
        if (InputVocabularySize <= 0)
            throw new ArgumentOutOfRangeException(name, $"{name}.InputVocabularySize must be positive.");
        if (OutputVocabularySize <= 0)
            throw new ArgumentOutOfRangeException(name, $"{name}.OutputVocabularySize must be positive.");
        if (HiddenSize <= 0)
            throw new ArgumentOutOfRangeException(name, $"{name}.HiddenSize must be positive.");
        if (NumberOfLayers <= 0)
            throw new ArgumentOutOfRangeException(name, $"{name}.NumberOfLayers must be positive.");
        if (NumberOfHeads <= 0)
            throw new ArgumentOutOfRangeException(name, $"{name}.NumberOfHeads must be positive.");
        if (HiddenSize % NumberOfHeads != 0)
            throw new ArgumentException(
                $"{name}.HiddenSize ({HiddenSize}) must be divisible by {name}.NumberOfHeads ({NumberOfHeads}).",
                name);
        if (BlockSize <= 0)
            throw new ArgumentOutOfRangeException(name, $"{name}.BlockSize must be positive.");
        if (FeedForwardSize < 0)
            throw new ArgumentOutOfRangeException(name, $"{name}.FeedForwardSize cannot be negative.");
        if (Dropout < 0.0 || Dropout >= 1.0)
            throw new ArgumentOutOfRangeException(name, $"{name}.Dropout must be in [0, 1).");
    }
}

/// <summary>Options for the Suno Bark text-prompted generative audio architecture.</summary>
/// <remarks>
/// <para>
/// Defaults match the public Bark checkpoints: three 24-layer, 1,024-wide transformers; a
/// 10,000-token semantic vocabulary; two coarse and eight total EnCodec codebooks at 24 kHz.
/// Construction remains allocation-light because transformer and embedding weights are lazy.
/// Use <see cref="TinyForTests"/> for fast conformance tests without changing the topology.
/// </para>
/// <para><b>For Beginners:</b> You normally do not need to change these values. A checkpoint was
/// trained for one exact configuration, so the model validates all connected dimensions for you
/// and reports the setting that is wrong.</para>
/// </remarks>
public class BarkOptions : CodecTtsOptions
{
    /// <summary>Creates the paper/checkpoint-faithful full configuration.</summary>
    public BarkOptions()
    {
        SampleRate = 24_000;
        NumCodebooks = 8;
        CodebookSize = 1_024;
        CodecFrameRate = 75;
        MaxTextLength = 256;
        MaxCodecFrames = 1_024;
        VocabSize = 129_600;
        LLMDim = 1_024;
        NumLLMLayers = 24;
        NumHeads = 16;
        DropoutRate = 0.0;

        Semantic = new BarkStageOptions
        {
            InputVocabularySize = 129_600,
            OutputVocabularySize = 10_048,
            HiddenSize = 1_024,
            NumberOfLayers = 24,
            NumberOfHeads = 16,
            BlockSize = 1_024,
            FeedForwardSize = 4_096,
            IsCausal = true,
        };
        Coarse = new BarkStageOptions
        {
            InputVocabularySize = 12_096,
            OutputVocabularySize = 12_096,
            HiddenSize = 1_024,
            NumberOfLayers = 24,
            NumberOfHeads = 16,
            BlockSize = 1_024,
            FeedForwardSize = 4_096,
            IsCausal = true,
        };
        Fine = new BarkStageOptions
        {
            InputVocabularySize = 1_056,
            OutputVocabularySize = 1_056,
            HiddenSize = 1_024,
            NumberOfLayers = 24,
            NumberOfHeads = 16,
            BlockSize = 1_024,
            FeedForwardSize = 4_096,
            IsCausal = false,
        };
    }

    /// <summary>Creates an independent copy.</summary>
    public BarkOptions(BarkOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Semantic = other.Semantic.Copy();
        Coarse = other.Coarse.Copy();
        Fine = other.Fine.Copy();
        SemanticVocabularySize = other.SemanticVocabularySize;
        TextEncodingOffset = other.TextEncodingOffset;
        SemanticPadTokenId = other.SemanticPadTokenId;
        SemanticInferenceTokenId = other.SemanticInferenceTokenId;
        TextPadTokenId = other.TextPadTokenId;
        CoarseSemanticPadTokenId = other.CoarseSemanticPadTokenId;
        CoarseInferenceTokenId = other.CoarseInferenceTokenId;
        SemanticRateHz = other.SemanticRateHz;
        CoarseRateHz = other.CoarseRateHz;
        NumberOfCoarseCodebooks = other.NumberOfCoarseCodebooks;
        NumberOfFineCodebooksGiven = other.NumberOfFineCodebooksGiven;
        SemanticHistoryLength = other.SemanticHistoryLength;
        CoarseHistoryLength = other.CoarseHistoryLength;
        CoarseSemanticContextLength = other.CoarseSemanticContextLength;
        CoarseSlidingWindowLength = other.CoarseSlidingWindowLength;
        FineWindowLength = other.FineWindowLength;
        FineWindowStride = other.FineWindowStride;
        MaxSemanticNewTokens = other.MaxSemanticNewTokens;
        UseKeyValueCache = other.UseKeyValueCache;
        TokenizerModelName = other.TokenizerModelName;
    }

    /// <summary>Semantic transformer configuration.</summary>
    public BarkStageOptions Semantic { get; set; }

    /// <summary>Coarse-acoustic transformer configuration.</summary>
    public BarkStageOptions Coarse { get; set; }

    /// <summary>Fine-acoustic transformer configuration.</summary>
    public BarkStageOptions Fine { get; set; }

    /// <summary>Number of actual semantic audio tokens (special tokens are outside this range).</summary>
    public int SemanticVocabularySize { get; set; } = 10_000;

    /// <summary>Offset applied to BERT text token ids in the semantic model's combined vocabulary.</summary>
    public int TextEncodingOffset { get; set; } = 10_048;

    /// <summary>Semantic padding token.</summary>
    public int SemanticPadTokenId { get; set; } = 10_000;

    /// <summary>Token that asks the semantic model to begin generation.</summary>
    public int SemanticInferenceTokenId { get; set; } = 129_599;

    /// <summary>Padding token used for the fixed 256-token BERT text section.</summary>
    public int TextPadTokenId { get; set; } = 129_595;

    /// <summary>Semantic padding token in the coarse model vocabulary.</summary>
    public int CoarseSemanticPadTokenId { get; set; } = 12_048;

    /// <summary>Token that asks the coarse model to begin generation.</summary>
    public int CoarseInferenceTokenId { get; set; } = 12_050;

    /// <summary>Semantic token rate from the HuBERT tokenizer.</summary>
    public double SemanticRateHz { get; set; } = 49.9;

    /// <summary>Coarse EnCodec frame rate.</summary>
    public double CoarseRateHz { get; set; } = 75.0;

    /// <summary>Number of autoregressively generated coarse codebooks.</summary>
    public int NumberOfCoarseCodebooks { get; set; } = 2;

    /// <summary>Fine codebooks supplied rather than predicted.</summary>
    public int NumberOfFineCodebooksGiven { get; set; } = 1;

    /// <summary>Maximum semantic history tokens included in a prompt.</summary>
    public int SemanticHistoryLength { get; set; } = 256;

    /// <summary>Maximum flattened coarse history tokens.</summary>
    public int CoarseHistoryLength { get; set; } = 630;

    /// <summary>Fixed semantic section prepended to each coarse generation window.</summary>
    public int CoarseSemanticContextLength { get; set; } = 256;

    /// <summary>Coarse tokens retained when advancing the sliding context.</summary>
    public int CoarseSlidingWindowLength { get; set; } = 60;

    /// <summary>Fine transformer window length.</summary>
    public int FineWindowLength { get; set; } = 1_024;

    /// <summary>Fine transformer window stride.</summary>
    public int FineWindowStride { get; set; } = 512;

    /// <summary>Maximum semantic tokens generated for one request.</summary>
    public int MaxSemanticNewTokens { get; set; } = 768;

    /// <summary>Use incremental KV-cached decoding for causal stages.</summary>
    public bool UseKeyValueCache { get; set; } = true;

    /// <summary>Hugging Face tokenizer paired with the released multilingual Bark checkpoints.</summary>
    public string TokenizerModelName { get; set; } = "bert-base-multilingual-cased";

    /// <summary>
    /// Returns a topology-faithful tiny configuration for unit tests and examples.
    /// </summary>
    public static BarkOptions TinyForTests(int seedVocabulary = 33)
    {
        if (seedVocabulary < 16)
            throw new ArgumentOutOfRangeException(nameof(seedVocabulary), "Tiny Bark vocabulary must be at least 16.");

        var options = new BarkOptions
        {
            SampleRate = 24_000,
            NumCodebooks = 8,
            CodebookSize = seedVocabulary,
            CodecFrameRate = 75,
            MaxTextLength = 4,
            MaxCodecFrames = 8,
            SemanticVocabularySize = seedVocabulary - 5,
            TextEncodingOffset = seedVocabulary,
            SemanticPadTokenId = seedVocabulary - 5,
            SemanticInferenceTokenId = seedVocabulary * 2 - 1,
            TextPadTokenId = seedVocabulary * 2 - 5,
            CoarseSemanticPadTokenId = seedVocabulary * 3,
            CoarseInferenceTokenId = seedVocabulary * 3 + 1,
            SemanticHistoryLength = 2,
            CoarseHistoryLength = 4,
            CoarseSemanticContextLength = 1,
            CoarseSlidingWindowLength = 2,
            FineWindowLength = 8,
            FineWindowStride = 4,
            MaxSemanticNewTokens = 4,
        };

        options.Semantic = TinyStage(seedVocabulary * 2, seedVocabulary, causal: true);
        options.Coarse = TinyStage(seedVocabulary * 4, seedVocabulary * 4, causal: true);
        options.Fine = TinyStage(seedVocabulary + 4, seedVocabulary + 4, causal: false);
        return options;
    }

    private static BarkStageOptions TinyStage(int inputVocabulary, int outputVocabulary, bool causal) => new()
    {
        InputVocabularySize = inputVocabulary,
        OutputVocabularySize = outputVocabulary,
        HiddenSize = 16,
        NumberOfLayers = 2,
        NumberOfHeads = 2,
        BlockSize = 8,
        FeedForwardSize = 64,
        IsCausal = causal,
    };

    /// <summary>Validates the complete connected Bark contract.</summary>
    public void Validate()
    {
        Guard.NotNull(Semantic);
        Guard.NotNull(Coarse);
        Guard.NotNull(Fine);
        Semantic.Validate(nameof(Semantic));
        Coarse.Validate(nameof(Coarse));
        Fine.Validate(nameof(Fine));

        if (!Semantic.IsCausal)
            throw new ArgumentException("Bark's semantic transformer must be causal.", nameof(Semantic));
        if (!Coarse.IsCausal)
            throw new ArgumentException("Bark's coarse transformer must be causal.", nameof(Coarse));
        if (Fine.IsCausal)
            throw new ArgumentException("Bark's fine transformer must be bidirectional.", nameof(Fine));
        if (SampleRate != 24_000)
            throw new ArgumentException("The released Bark/EnCodec checkpoints require a 24,000 Hz sample rate.", nameof(SampleRate));
        if (NumCodebooks <= 0 || NumberOfCoarseCodebooks <= 0 || NumberOfCoarseCodebooks > NumCodebooks)
            throw new ArgumentException("Bark requires at least one coarse codebook and no more coarse codebooks than total codebooks.");
        if (NumberOfFineCodebooksGiven <= 0 || NumberOfFineCodebooksGiven >= NumCodebooks)
            throw new ArgumentException("NumberOfFineCodebooksGiven must be between one and NumCodebooks - 1.");
        if (NumberOfFineCodebooksGiven > NumberOfCoarseCodebooks)
            throw new ArgumentException(
                "NumberOfFineCodebooksGiven cannot exceed the number of codebooks supplied by the coarse stage.");
        if (CodebookSize <= 0 || SemanticVocabularySize <= 0)
            throw new ArgumentException("Codebook and semantic vocabulary sizes must be positive.");
        if (SemanticPadTokenId < SemanticVocabularySize || SemanticPadTokenId >= Semantic.OutputVocabularySize)
            throw new ArgumentException("SemanticPadTokenId must be a special output token outside the semantic audio vocabulary.");
        if (SemanticInferenceTokenId < 0 || SemanticInferenceTokenId >= Semantic.InputVocabularySize)
            throw new ArgumentException("SemanticInferenceTokenId is outside the semantic input vocabulary.");
        if (TextPadTokenId < 0 || TextPadTokenId >= Semantic.InputVocabularySize)
            throw new ArgumentException("TextPadTokenId is outside the semantic input vocabulary.");
        if (CoarseInferenceTokenId < 0 || CoarseInferenceTokenId >= Coarse.InputVocabularySize)
            throw new ArgumentException("CoarseInferenceTokenId is outside the coarse input vocabulary.");
        if (Fine.InputVocabularySize < CodebookSize || Fine.OutputVocabularySize < CodebookSize)
            throw new ArgumentException("Fine transformer vocabularies must contain the complete EnCodec codebook.");
        if (FineWindowLength <= 0 || FineWindowLength > Fine.BlockSize)
            throw new ArgumentException("FineWindowLength must be positive and no larger than Fine.BlockSize.");
        if (FineWindowStride <= 0 || FineWindowStride > FineWindowLength)
            throw new ArgumentException("FineWindowStride must be positive and no larger than FineWindowLength.");
        if (SemanticHistoryLength < 0 || CoarseHistoryLength < 0 || CoarseSemanticContextLength <= 0
            || CoarseSlidingWindowLength <= 0)
            throw new ArgumentException("Bark history lengths cannot be negative and the coarse sliding window must be positive.");
        if (CoarseSemanticContextLength + 1 + CoarseHistoryLength + CoarseSlidingWindowLength > Coarse.BlockSize)
            throw new ArgumentException(
                "CoarseSemanticContextLength + inference token + CoarseHistoryLength + CoarseSlidingWindowLength "
                + "must fit Coarse.BlockSize.");
        if (MaxSemanticNewTokens <= 0 || MaxTextLength + SemanticHistoryLength + 1 > Semantic.BlockSize)
            throw new ArgumentException(
                "The semantic text, history, and inference marker must fit Semantic.BlockSize; generation advances through a sliding KV context.");
    }
}
