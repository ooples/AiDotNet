using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements BERT's parallel word, learned-position, and token-type embedding front-end.
/// </summary>
/// <remarks>
/// The three inputs are index lookups over independent tables. Their vectors are added, normalized,
/// and dropped out. This is intentionally one composite layer: placing the lookup tables sequentially
/// would feed continuous hidden states into later index lookups and would not represent BERT.
/// </remarks>
[LayerCategory(LayerCategory.Embedding)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true,
    TestInputShape = "1, 4", TestConstructorArgs = "100, 16, 32")]
[TensorPort("input_ids", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Role = TensorPortRole.TokenIds, MaxExclusiveMember = "_vocabularySize")]
[TensorPort("token_type_ids", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Role = TensorPortRole.TokenTypeIds, Required = false, MaxExclusiveMember = "_tokenTypeVocabularySize")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous,
    Role = TensorPortRole.Features)]
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class BertEmbeddingLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _vocabularySize;
    private readonly int _hiddenSize;
    private readonly int _maxSequenceLength;
    private readonly int _tokenTypeVocabularySize;
    private readonly double _dropoutProbability;
    private readonly double _layerNormEpsilon;

    // TrainableParameterGenerator discovers these fields and generates recursive sub-layer
    // registration/parameter plumbing. The composite owns no manual parameter-vector overrides.
    private readonly EmbeddingLayer<T> _wordEmbedding;
    private readonly EmbeddingLayer<T> _positionEmbedding;
    private readonly EmbeddingLayer<T> _tokenTypeEmbedding;
    private readonly LayerNormalizationLayer<T> _normalization;
    private readonly DropoutLayer<T> _dropout;

    /// <summary>
    /// Creates a canonical BERT embedding block.
    /// </summary>
    public BertEmbeddingLayer(
        [LayerState] int vocabularySize,
        [LayerState] int hiddenSize,
        [LayerState] int maxSequenceLength,
        [LayerState] int tokenTypeVocabularySize = 2,
        [LayerState] double dropoutProbability = 0.1,
        [LayerState] double layerNormEpsilon = 1e-12)
        : base([1], [hiddenSize])
    {
        if (vocabularySize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (maxSequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
        if (tokenTypeVocabularySize <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokenTypeVocabularySize));

        _vocabularySize = vocabularySize;
        _hiddenSize = hiddenSize;
        _maxSequenceLength = maxSequenceLength;
        _tokenTypeVocabularySize = tokenTypeVocabularySize;
        _dropoutProbability = dropoutProbability;
        _layerNormEpsilon = layerNormEpsilon;
        _wordEmbedding = new EmbeddingLayer<T>(vocabularySize, hiddenSize);
        _positionEmbedding = new EmbeddingLayer<T>(maxSequenceLength, hiddenSize);
        _tokenTypeEmbedding = new EmbeddingLayer<T>(tokenTypeVocabularySize, hiddenSize);
        _normalization = new LayerNormalizationLayer<T>(hiddenSize, layerNormEpsilon);
        _dropout = new DropoutLayer<T>(dropoutProbability);
    }

    /// <summary>Word lookup table.</summary>
    public EmbeddingLayer<T> WordEmbedding => _wordEmbedding;

    /// <summary>Learned absolute-position lookup table.</summary>
    public EmbeddingLayer<T> PositionEmbedding => _positionEmbedding;

    /// <summary>Token-type (segment) lookup table.</summary>
    public EmbeddingLayer<T> TokenTypeEmbedding => _tokenTypeEmbedding;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank switch
    {
        1 =>
        [
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_hiddenSize)),
        ],
        2 =>
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_hiddenSize)),
        ],
        _ => null,
    };

    /// <summary>
    /// Looks up input tokens with generated position IDs and all-zero token-type IDs.
    /// </summary>
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => ForwardCore(input, null);

    /// <inheritdoc />
    protected override Tensor<T> ForwardTracedPorts(IReadOnlyDictionary<string, Tensor<T>> inputs)
    {
        if (!inputs.TryGetValue("input_ids", out var inputIds))
            throw new ArgumentException("Missing required input port 'input_ids'.", nameof(inputs));
        inputs.TryGetValue("token_type_ids", out var tokenTypeIds);
        return ForwardCore(inputIds, tokenTypeIds);
    }

    /// <summary>
    /// Positional multi-input form: <c>[inputIds]</c> or <c>[inputIds, tokenTypeIds]</c>.
    /// </summary>
    protected override Tensor<T> ForwardTracedMany(params Tensor<T>[] inputs)
    {
        if (inputs is null || inputs.Length is < 1 or > 2)
            throw new ArgumentException(
                $"BertEmbeddingLayer accepts input IDs and optional token-type IDs; got {inputs?.Length ?? 0} tensors.",
                nameof(inputs));

        return ForwardCore(inputs[0], inputs.Length == 2 ? inputs[1] : null);
    }

    private Tensor<T> ForwardCore(Tensor<T> inputIds, Tensor<T>? tokenTypeIds)
    {
        // This is the standard generated-composite initialization hook. It resolves any
        // deferred shape and, through the generated EnsureInitialized override, registers
        // every child with LayerBase's structural walkers before the children execute.
        EnsureInitializedFromInput(inputIds);

        int sequenceLength = GetSequenceLength(inputIds);
        if (sequenceLength > _maxSequenceLength)
        {
            throw new ArgumentException(
                $"Sequence length {sequenceLength} exceeds the configured maximum {_maxSequenceLength}.",
                nameof(inputIds));
        }

        if (tokenTypeIds is not null)
            ValidateMatchingShape(inputIds, tokenTypeIds);

        var positionIds = new Tensor<T>(inputIds._shape);
        for (int i = 0; i < positionIds.Length; i++)
            positionIds.SetFlat(i, NumOps.FromDouble(i % sequenceLength));

        var effectiveTokenTypeIds = tokenTypeIds ?? new Tensor<T>(inputIds._shape);

        var words = _wordEmbedding.Forward(inputIds);
        var positions = _positionEmbedding.Forward(positionIds);
        var tokenTypes = _tokenTypeEmbedding.Forward(effectiveTokenTypeIds);
        var combined = Engine.TensorAdd(Engine.TensorAdd(words, positions), tokenTypes);

        return _dropout.Forward(_normalization.Forward(combined));
    }

    private static int GetSequenceLength(Tensor<T> inputIds)
    {
        if (inputIds.Rank == 1)
            return inputIds.Shape[0];
        if (inputIds.Rank == 2)
            return inputIds.Shape[1];

        throw new ArgumentException(
            $"BertEmbeddingLayer expects [sequence] or [batch, sequence] token IDs; "
            + $"got [{string.Join(", ", inputIds.Shape.ToArray())}].",
            nameof(inputIds));
    }

    private static void ValidateMatchingShape(Tensor<T> inputIds, Tensor<T> tokenTypeIds)
    {
        if (inputIds.Rank != tokenTypeIds.Rank)
            throw new ArgumentException("Token-type IDs must have the same shape as input IDs.", nameof(tokenTypeIds));

        for (int i = 0; i < inputIds.Rank; i++)
        {
            if (inputIds.Shape[i] != tokenTypeIds.Shape[i])
                throw new ArgumentException("Token-type IDs must have the same shape as input IDs.", nameof(tokenTypeIds));
        }
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _wordEmbedding.ResetState();
        _positionEmbedding.ResetState();
        _tokenTypeEmbedding.ResetState();
        _normalization.ResetState();
        _dropout.ResetState();
    }
}
