using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements BERT's parallel word, learned-position, and token-type embedding front-end.
/// </summary>
/// <remarks>
/// The three inputs are index lookups over independent tables. Their vectors are added, normalized,
/// and dropped out. This is intentionally one composite layer: placing the lookup tables sequentially
/// would feed continuous hidden states into later index lookups and would not represent BERT.
/// <para>
/// RoBERTa (Liu et al. 2019; Hugging Face <c>RobertaEmbeddings</c>) uses the same block with one difference: the
/// position of a token is counted over the NON-padding tokens and offset past the padding index,
/// <c>position = cumsum(id != pad) * (id != pad) + pad</c>, so real tokens start at <c>pad + 1</c> and every
/// padding token reads the padding row. Pass <c>paddingTokenId</c> to get that numbering; leave it negative for
/// BERT's <c>0 .. L-1</c>.
/// </para>
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
    private readonly int _paddingTokenId;

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
        [LayerState] double layerNormEpsilon = 1e-12,
        [LayerState] int paddingTokenId = -1)
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
        // With a padding index, positions are numbered from paddingTokenId + 1, so a sequence of length L reads rows up
        // to L + paddingTokenId. Even one token needs paddingTokenId + 1 to be a row: anything higher rejects every input.
        if (paddingTokenId > maxSequenceLength - 2)
            throw new ArgumentOutOfRangeException(nameof(paddingTokenId),
                $"Positions start after the padding index, so it must be at most maxSequenceLength - 2 ({maxSequenceLength - 2}); got {paddingTokenId}.");

        _vocabularySize = vocabularySize;
        _hiddenSize = hiddenSize;
        _maxSequenceLength = maxSequenceLength;
        _tokenTypeVocabularySize = tokenTypeVocabularySize;
        _dropoutProbability = dropoutProbability;
        _layerNormEpsilon = layerNormEpsilon;
        _paddingTokenId = paddingTokenId;
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

    /// <summary>The LayerNorm applied to the summed embeddings.</summary>
    public LayerNormalizationLayer<T> Normalization => _normalization;

    /// <summary>The padding token id that RoBERTa-style positions count around, or a negative value for BERT
    /// positions <c>0 .. L-1</c>.</summary>
    public int PaddingTokenId => _paddingTokenId;

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
        // RoBERTa positions run from pad + 1, so the table needs pad + 1 more rows than the sequence.
        int positionsNeeded = _paddingTokenId >= 0 ? sequenceLength + _paddingTokenId + 1 : sequenceLength;
        if (positionsNeeded > _maxSequenceLength)
        {
            throw new ArgumentException(
                $"Sequence length {sequenceLength} needs {positionsNeeded} position rows; the table has {_maxSequenceLength}.",
                nameof(inputIds));
        }

        if (tokenTypeIds is not null)
            ValidateMatchingShape(inputIds, tokenTypeIds);

        Tensor<T> positionIds;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            // Position ids are constant data, built off the tape as the old fresh tensor was.
            // An empty sequence has no rows; the padding-aware path divided by the length, as before.
            int rows = _paddingTokenId < 0 && sequenceLength == 0 ? 0 : inputIds.Length / sequenceLength;
            if (rows == 0)
            {
                positionIds = new Tensor<T>(inputIds._shape);
            }
            else if (_paddingTokenId < 0)
            {
                // 0..sequenceLength-1 in every row.
                var row = Engine.Reshape(Engine.TensorArange<T>(0, sequenceLength), new[] { 1, sequenceLength });
                positionIds = Engine.Reshape(Engine.TensorTile(row, new[] { rows, 1 }), inputIds._shape);
            }
            else
            {
                // create_position_ids_from_input_ids: count the non-padding tokens of each row.
                // A token is padding when its id rounds (half to even) to the padding id; NaN never does.
                var ids = Engine.Reshape(inputIds, new[] { rows, sequenceLength }).Contiguous().Cast<double>();
                double pad = _paddingTokenId;
                var isPadding = Engine.TensorEquals(Engine.TensorRound(ids), pad);
                var ones = new Tensor<double>(new[] { rows, sequenceLength });
                ones.Fill(1.0);
                var counted = Engine.TensorCumSum(Engine.TensorSubtract(ones, isPadding), 1);
                var padIds = new Tensor<double>(new[] { rows, sequenceLength });
                padIds.Fill(pad);
                positionIds = Engine.Reshape(
                    Engine.TensorWhere(isPadding, padIds, Engine.TensorAddScalar(counted, pad)), inputIds._shape).Cast<T>();
            }
        }

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
