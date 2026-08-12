using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Reusable FastText word/subword embedding front end. One shared input matrix contains word rows
/// followed by hashed subword rows, and the parallel lookups are averaged into a bag representation.
/// </summary>
[LayerCategory(LayerCategory.Embedding)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true,
    TestInputShape = "1, 4", TestConstructorArgs = "100, 256, 16")]
[TensorPort("token_ids", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Role = TensorPortRole.TokenIds, MaxExclusiveMember = "_vocabularySize")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous,
    Role = TensorPortRole.Features)]
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class FastTextEmbeddingLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _vocabularySize;
    private readonly int _bucketSize;
    private readonly int _embeddingDimension;
    private readonly EmbeddingLayer<T> _featureEmbedding;
    private readonly MeanLayer<T> _unbatchedMean;
    private readonly MeanLayer<T> _batchedMean;

    /// <summary>Creates a word-plus-hashed-subword embedding layer.</summary>
    public FastTextEmbeddingLayer(
        [LayerState] int vocabularySize,
        [LayerState] int bucketSize,
        [LayerState] int embeddingDimension)
        : base([1], [embeddingDimension])
    {
        if (vocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (bucketSize <= 0) throw new ArgumentOutOfRangeException(nameof(bucketSize));
        if (embeddingDimension <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDimension));

        _vocabularySize = vocabularySize;
        _bucketSize = bucketSize;
        _embeddingDimension = embeddingDimension;
        _featureEmbedding = new EmbeddingLayer<T>(
            checked(vocabularySize + bucketSize),
            embeddingDimension);
        _unbatchedMean = new MeanLayer<T>(axis: 0);
        _batchedMean = new MeanLayer<T>(axis: 1);
    }

    /// <summary>
    /// The paper-faithful shared word/subword input matrix, exposed for pretrained initialization
    /// and the string-token embedding API.
    /// </summary>
    public EmbeddingLayer<T> FeatureEmbedding => _featureEmbedding;

    /// <summary>Looks up already-packed word/subword feature IDs without applying bag averaging.</summary>
    public Tensor<T> LookupFeatures(Tensor<T> featureIds) => _featureEmbedding.Forward(featureIds);

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank switch
    {
        1 =>
        [
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_embeddingDimension)),
        ],
        2 =>
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_embeddingDimension)),
        ],
        _ => null,
    };

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);

        // The generic tensor API carries IDs in T. The primary lookup performs the authoritative
        // integer/range validation. Derive a stable bucket ID from each valid token ID, then run the
        // two trainable tables as parallel tape-tracked children.
        if (input.Rank is not (1 or 2))
            throw new ArgumentException(
                $"FastText feature bags require rank 1 or 2 input; got rank {input.Rank}.",
                nameof(input));

        var words = _featureEmbedding.Forward(input);
        var bucketIds = new Tensor<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
        {
            int tokenId = Convert.ToInt32(NumOps.ToDouble(input[i]));
            uint mixed = unchecked((uint)tokenId * 16777619u + 2166136261u);
            int subwordId = checked(_vocabularySize + (int)(mixed % (uint)_bucketSize));
            bucketIds.SetFlat(i, NumOps.FromDouble(subwordId));
        }

        var subwords = _featureEmbedding.Forward(bucketIds);
        var combined = Engine.TensorMultiplyScalar(
            Engine.TensorAdd(words, subwords),
            NumOps.FromDouble(0.5));
        return input.Rank == 1
            ? _unbatchedMean.Forward(combined)
            : _batchedMean.Forward(combined);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _featureEmbedding.ResetState();
        _unbatchedMean.ResetState();
        _batchedMean.ResetState();
    }
}
