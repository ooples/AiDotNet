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

        // The generic tensor API carries IDs in T. The generated input-domain contract performs the
        // authoritative integer/range validation before this method. Derive one stable bucket ID for
        // each valid word ID and pack both feature groups for a single tape-tracked shared lookup.
        if (input.Rank is not (1 or 2))
            throw new ArgumentException(
                $"FastText feature bags require rank 1 or 2 input; got rank {input.Rank}.",
                nameof(input));

        int featureCount = input.Shape[input.Rank - 1];
        int[] packedShape = input.Rank == 1
            ? [checked(featureCount * 2)]
            : [input.Shape[0], checked(featureCount * 2)];
        var packedIds = new Tensor<T>(packedShape);
        for (int i = 0; i < input.Length; i++)
        {
            int tokenId = Convert.ToInt32(NumOps.ToDouble(input[i]));
            uint mixed = unchecked((uint)tokenId * 16777619u + 2166136261u);
            int subwordId = checked(_vocabularySize + (int)(mixed % (uint)_bucketSize));
            int rowOffset = input.Rank == 1 ? 0 : (i / featureCount) * featureCount * 2;
            int featureOffset = i % featureCount;
            packedIds.SetFlat(rowOffset + featureOffset, input.GetFlat(i));
            packedIds.SetFlat(
                rowOffset + featureCount + featureOffset,
                NumOps.FromDouble(subwordId));
        }

        // A single lookup over the packed bag produces one dense embedding gradient rather than
        // evaluating the same enormous parameter tensor twice. This matches fastText's input-matrix
        // operation directly: all word and n-gram feature IDs are looked up together, then averaged.
        var combined = _featureEmbedding.Forward(packedIds);
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
