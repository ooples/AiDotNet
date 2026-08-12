using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Reusable FastText word/subword embedding front end. The lookup tables are parallel children whose
/// continuous vectors are combined; neither lookup is ever fed the other lookup's output.
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
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class FastTextEmbeddingLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _vocabularySize;
    private readonly int _bucketSize;
    private readonly int _embeddingDimension;
    private readonly EmbeddingLayer<T> _wordEmbedding;
    private readonly EmbeddingLayer<T> _subwordEmbedding;

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
        _wordEmbedding = new EmbeddingLayer<T>(vocabularySize, embeddingDimension);
        _subwordEmbedding = new EmbeddingLayer<T>(bucketSize, embeddingDimension);
    }

    /// <summary>Whole-word lookup table, exposed for pretrained initialization.</summary>
    public EmbeddingLayer<T> WordEmbedding => _wordEmbedding;

    /// <summary>Hashed character n-gram lookup table, exposed for pretrained initialization.</summary>
    public EmbeddingLayer<T> SubwordEmbedding => _subwordEmbedding;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank switch
    {
        1 =>
        [
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_embeddingDimension)),
        ],
        2 =>
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
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
        var words = _wordEmbedding.Forward(input);
        var bucketIds = new Tensor<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
        {
            int tokenId = Convert.ToInt32(NumOps.ToDouble(input[i]));
            uint mixed = unchecked((uint)tokenId * 16777619u + 2166136261u);
            bucketIds.SetFlat(i, NumOps.FromDouble(mixed % (uint)_bucketSize));
        }

        var subwords = _subwordEmbedding.Forward(bucketIds);
        return Engine.TensorAdd(words, subwords);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _wordEmbedding.ResetState();
        _subwordEmbedding.ResetState();
    }
}
