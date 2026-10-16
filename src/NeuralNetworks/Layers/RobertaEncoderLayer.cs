using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The RoBERTa text encoder (Liu et al. 2019) up to its pooled output, as Hugging Face <c>RobertaModel</c> computes
/// <c>pooler_output</c>: token ids in, one vector per sequence out.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The computation, layer by layer:
/// <list type="bullet">
/// <item><b>Embeddings</b> (<see cref="BertEmbeddingLayer{T}"/> in RoBERTa mode): word + learned position +
/// token-type, LayerNorm, dropout. Positions are counted over the non-padding tokens starting at
/// <c>pad + 1</c>.</item>
/// <item><b>N post-LN encoder layers</b>: biased query/key/value projections, scaled dot-product attention over
/// <c>heads</c> heads with padding keys masked out, attention-probability dropout, an output projection, dropout,
/// then <c>LayerNorm(x + attention)</c>; an intermediate projection with the exact (erf) GELU, an output
/// projection, dropout, then <c>LayerNorm(h + ffn)</c>.</item>
/// <item><b>Pooler</b>: the first (<c>&lt;s&gt;</c>) token's hidden state through a dense layer with tanh.</item>
/// </list>
/// </para>
/// <para>
/// The attention mask is derived from the ids: a position is padding exactly when its id equals the padding
/// token id, which is what the RoBERTa tokenizer's <c>attention_mask</c> encodes for a padded batch (the padding
/// id never occurs inside text).
/// </para>
/// <para><b>For Beginners:</b> This reads a sentence as numbers (token ids), lets every word look at every other
/// word over several rounds, and then summarizes the whole sentence into a single vector taken from the special
/// start-of-sentence token.</para>
/// </remarks>
[LayerCategory(LayerCategory.Transformer)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, HasTrainingMode = true, Cost = ComputeCost.High,
    TestInputShape = "2, 5", TestConstructorArgs = "32, 8, 1, 2, 16, 16, 1, 0.0, 1e-5")]
[TensorPort("input_ids", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Role = TensorPortRole.TokenIds, MaxExclusiveMember = "_vocabularySize")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous,
    Role = TensorPortRole.Features)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class RobertaEncoderLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _vocabularySize;
    private readonly int _hidden;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _intermediate;
    private readonly int _maxPositions;
    private readonly int _paddingTokenId;
    private readonly double _dropoutRate;
    private readonly double _layerNormEpsilon;

    [SubLayerInput("1")]
    private readonly BertEmbeddingLayer<T> _embeddings;
    [SubLayerInput("_hidden")]
    private readonly List<DenseLayer<T>> _query = new();
    [SubLayerInput("_hidden")]
    private readonly List<DenseLayer<T>> _key = new();
    [SubLayerInput("_hidden")]
    private readonly List<DenseLayer<T>> _value = new();
    [SubLayerInput("_hidden")]
    private readonly List<DenseLayer<T>> _attentionOutput = new();
    [SubLayerInput("_hidden")]
    private readonly List<LayerNormalizationLayer<T>> _attentionNorm = new();
    [SubLayerInput("_hidden")]
    private readonly List<DenseLayer<T>> _intermediateDense = new();
    [SubLayerInput("_intermediate")]
    private readonly List<DenseLayer<T>> _outputDense = new();
    [SubLayerInput("_hidden")]
    private readonly List<LayerNormalizationLayer<T>> _outputNorm = new();
    [SubLayerInput("_hidden")]
    private readonly List<DropoutLayer<T>> _attentionProbabilityDropout = new();
    [SubLayerInput("_hidden")]
    private readonly List<DropoutLayer<T>> _attentionHiddenDropout = new();
    [SubLayerInput("_hidden")]
    private readonly List<DropoutLayer<T>> _outputHiddenDropout = new();
    [SubLayerInput("_hidden")]
    private readonly DenseLayer<T> _pooler;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates the encoder. RoBERTa-base: vocabulary 50265, hidden 768, 12 layers, 12 heads, intermediate 3072,
    /// 514 positions, padding id 1, dropout 0.1, LayerNorm epsilon 1e-5.
    /// </summary>
    /// <param name="vocabularySize">Rows of the word-embedding table.</param>
    /// <param name="hiddenSize">Model width.</param>
    /// <param name="numLayers">Number of encoder layers.</param>
    /// <param name="numHeads">Attention heads per layer; must divide <paramref name="hiddenSize"/>.</param>
    /// <param name="intermediateSize">Width of the feed-forward intermediate projection.</param>
    /// <param name="maxPositions">Rows of the position table (RoBERTa: max length + padding id + 1).</param>
    /// <param name="paddingTokenId">The padding token id; padding keys are masked and positions count around it.</param>
    /// <param name="dropoutRate">Hidden and attention-probability dropout.</param>
    /// <param name="layerNormEpsilon">Epsilon of every LayerNorm.</param>
    public RobertaEncoderLayer(
        [LayerState] int vocabularySize,
        [LayerState] int hiddenSize,
        [LayerState] int numLayers,
        [LayerState] int numHeads,
        [LayerState] int intermediateSize,
        [LayerState] int maxPositions,
        [LayerState] int paddingTokenId,
        [LayerState] double dropoutRate,
        [LayerState] double layerNormEpsilon)
        : base(new[] { -1 }, new[] { hiddenSize })
    {
        if (vocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (hiddenSize <= 0 || numHeads <= 0 || hiddenSize % numHeads != 0)
            throw new ArgumentException($"hiddenSize ({hiddenSize}) must be a positive multiple of numHeads ({numHeads}).", nameof(numHeads));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (intermediateSize <= 0) throw new ArgumentOutOfRangeException(nameof(intermediateSize));
        if (paddingTokenId < 0 || paddingTokenId >= vocabularySize)
            throw new ArgumentOutOfRangeException(nameof(paddingTokenId), "The padding id must be a token of the vocabulary.");
        if (maxPositions <= paddingTokenId + 1)
            throw new ArgumentOutOfRangeException(nameof(maxPositions), "RoBERTa positions start at paddingTokenId + 1.");
        if (dropoutRate < 0.0 || dropoutRate >= 1.0) throw new ArgumentOutOfRangeException(nameof(dropoutRate));
        if (layerNormEpsilon <= 0.0) throw new ArgumentOutOfRangeException(nameof(layerNormEpsilon));

        _vocabularySize = vocabularySize;
        _hidden = hiddenSize;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _intermediate = intermediateSize;
        _maxPositions = maxPositions;
        _paddingTokenId = paddingTokenId;
        _dropoutRate = dropoutRate;
        _layerNormEpsilon = layerNormEpsilon;

        // RoBERTa has a single token type (type_vocab_size = 1).
        _embeddings = new BertEmbeddingLayer<T>(vocabularySize, hiddenSize, maxPositions,
            tokenTypeVocabularySize: 1, dropoutProbability: dropoutRate, layerNormEpsilon: layerNormEpsilon,
            paddingTokenId: paddingTokenId);
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        for (int i = 0; i < numLayers; i++)
        {
            _query.Add(new DenseLayer<T>(hiddenSize, identity));
            _key.Add(new DenseLayer<T>(hiddenSize, identity));
            _value.Add(new DenseLayer<T>(hiddenSize, identity));
            _attentionOutput.Add(new DenseLayer<T>(hiddenSize, identity));
            _attentionNorm.Add(new LayerNormalizationLayer<T>(hiddenSize, layerNormEpsilon));
            // The exact GELU is applied after this projection (Hugging Face "gelu" is the erf form).
            _intermediateDense.Add(new DenseLayer<T>(intermediateSize, identity));
            _outputDense.Add(new DenseLayer<T>(hiddenSize, identity));
            _outputNorm.Add(new LayerNormalizationLayer<T>(hiddenSize, layerNormEpsilon));
            _attentionProbabilityDropout.Add(new DropoutLayer<T>(dropoutRate));
            _attentionHiddenDropout.Add(new DropoutLayer<T>(dropoutRate));
            _outputHiddenDropout.Add(new DropoutLayer<T>(dropoutRate));
        }
        _pooler = new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new TanhActivation<T>());
        foreach (var layer in SubLayers()) RegisterSubLayer(layer);
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        yield return _embeddings;
        for (int i = 0; i < _numLayers; i++)
        {
            yield return _query[i]; yield return _key[i]; yield return _value[i];
            yield return _attentionProbabilityDropout[i];
            yield return _attentionOutput[i]; yield return _attentionHiddenDropout[i]; yield return _attentionNorm[i];
            yield return _intermediateDense[i]; yield return _outputDense[i]; yield return _outputHiddenDropout[i];
            yield return _outputNorm[i];
        }
        yield return _pooler;
    }

    /// <summary>The embedding block (word, position, token type, LayerNorm).</summary>
    public BertEmbeddingLayer<T> Embeddings => _embeddings;

    /// <summary>The pooler's dense layer (tanh), applied to the first token.</summary>
    public DenseLayer<T> Pooler => _pooler;

    /// <summary>Number of encoder layers.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Model width.</summary>
    public int HiddenSize => _hidden;

    /// <summary>The padding token id.</summary>
    public int PaddingTokenId => _paddingTokenId;

    /// <summary>The sub-layers of encoder layer <paramref name="index"/>, in Hugging Face's naming:
    /// <c>attention.self.{query,key,value}</c>, <c>attention.output.{dense,LayerNorm}</c>,
    /// <c>intermediate.dense</c>, <c>output.{dense,LayerNorm}</c>.</summary>
    internal (DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> AttentionOutput,
        LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Intermediate, DenseLayer<T> Output,
        LayerNormalizationLayer<T> OutputNorm) EncoderLayer(int index) =>
        (_query[index], _key[index], _value[index], _attentionOutput[index], _attentionNorm[index],
         _intermediateDense[index], _outputDense[index], _outputNorm[index]);

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_hidden)),
        }
        : null;

    /// <inheritdoc/>
    /// <remarks>Input: token ids <c>[batch, length]</c> (or <c>[length]</c>, read as one sequence). Output: the
    /// pooled vector <c>[batch, hidden]</c>.</remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var ids = input.Rank == 1 ? Engine.Reshape(input, new[] { 1, input.Shape[0] }) : input;
        if (ids.Rank != 2)
            throw new ArgumentException(
                $"RobertaEncoderLayer expects token ids [batch, length]; got [{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));
        int batch = ids.Shape[0], length = ids.Shape[1], headDim = _hidden / _numHeads;

        var keyBias = PaddingBias(ids, batch, length);                                  // [B, 1, 1, L]
        var scoreShape = new[] { batch, _numHeads, length, length };
        var maskBias = Engine.TensorBroadcastTo(keyBias, scoreShape);
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(headDim));

        var x = Engine.Reshape(_embeddings.Forward(ids), new[] { batch * length, _hidden });
        for (int i = 0; i < _numLayers; i++)
        {
            Tensor<T> Heads(Tensor<T> projected) =>
                Engine.TensorPermute(Engine.Reshape(projected, new[] { batch, length, _numHeads, headDim }), new[] { 0, 2, 1, 3 });
            var q = Heads(_query[i].Forward(x));
            var k = Engine.TensorPermute(
                Engine.Reshape(_key[i].Forward(x), new[] { batch, length, _numHeads, headDim }), new[] { 0, 2, 3, 1 });
            var v = Heads(_value[i].Forward(x));
            var scores = Engine.TensorAdd(Engine.TensorMultiplyScalar(Engine.BatchMatMul(q, k), scale), maskBias);
            var probabilities = _attentionProbabilityDropout[i].Forward(Engine.Softmax(scores, axis: -1));
            var context = Engine.Reshape(
                Engine.TensorPermute(Engine.BatchMatMul(probabilities, v), new[] { 0, 2, 1, 3 }),
                new[] { batch * length, _hidden });
            var attended = _attentionHiddenDropout[i].Forward(_attentionOutput[i].Forward(context));
            var h = _attentionNorm[i].Forward(Engine.TensorAdd(attended, x));
            var inner = ActivationHelper.ExactGelu(_intermediateDense[i].Forward(h), Engine);
            var ffn = _outputHiddenDropout[i].Forward(_outputDense[i].Forward(inner));
            x = _outputNorm[i].Forward(Engine.TensorAdd(ffn, h));
        }

        var sequence = Engine.Reshape(x, new[] { batch, length, _hidden });
        var first = Engine.Reshape(Engine.TensorSlice(sequence, new[] { 0, 0, 0 }, new[] { batch, 1, _hidden }),
            new[] { batch, _hidden });
        return _pooler.Forward(first);
    }

    // Hugging Face adds the dtype's most negative value to masked keys; any value that drives exp() to exactly 0
    // after the row maximum is subtracted gives the same probabilities, and -1e9 does so in float and double.
    private Tensor<T> PaddingBias(Tensor<T> ids, int batch, int length)
    {
        var bias = new Tensor<T>(new[] { batch, 1, 1, length });
        T blocked = NumOps.FromDouble(-1e9);
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < length; s++)
                if ((int)Math.Round(NumOps.ToDouble(ids[b, s])) == _paddingTokenId) bias[b, 0, 0, s] = blocked;
        return bias;
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var invariant = System.Globalization.CultureInfo.InvariantCulture;
        metadata["VocabularySize"] = _vocabularySize.ToString(invariant);
        metadata["HiddenSize"] = _hidden.ToString(invariant);
        metadata["NumLayers"] = _numLayers.ToString(invariant);
        metadata["NumHeads"] = _numHeads.ToString(invariant);
        metadata["IntermediateSize"] = _intermediate.ToString(invariant);
        metadata["MaxPositions"] = _maxPositions.ToString(invariant);
        metadata["PaddingTokenId"] = _paddingTokenId.ToString(invariant);
        metadata["DropoutRate"] = _dropoutRate.ToString("R", invariant);
        metadata["LayerNormEpsilon"] = _layerNormEpsilon.ToString("R", invariant);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var layer in SubLayers()) layer.ResetState(); }
}
