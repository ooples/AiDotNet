using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// ABINet's Bidirectional Cloze Network (Fang et al., "Read Like Humans", CVPR 2021, Sec. 3.3; reference
/// <c>BCNLanguage</c>): a stack of cross-attention-only transformer decoder layers.
/// </summary>
/// <remarks>
/// <para>
/// The input is the projected character-probability sequence. A sinusoidal positional encoding is added to
/// it, and the result is the attention MEMORY (keys and values). The queries are the positional encodings
/// alone, so what a position predicts is built only from its context. The location mask blocks every
/// position from attending to itself (paper Eq. 3), which makes the network a cloze in both directions.
/// </para>
/// <para>
/// Each sub-layer is the reference <c>TransformerDecoderLayer(self_attn=False)</c>, post-norm:
/// <c>tgt = LN(tgt + MHA(tgt, memory, memory, mask))</c>, then <c>tgt = LN(tgt + FFN(tgt))</c> with a ReLU
/// feed-forward. Sub-layer <c>k + 1</c> queries with sub-layer <c>k</c>'s output, and every sub-layer
/// attends to the same memory.
/// </para>
/// <para><b>For Beginners:</b> This is the language model that fixes spelling. Each character position looks
/// at every other character, but never at itself, and guesses what belongs there, like filling in a blank.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 2, 16, 2")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output,
    Note = "Each position is re-predicted from its context: every axis survives at its input size.")]
[AutoParameters]
public partial class BidirectionalClozeNetworkLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDim;
    private readonly int _numHeads;
    private readonly int _feedForwardDim;
    private readonly int _numLayers;

    [SubLayerInput("_modelDim")]
    private readonly List<DenseLayer<T>> _queries = new();
    [SubLayerInput("_modelDim")]
    private readonly List<DenseLayer<T>> _keys = new();
    [SubLayerInput("_modelDim")]
    private readonly List<DenseLayer<T>> _values = new();
    [SubLayerInput("_modelDim")]
    private readonly List<DenseLayer<T>> _outputs = new();
    [SubLayerInput("_modelDim")]
    private readonly List<LayerNormalizationLayer<T>> _attentionNorms = new();
    [SubLayerInput("_modelDim")]
    private readonly List<DenseLayer<T>> _feedForward1 = new();
    [SubLayerInput("_feedForwardDim")]
    private readonly List<DenseLayer<T>> _feedForward2 = new();
    [SubLayerInput("_modelDim")]
    private readonly List<LayerNormalizationLayer<T>> _feedForwardNorms = new();

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Creates the network. The ABINet paper uses 512 wide, 8 heads, a 2048 feed-forward and 4 layers.</summary>
    public BidirectionalClozeNetworkLayer(
        [LayerState] int modelDim,
        [LayerState] int numHeads = 8,
        [LayerState] int feedForwardDim = 2048,
        [LayerState] int numLayers = 4)
        : base(new[] { -1, modelDim }, new[] { -1, modelDim })
    {
        if (modelDim <= 0) throw new ArgumentOutOfRangeException(nameof(modelDim));
        if (numHeads <= 0 || modelDim % numHeads != 0)
            throw new ArgumentException($"modelDim ({modelDim}) must be a positive multiple of numHeads ({numHeads}).", nameof(numHeads));
        if (feedForwardDim <= 0) throw new ArgumentOutOfRangeException(nameof(feedForwardDim));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));

        _modelDim = modelDim;
        _numHeads = numHeads;
        _feedForwardDim = feedForwardDim;
        _numLayers = numLayers;
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        for (int i = 0; i < numLayers; i++)
        {
            _queries.Add(new DenseLayer<T>(modelDim, identity));
            _keys.Add(new DenseLayer<T>(modelDim, identity));
            _values.Add(new DenseLayer<T>(modelDim, identity));
            _outputs.Add(new DenseLayer<T>(modelDim, identity));
            _attentionNorms.Add(new LayerNormalizationLayer<T>());
            _feedForward1.Add(new DenseLayer<T>(feedForwardDim, (IActivationFunction<T>)new ReLUActivation<T>()));
            _feedForward2.Add(new DenseLayer<T>(modelDim, identity));
            _feedForwardNorms.Add(new LayerNormalizationLayer<T>());
        }

        foreach (var layer in AllSubLayers()) RegisterSubLayer(layer);
    }

    private IEnumerable<LayerBase<T>> AllSubLayers()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            yield return _queries[i];
            yield return _keys[i];
            yield return _values[i];
            yield return _outputs[i];
            yield return _attentionNorms[i];
            yield return _feedForward1[i];
            yield return _feedForward2[i];
            yield return _feedForwardNorms[i];
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Shape.Length == 2;
        if (unbatched)
            input = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1]]);
        if (input.Shape.Length != 3)
            throw new ArgumentException($"BidirectionalClozeNetworkLayer expects rank-2 [S, D] or rank-3 [B, S, D], got rank {input.Shape.Length}.", nameof(input));

        int batch = input.Shape[0], length = input.Shape[1], dim = input.Shape[2];
        if (dim != _modelDim)
            throw new ArgumentException($"BidirectionalClozeNetworkLayer was configured for modelDim={_modelDim} but got D={dim}.", nameof(input));
        int heads = _numHeads, headDim = dim / heads;

        // PositionalEncoding: pe[p, 2i] = sin(p / 10000^(2i/d)), pe[p, 2i+1] = cos(p / 10000^(2i/d)).
        var pe = new Tensor<T>(new[] { 1, length, dim });
        for (int p = 0; p < length; p++)
            for (int i = 0; i < dim; i += 2)
            {
                double angle = p / Math.Pow(10000, (double)i / dim);
                pe[0, p, i] = NumOps.FromDouble(Math.Sin(angle));
                if (i + 1 < dim) pe[0, p, i + 1] = NumOps.FromDouble(Math.Cos(angle));
            }
        var position = Engine.TensorBroadcastTo(pe, new[] { batch, length, dim });
        var memory = Engine.TensorAdd(input, position);
        var target = position;

        // Location mask: a position never attends to itself (additive -1e9 on the diagonal).
        var mask = new Tensor<T>(new[] { batch * heads, length, length });
        T blocked = NumOps.FromDouble(-1e9);
        for (int b = 0; b < batch * heads; b++)
            for (int i = 0; i < length; i++)
                mask[b, i, i] = blocked;

        Tensor<T> Split(Tensor<T> x) => Engine.Reshape(
            Engine.TensorPermute(Engine.Reshape(x, new[] { batch, length, heads, headDim }), new[] { 0, 2, 1, 3 }),
            new[] { batch * heads, length, headDim });

        for (int l = 0; l < _numLayers; l++)
        {
            var q = Split(_queries[l].Forward(target));
            var k = Split(_keys[l].Forward(memory));
            var v = Split(_values[l].Forward(memory));
            var scores = Engine.TensorDivideScalar(
                Engine.TensorBatchMatMul<T>(q, Engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(Math.Sqrt(headDim)));
            var weights = Engine.TensorSoftmax(Engine.TensorAdd(scores, mask), axis: 2);
            var attended = Engine.Reshape(
                Engine.TensorPermute(Engine.Reshape(Engine.TensorBatchMatMul<T>(weights, v), new[] { batch, heads, length, headDim }), new[] { 0, 2, 1, 3 }),
                new[] { batch, length, dim });
            target = _attentionNorms[l].Forward(Engine.TensorAdd(target, _outputs[l].Forward(attended)));
            target = _feedForwardNorms[l].Forward(Engine.TensorAdd(target, _feedForward2[l].Forward(_feedForward1[l].Forward(target))));
        }

        return unbatched ? Engine.Reshape(target, [length, dim]) : target;
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDim"] = _modelDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["FeedForwardDim"] = _feedForwardDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumLayers"] = _numLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var layer in AllSubLayers()) layer.ResetState();
    }
}
