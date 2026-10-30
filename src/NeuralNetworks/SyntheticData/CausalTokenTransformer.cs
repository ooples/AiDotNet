using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// The decoder-only causal transformer the token-sequence tabular generators share (REaLTabFormer, TabLLM-Gen): learned
/// token and position embeddings, then pre-LayerNorm blocks of multi-head causal self-attention and a feed-forward pair,
/// each with a residual connection, then a final LayerNorm (GPT-2, Radford et al. 2019). Every op is tape-tracked, so
/// a training step differentiates the whole stack.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal sealed class CausalTokenTransformer<T>
{
    private readonly EmbeddingLayer<T> _tokenEmbedding;
    private readonly EmbeddingLayer<T> _positionEmbedding;
    private readonly List<LayerNormalizationLayer<T>> _attentionNorms = new();
    private readonly List<FullyConnectedLayer<T>> _queries = new();
    private readonly List<FullyConnectedLayer<T>> _keys = new();
    private readonly List<FullyConnectedLayer<T>> _values = new();
    private readonly List<FullyConnectedLayer<T>> _outputs = new();
    private readonly List<LayerNormalizationLayer<T>> _feedForwardNorms = new();
    private readonly List<(ILayer<T> Up, ILayer<T> Down)> _feedForward;
    private readonly List<DropoutLayer<T>> _dropouts = new();
    private readonly LayerNormalizationLayer<T> _finalNorm;
    private readonly List<ILayer<T>> _allLayers = new();
    private readonly int _embeddingDim;
    private readonly int _numHeads;

    /// <summary>
    /// Builds the stack.
    /// </summary>
    /// <param name="vocabularySize">The number of token ids.</param>
    /// <param name="maxSequenceLength">The longest sequence the position embedding covers.</param>
    /// <param name="embeddingDim">The model width.</param>
    /// <param name="numHeads">The requested attention heads; the largest divisor of <paramref name="embeddingDim"/> not
    /// above it is used.</param>
    /// <param name="dropoutRate">The residual dropout rate.</param>
    /// <param name="feedForward">Each block's feed-forward pair (width-expanding layer with its activation, then the
    /// projection back to <paramref name="embeddingDim"/>), one per block.</param>
    public CausalTokenTransformer(int vocabularySize, int maxSequenceLength, int embeddingDim, int numHeads,
        double dropoutRate, IReadOnlyList<(ILayer<T> Up, ILayer<T> Down)> feedForward)
    {
        if (vocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (maxSequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));

        _embeddingDim = embeddingDim;
        _numHeads = Math.Max(1, Math.Min(numHeads, embeddingDim));
        while (embeddingDim % _numHeads != 0) _numHeads--;

        _tokenEmbedding = new EmbeddingLayer<T>(vocabularySize, embeddingDim);
        _positionEmbedding = new EmbeddingLayer<T>(maxSequenceLength, embeddingDim);
        _allLayers.Add(_tokenEmbedding);
        _allLayers.Add(_positionEmbedding);

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _feedForward = new List<(ILayer<T>, ILayer<T>)>(feedForward);
        foreach (var (up, down) in _feedForward)
        {
            var attentionNorm = new LayerNormalizationLayer<T>(embeddingDim);
            var q = new FullyConnectedLayer<T>(embeddingDim, identity);
            var k = new FullyConnectedLayer<T>(embeddingDim, identity);
            var v = new FullyConnectedLayer<T>(embeddingDim, identity);
            var o = new FullyConnectedLayer<T>(embeddingDim, identity);
            var feedForwardNorm = new LayerNormalizationLayer<T>(embeddingDim);
            _attentionNorms.Add(attentionNorm);
            _queries.Add(q);
            _keys.Add(k);
            _values.Add(v);
            _outputs.Add(o);
            _feedForwardNorms.Add(feedForwardNorm);
            _dropouts.Add(new DropoutLayer<T>(dropoutRate));
            _allLayers.AddRange(new ILayer<T>[] { attentionNorm, q, k, v, o, feedForwardNorm, up, down });
        }
        _finalNorm = new LayerNormalizationLayer<T>(embeddingDim);
        _allLayers.Add(_finalNorm);
    }

    /// <summary>Every layer of the stack, the feed-forward pairs included: the parameter group a training step moves.</summary>
    public IReadOnlyList<ILayer<T>> Layers => _allLayers;

    /// <summary>
    /// The hidden states [batch, sequence, embeddingDim] for <paramref name="tokens"/> [batch, sequence] (token ids as
    /// values); position p attends to positions 0..p only.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> tokens, bool training)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = tokens.Shape[0];
        int seq = tokens.Shape[1];
        int d = _embeddingDim;
        int headDim = d / _numHeads;

        var positions = new Tensor<T>([batch, seq]);
        for (int b = 0; b < batch; b++)
            for (int p = 0; p < seq; p++)
                positions[b, p] = numOps.FromDouble(p);

        var x = engine.Reshape(
            engine.TensorAdd(_tokenEmbedding.Forward(tokens), _positionEmbedding.Forward(positions)), [batch * seq, d]);

        Tensor<T> Heads(Tensor<T> flat)
            => engine.TensorPermute(engine.Reshape(flat, [batch, seq, _numHeads, headDim]), [0, 2, 1, 3]);

        for (int l = 0; l < _feedForward.Count; l++)
        {
            var h = _attentionNorms[l].Forward(x);
            var attended = engine.ScaledDotProductAttention(
                Heads(_queries[l].Forward(h)), Heads(_keys[l].Forward(h)), Heads(_values[l].Forward(h)),
                isCausal: true);
            var merged = engine.Reshape(engine.TensorPermute(attended, [0, 2, 1, 3]).Contiguous(), [batch * seq, d]);
            x = engine.TensorAdd(x, _outputs[l].Forward(merged));

            var f = _feedForward[l].Down.Forward(_feedForward[l].Up.Forward(_feedForwardNorms[l].Forward(x)));
            if (training)
            {
                _dropouts[l].SetTrainingMode(true);
                f = _dropouts[l].Forward(f);
            }
            x = engine.TensorAdd(x, f);
        }

        return engine.Reshape(_finalNorm.Forward(x), [batch, seq, d]);
    }

    /// <summary>
    /// The default feed-forward pairs: <paramref name="count"/> blocks of FullyConnected(<paramref name="hiddenDim"/>,
    /// <paramref name="activation"/>) then FullyConnected(<paramref name="embeddingDim"/>).
    /// </summary>
    public static List<ILayer<T>> DefaultFeedForwardLayers(int count, int hiddenDim, int embeddingDim,
        IActivationFunction<T> activation)
    {
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var layers = new List<ILayer<T>>(count * 2);
        for (int l = 0; l < count; l++)
        {
            layers.Add(new FullyConnectedLayer<T>(hiddenDim, activation));
            layers.Add(new FullyConnectedLayer<T>(embeddingDim, identity));
        }
        return layers;
    }

    /// <summary>
    /// Pairs <paramref name="layers"/> two at a time into <paramref name="count"/> feed-forward pairs when they hold
    /// exactly that many; otherwise returns fresh default pairs.
    /// </summary>
    public static List<(ILayer<T> Up, ILayer<T> Down)> PairFeedForward(IReadOnlyList<ILayer<T>> layers, int count,
        int hiddenDim, int embeddingDim, IActivationFunction<T> activation)
    {
        var source = layers.Count == count * 2
            ? layers
            : DefaultFeedForwardLayers(count, hiddenDim, embeddingDim, activation);
        var pairs = new List<(ILayer<T>, ILayer<T>)>(count);
        for (int l = 0; l < count; l++) pairs.Add((source[2 * l], source[2 * l + 1]));
        return pairs;
    }
}
