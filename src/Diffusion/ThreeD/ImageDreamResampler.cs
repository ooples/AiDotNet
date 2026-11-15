using AiDotNet.Diffusion.VAE;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// ImageDream's image-prompt resampler (Wang and Shi, 2023): the IP-Adapter-Plus perceiver that turns the CLIP image
/// encoder's penultimate hidden states into a fixed number of context tokens.
/// </summary>
/// <remarks>
/// <para>
/// The official <c>Resampler</c> (<c>imagedream/ldm/modules/diffusionmodules/adaptors.py</c>, after open_flamingo),
/// built by <c>MultiViewUNetModel</c> as <c>image_embed</c> with <c>dim = context_dim (1024), depth = 4,
/// dim_head = 64, heads = 12, num_queries = ip_dim (16), embedding_dim = 1280, output_dim = context_dim, ff_mult = 4</c>:
/// </para>
/// <code>
/// latents = learned [1, Q, dim]   (randn / sqrt(dim))
/// x = proj_in(features)           (Linear embedding_dim -> dim, with bias)
/// per layer:
///   latents = PerceiverAttention(x, latents) + latents
///       q = to_q(LN2(latents)),  [k, v] = to_kv(cat(LN1(x), LN2(latents)))  (all bias-free; to_out bias-free)
///   latents = FF(latents) + latents      (LN, Linear dim->4dim, GELU, Linear 4dim->dim, bias-free)
/// out = norm_out(proj_out(latents))
/// </code>
/// <para>The attention scales q and k by <c>dim_head^-1/4</c> each, which is softmax(q k^T / sqrt(dim_head)).</para>
/// </remarks>
internal sealed class ImageDreamResampler<T> : KlModule<T>
{
    private readonly int _dim;
    private readonly int _queries;
    private readonly int _embeddingDim;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly LazyWeight<T> _latents;
    private readonly TsrLinear<T> _projectionIn;
    private readonly List<(PerceiverAttention Attention, FeedForward FeedForward)> _layers = new();
    private readonly TsrLinear<T> _projectionOut;
    private readonly TsrLayerNorm<T> _normOut;

    /// <param name="dim">The latent width (ImageDream: the U-Net's context width, 1024).</param>
    /// <param name="depth">The number of perceiver layers (4).</param>
    /// <param name="headDim">The head width (64).</param>
    /// <param name="heads">The number of heads (12).</param>
    /// <param name="queries">The number of output tokens (16).</param>
    /// <param name="embeddingDim">The CLIP hidden width (ViT-H/14: 1280).</param>
    /// <param name="outputDim">The output token width (the context width, 1024).</param>
    /// <param name="feedForwardMultiplier">The feed-forward expansion (4).</param>
    public ImageDreamResampler(int dim, int depth, int headDim, int heads, int queries, int embeddingDim, int outputDim,
        int feedForwardMultiplier)
    {
        if (dim <= 0 || depth <= 0 || headDim <= 0 || heads <= 0 || queries <= 0 || embeddingDim <= 0 || outputDim <= 0
            || feedForwardMultiplier <= 0)
            throw new ArgumentException("Every resampler size must be positive.");
        _dim = dim;
        _queries = queries;
        _embeddingDim = embeddingDim;
        _heads = heads;
        _headDim = headDim;
        OutputDim = outputDim;
        _latents = TsrInit<T>.Normal(new[] { 1, queries, dim }, 1.0 / Math.Sqrt(dim));
        _projectionIn = new TsrLinear<T>(embeddingDim, dim);
        for (int i = 0; i < depth; i++)
            _layers.Add((new PerceiverAttention(dim, heads, headDim), new FeedForward(dim, checked(dim * feedForwardMultiplier))));
        _projectionOut = new TsrLinear<T>(dim, outputDim);
        _normOut = new TsrLayerNorm<T>(outputDim, 1e-5);
    }

    /// <summary>The number of output tokens (ImageDream's <c>ip_dim</c>).</summary>
    public int Queries => _queries;

    /// <summary>The input feature width.</summary>
    public int EmbeddingDim => _embeddingDim;

    /// <summary>The output token width.</summary>
    public int OutputDim { get; }

    /// <summary>Resamples image features <c>[B, N, embeddingDim]</c> to tokens <c>[B, queries, outputDim]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> features)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (features.Rank != 3 || features.Shape[2] != _embeddingDim)
            throw new ArgumentException($"The resampler reads [B, N, {_embeddingDim}] features.", nameof(features));
        int batch = features.Shape[0];
        var latents = Engine.TensorBroadcastTo(_latents.Value, new[] { batch, _queries, _dim }).Contiguous();
        var x = _projectionIn.Forward(features);
        foreach (var (attention, feedForward) in _layers)
        {
            latents = Engine.TensorAdd(attention.Forward(x, latents), latents);
            latents = Engine.TensorAdd(feedForward.Forward(latents), latents);
        }
        return _normOut.Forward(_projectionOut.Forward(latents));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "latents"), _latents.Value);
        foreach (var named in _projectionIn.NamedTensors(Join(prefix, "proj_in"))) yield return named;
        for (int i = 0; i < _layers.Count; i++)
        {
            foreach (var named in _layers[i].Attention.NamedTensors(Join(prefix, $"layers.{i}.0"))) yield return named;
            foreach (var named in _layers[i].FeedForward.NamedTensors(Join(prefix, $"layers.{i}.1"))) yield return named;
        }
        foreach (var named in _projectionOut.NamedTensors(Join(prefix, "proj_out"))) yield return named;
        foreach (var named in _normOut.NamedTensors(Join(prefix, "norm_out"))) yield return named;
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _latents;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _projectionIn;
        foreach (var (attention, feedForward) in _layers)
        {
            yield return attention;
            yield return feedForward;
        }
        yield return _projectionOut;
        yield return _normOut;
    }

    /// <summary>IP-Adapter's <c>PerceiverAttention</c>: the latents attend to the image features and to themselves.</summary>
    internal sealed class PerceiverAttention : KlModule<T>
    {
        private readonly int _heads;
        private readonly int _headDim;
        private readonly TsrLayerNorm<T> _norm1;
        private readonly TsrLayerNorm<T> _norm2;
        private readonly TsrLinear<T> _query;
        private readonly TsrLinear<T> _keyValue;
        private readonly TsrLinear<T> _output;

        public PerceiverAttention(int dim, int heads, int headDim)
        {
            _heads = heads;
            _headDim = headDim;
            int inner = checked(heads * headDim);
            _norm1 = new TsrLayerNorm<T>(dim, 1e-5);
            _norm2 = new TsrLayerNorm<T>(dim, 1e-5);
            _query = new TsrLinear<T>(dim, inner, bias: false);
            _keyValue = new TsrLinear<T>(dim, checked(inner * 2), bias: false);
            _output = new TsrLinear<T>(inner, dim, bias: false);
        }

        public Tensor<T> Forward(Tensor<T> x, Tensor<T> latents)
        {
            var image = _norm1.Forward(x);
            var normed = _norm2.Forward(latents);
            int batch = latents.Shape[0], length = latents.Shape[1];
            int inner = _heads * _headDim;
            var q = _query.Forward(normed);
            var kv = _keyValue.Forward(Engine.TensorConcatenate(new[] { image, normed }, 1));
            int keys = kv.Shape[1];
            // to_kv(...).chunk(2, dim=-1): the first half of the output features is k, the second v.
            var k = Engine.TensorNarrow(kv, 2, 0, inner).Contiguous();
            var v = Engine.TensorNarrow(kv, 2, inner, inner).Contiguous();
            Tensor<T> Heads(Tensor<T> t, int tokens)
                => Engine.TensorPermute(Engine.Reshape(t, new[] { batch, tokens, _heads, _headDim }), new[] { 0, 2, 1, 3 });
            var attended = Engine.ScaledDotProductAttention(Heads(q, length), Heads(k, keys), Heads(v, keys),
                mask: null, scale: null, isCausal: false, softcap: 0.0);
            attended = Engine.TensorPermute(attended, new[] { 0, 2, 1, 3 }).Contiguous();
            return _output.Forward(Engine.Reshape(attended, new[] { batch, length, inner }));
        }

        public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
            => _norm1.NamedTensors(Join(prefix, "norm1"))
                .Concat(_norm2.NamedTensors(Join(prefix, "norm2")))
                .Concat(_query.NamedTensors(Join(prefix, "to_q")))
                .Concat(_keyValue.NamedTensors(Join(prefix, "to_kv")))
                .Concat(_output.NamedTensors(Join(prefix, "to_out")));

        protected override IEnumerable<LazyWeightModule<T>?> Children()
        {
            yield return _norm1;
            yield return _norm2;
            yield return _query;
            yield return _keyValue;
            yield return _output;
        }
    }

    /// <summary>IP-Adapter's <c>FeedForward</c>: <c>Sequential(LayerNorm, Linear, GELU, Linear)</c>, bias-free linears.</summary>
    internal sealed class FeedForward : KlModule<T>
    {
        private readonly TsrLayerNorm<T> _norm;
        private readonly TsrLinear<T> _in;
        private readonly TsrLinear<T> _out;

        public FeedForward(int dim, int inner)
        {
            _norm = new TsrLayerNorm<T>(dim, 1e-5);
            _in = new TsrLinear<T>(dim, inner, bias: false);
            _out = new TsrLinear<T>(inner, dim, bias: false);
        }

        public Tensor<T> Forward(Tensor<T> x) => _out.Forward(TsrActivations<T>.Gelu(_in.Forward(_norm.Forward(x))));

        public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
            => _norm.NamedTensors(Join(prefix, "0"))
                .Concat(_in.NamedTensors(Join(prefix, "1")))
                .Concat(_out.NamedTensors(Join(prefix, "3")));

        protected override IEnumerable<LazyWeightModule<T>?> Children()
        {
            yield return _norm;
            yield return _in;
            yield return _out;
        }
    }
}
