using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of a VALL-E decoder stack (Wang et al. 2023, §5.1; lifeiteng/vall-e <c>TransformerEncoder</c>).</summary>
/// <param name="ModelDim">Embedding width (1024).</param>
/// <param name="Heads">Attention heads (16).</param>
/// <param name="Layers">Layers (12).</param>
/// <param name="FeedForwardDim">Feed-forward width (4096).</param>
/// <param name="Dropout">Dropout of attention weights, residual branches and the feed-forward (0.1).</param>
/// <param name="AdaptiveNorm">Whether the layer norms are adaptive (the NAR model, §4.2.2 Eq. 5).</param>
internal sealed record VallETransformerConfiguration(
    int ModelDim, int Heads, int Layers, int FeedForwardDim, double Dropout, bool AdaptiveNorm);

/// <summary>
/// A pre-norm Transformer layer as VALL-E's reference builds it (lifeiteng/vall-e <c>TransformerEncoderLayer</c> with
/// <c>norm_first=True</c>): <c>x + SelfAttention(Norm₁(x))</c>, then <c>x + FeedForward(Norm₂(x))</c>, with ReLU
/// between the feed-forward projections and dropout on the attention weights, inside the feed-forward and on both
/// residual branches.
/// </summary>
/// <remarks>
/// In the NAR model each norm is the paper's adaptive layer normalization, <c>AdaLN(h, i) = aᵢ LayerNorm(h) + bᵢ</c>
/// (§4.2.2 Eq. 5), whose <c>aᵢ, bᵢ</c> are a linear projection of stage <c>i</c>'s embedding; the LayerNorm keeps its
/// own affine parameters, as the reference's does. The attention is PyTorch's multi-head attention: one input
/// projection to queries, keys and values (with bias), scores scaled by <c>headDim^-1/2</c>, an additive mask, and an
/// output projection.
/// </remarks>
internal sealed class VallEEncoderLayer<T>
{
    private readonly IEngine _engine;
    private readonly VallETransformerConfiguration _c;

    public VallEEncoderLayer(IEngine engine, List<LayerBase<T>> layers, VallETransformerConfiguration configuration)
    {
        _engine = engine;
        _c = configuration;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        InProjection = Own(layers, new DenseLayer<T>(3 * _c.ModelDim, identity));
        OutProjection = Own(layers, new DenseLayer<T>(_c.ModelDim, identity));
        FeedForwardIn = Own(layers, new DenseLayer<T>(_c.FeedForwardDim, identity));
        FeedForwardOut = Own(layers, new DenseLayer<T>(_c.ModelDim, identity));
        Norm1 = Own(layers, new LayerNormalizationLayer<T>(_c.ModelDim));
        Norm2 = Own(layers, new LayerNormalizationLayer<T>(_c.ModelDim));
        if (_c.AdaptiveNorm)
        {
            Adaptive1 = Own(layers, new DenseLayer<T>(2 * _c.ModelDim, identity));
            Adaptive2 = Own(layers, new DenseLayer<T>(2 * _c.ModelDim, identity));
        }
    }

    public DenseLayer<T> InProjection { get; }
    public DenseLayer<T> OutProjection { get; }
    public DenseLayer<T> FeedForwardIn { get; }
    public DenseLayer<T> FeedForwardOut { get; }
    public LayerNormalizationLayer<T> Norm1 { get; }
    public LayerNormalizationLayer<T> Norm2 { get; }
    public DenseLayer<T>? Adaptive1 { get; }
    public DenseLayer<T>? Adaptive2 { get; }

    internal static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    /// <summary>One layer on <c>[length, dim]</c>; <paramref name="stage"/> is the NAR stage embedding <c>[1, dim]</c>,
    /// <paramref name="mask"/> an additive attention mask <c>[length, length]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? stage, Tensor<T>? mask, bool training, Random random)
    {
        x = _engine.TensorAdd(x, Drop(Attention(Normalize(Norm1, Adaptive1, x, stage), mask, training, random), training, random));
        var hidden = Drop(_engine.ReLU(FeedForwardIn.Forward(Normalize(Norm2, Adaptive2, x, stage))), training, random);
        return _engine.TensorAdd(x, Drop(FeedForwardOut.Forward(hidden), training, random));
    }

    internal Tensor<T> Normalize(LayerNormalizationLayer<T> norm, DenseLayer<T>? adaptive, Tensor<T> x, Tensor<T>? stage)
        => VallETransformer<T>.Normalize(_engine, norm, adaptive, x, stage, _c.ModelDim);

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random) =>
        training && _c.Dropout > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, _c.Dropout, random) : x;

    private Tensor<T> Attention(Tensor<T> x, Tensor<T>? mask, bool training, Random random)
    {
        int length = x.Shape[0], dim = _c.ModelDim, heads = _c.Heads, headDim = dim / heads;
        var projected = InProjection.Forward(x);                                                             // [L, 3d]
        Tensor<T> Heads(int part) => _engine.TensorPermute(_engine.Reshape(
            _engine.TensorSlice(projected, new[] { 0, part * dim }, new[] { length, dim }),
            new[] { length, heads, headDim }), new[] { 1, 0, 2 });                                           // [H, L, d]
        var ops = MathHelper.GetNumericOperations<T>();
        var q = _engine.TensorMultiplyScalar(Heads(0), ops.FromDouble(1.0 / Math.Sqrt(headDim)));
        var scores = _engine.BatchMatMul(q, _engine.TensorPermute(Heads(1), new[] { 0, 2, 1 }));         // [H, L, L]
        if (mask is not null)
            scores = _engine.TensorAdd(scores, _engine.TensorTile(_engine.Reshape(mask, new[] { 1, length, length }), new[] { heads, 1, 1 }));
        var weights = Drop(_engine.Softmax(scores, axis: 2), training, random);
        var context = _engine.BatchMatMul(weights, Heads(2));                                                // [H, L, d]
        var merged = _engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { length, dim });
        return OutProjection.Forward(merged);
    }
}

/// <summary>
/// VALL-E's decoder stack (lifeiteng/vall-e <c>TransformerEncoder</c>, pre-norm): the layers and a final LayerNorm,
/// adaptive in the NAR model.
/// </summary>
internal sealed class VallETransformer<T>
{
    private readonly IEngine _engine;

    public VallETransformer(IEngine engine, List<LayerBase<T>> layers, VallETransformerConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        for (int i = 0; i < configuration.Layers; i++) Layers.Add(new VallEEncoderLayer<T>(engine, layers, configuration));
        FinalNorm = VallEEncoderLayer<T>.Own(layers, new LayerNormalizationLayer<T>(configuration.ModelDim));
        if (configuration.AdaptiveNorm)
            FinalAdaptive = VallEEncoderLayer<T>.Own(layers,
                new DenseLayer<T>(2 * configuration.ModelDim, new IdentityActivation<T>() as IActivationFunction<T>));
    }

    public VallETransformerConfiguration Configuration { get; }
    public List<VallEEncoderLayer<T>> Layers { get; } = new();
    public LayerNormalizationLayer<T> FinalNorm { get; }
    public DenseLayer<T>? FinalAdaptive { get; }

    /// <summary>The stack on <c>[length, dim]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? stage, Tensor<T>? mask, bool training, Random random)
    {
        foreach (var layer in Layers) x = layer.Forward(x, stage, mask, training, random);
        return Normalize(_engine, FinalNorm, FinalAdaptive, x, stage, Configuration.ModelDim);
    }

    // AdaptiveLayerNorm: weight, bias = split(Linear(stage)); weight · LayerNorm(x) + bias. A plain LayerNorm otherwise.
    internal static Tensor<T> Normalize(IEngine engine, LayerNormalizationLayer<T> norm, DenseLayer<T>? adaptive,
        Tensor<T> x, Tensor<T>? stage, int dim)
    {
        var normed = norm.Forward(x);
        if (adaptive is null) return normed;
        if (stage is null) throw new InvalidOperationException("The adaptive layer norm needs the stage embedding.");
        int length = x.Shape[0];
        var projected = adaptive.Forward(stage);                                                             // [1, 2d]
        var scale = engine.TensorTile(engine.TensorSlice(projected, new[] { 0, 0 }, new[] { 1, dim }), new[] { length, 1 });
        var shift = engine.TensorTile(engine.TensorSlice(projected, new[] { 0, dim }, new[] { 1, dim }), new[] { length, 1 });
        return engine.TensorAdd(engine.TensorMultiply(scale, normed), shift);
    }

    /// <summary>PyTorch's default initialization, which the reference keeps (its <c>_init_weights</c> is commented out):
    /// each <c>nn.Linear</c> uniform in ±1/√fan-in (weights and biases), <c>nn.MultiheadAttention</c>'s input
    /// projection Xavier-uniform with zero biases and a zero output-projection bias, LayerNorm at one and zero.</summary>
    public void InitializeLikePyTorch(Random random)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int d = Configuration.ModelDim, ff = Configuration.FeedForwardDim;
        void Fill(DenseLayer<T> layer, int fanIn, Func<double> weight, Func<double> bias)
        {
            using (new NoGradScope<T>()) layer.Forward(new Tensor<T>(new[] { 1, fanIn }));
            var w = layer.GetWeights();
            for (int i = 0; i < w.Length; i++) w[i] = ops.FromDouble(weight());
            _engine.InvalidatePersistentTensor(w);
            var b = layer.GetBiases();
            for (int i = 0; i < b.Length; i++) b[i] = ops.FromDouble(bias());
            _engine.InvalidatePersistentTensor(b);
        }
        Func<double> Uniform(double bound) => () => (2 * random.NextDouble() - 1) * bound;
        void Linear(DenseLayer<T> layer, int fanIn) => Fill(layer, fanIn, Uniform(1 / Math.Sqrt(fanIn)), Uniform(1 / Math.Sqrt(fanIn)));
        foreach (var layer in Layers)
        {
            Fill(layer.InProjection, d, Uniform(Math.Sqrt(6.0 / (d + 3 * d))), () => 0.0);
            Fill(layer.OutProjection, d, Uniform(1 / Math.Sqrt(d)), () => 0.0);
            Linear(layer.FeedForwardIn, d);
            Linear(layer.FeedForwardOut, ff);
            if (layer.Adaptive1 is not null) Linear(layer.Adaptive1, d);
            if (layer.Adaptive2 is not null) Linear(layer.Adaptive2, d);
        }
        if (FinalAdaptive is not null) Linear(FinalAdaptive, d);
    }

    /// <summary>Loads the reference's parameters under <paramref name="prefix"/> (e.g. <c>ar_decoder</c>).</summary>
    public void LoadTorchWeights(IEngine engine, string prefix, Func<string, int[], double[]> read)
    {
        int d = Configuration.ModelDim, ff = Configuration.FeedForwardDim;
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            string p = $"{prefix}.layers.{i}";
            TorchParameters.Linear(engine, layer.InProjection, d, 3 * d, read($"{p}.self_attn.in_proj_weight", new[] { 3 * d, d }),
                read($"{p}.self_attn.in_proj_bias", new[] { 3 * d }));
            TorchParameters.Linear(engine, layer.OutProjection, d, d, read($"{p}.self_attn.out_proj.weight", new[] { d, d }),
                read($"{p}.self_attn.out_proj.bias", new[] { d }));
            TorchParameters.Linear(engine, layer.FeedForwardIn, d, ff, read($"{p}.linear1.weight", new[] { ff, d }), read($"{p}.linear1.bias", new[] { ff }));
            TorchParameters.Linear(engine, layer.FeedForwardOut, ff, d, read($"{p}.linear2.weight", new[] { d, ff }), read($"{p}.linear2.bias", new[] { d }));
            LoadNorm(engine, $"{p}.norm1", layer.Norm1, layer.Adaptive1, read);
            LoadNorm(engine, $"{p}.norm2", layer.Norm2, layer.Adaptive2, read);
        }
        LoadNorm(engine, $"{prefix}.norm", FinalNorm, FinalAdaptive, read);
    }

    private void LoadNorm(IEngine engine, string name, LayerNormalizationLayer<T> norm, DenseLayer<T>? adaptive,
        Func<string, int[], double[]> read)
    {
        int d = Configuration.ModelDim;
        if (adaptive is null)
        {
            TorchParameters.LayerNorm(engine, norm, read($"{name}.weight", new[] { d }), read($"{name}.bias", new[] { d }));
            return;
        }
        TorchParameters.LayerNorm(engine, norm, read($"{name}.norm.weight", new[] { d }), read($"{name}.norm.bias", new[] { d }));
        TorchParameters.Linear(engine, adaptive, d, 2 * d, read($"{name}.project_layer.weight", new[] { 2 * d, d }),
            read($"{name}.project_layer.bias", new[] { 2 * d }));
    }
}

/// <summary>
/// The reference's <c>SinePositionalEmbedding</c>: <c>x · scale + α · PE</c> with the sinusoidal table of Vaswani et al.
/// (sines on even, cosines on odd channels), then dropout. VALL-E's AR model learns <c>α</c> (initialized to 1); the
/// NAR model keeps it fixed at 1.
/// </summary>
internal sealed class VallEPositionalEncoding<T>
{
    private readonly IEngine _engine;
    private readonly int _dim;
    private readonly double _dropout;

    public VallEPositionalEncoding(IEngine engine, List<LayerBase<T>> layers, int dim, double dropout, bool learnScale)
    {
        _engine = engine;
        _dim = dim;
        _dropout = dropout;
        if (learnScale)
        {
            // α as a one-entry table, so it is an ordinary trainable, serialized parameter.
            Alpha = VallEEncoderLayer<T>.Own(layers, new TiedEmbeddingLayer<T>(1, 1));
            Alpha.Reinitialize(() => 1.0);
        }
    }

    /// <summary>The learned scale α (the AR model's), or null when it is fixed at 1.</summary>
    public TiedEmbeddingLayer<T>? Alpha { get; }

    public Tensor<T> Forward(Tensor<T> x, bool training, Random random)
    {
        int length = x.Shape[0];
        var ops = MathHelper.GetNumericOperations<T>();
        var table = new Tensor<T>(new[] { length, _dim });
        for (int t = 0; t < length; t++)
            for (int i = 0; i < _dim; i += 2)
            {
                double angle = t * Math.Exp(i * -(Math.Log(10000.0) / _dim));
                table[t, i] = ops.FromDouble(Math.Sin(angle));
                if (i + 1 < _dim) table[t, i + 1] = ops.FromDouble(Math.Cos(angle));
            }
        Tensor<T> scaled = table;
        if (Alpha is not null)
        {
            var alpha = Alpha.Forward(new Tensor<T>(new[] { 1 }));                                           // [1, 1]
            scaled = _engine.TensorMultiply(table, _engine.TensorTile(alpha, new[] { length, _dim }));
        }
        var output = _engine.TensorAdd(x, scaled);
        return training && _dropout > 0 ? T5Seq2Seq<T>.Dropout(_engine, output, _dropout, random) : output;
    }
}
