using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>Initializers for TripoSR weights the shared <see cref="LazyWeight{T}"/> factories do not cover.</summary>
/// <typeparam name="T">The numeric type of the weights.</typeparam>
internal static class TsrInit<T>
{
    /// <summary>Uniform in +-<paramref name="bound"/>.</summary>
    public static LazyWeight<T> Uniform(int[] shape, double bound)
        => Sampled(shape, random => ((random.NextDouble() * 2.0) - 1.0) * bound);

    /// <summary>Normal with mean zero and standard deviation <paramref name="std"/> (Box-Muller).</summary>
    public static LazyWeight<T> Normal(int[] shape, double std)
        => Sampled(shape, random =>
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            return std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        });

    // Draws in double on the host (initialization is a construction-time constant, not a forward op) and casts once.
    private static LazyWeight<T> Sampled(int[] shape, Func<Random, double> sample) => new(shape, (tensor, random) =>
    {
        var values = new double[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = sample(random);
        new Tensor<double>(shape, new Vector<double>(values)).Cast<T>().AsSpan().CopyTo(tensor.AsWritableSpan());
    });
}

/// <summary>The activations TripoSR's modules apply, in engine ops.</summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class TsrActivations<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>The exact GELU, 0.5 x (1 + erf(x / sqrt 2)), as torch's default <c>F.gelu</c> and HF ViT use.</summary>
    public static Tensor<T> Gelu(Tensor<T> x)
    {
        var erf = Engine.TensorErf(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(1.0 / Math.Sqrt(2.0))));
        return Engine.TensorMultiply(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(0.5)), Engine.TensorAddScalar(erf, NumOps.One));
    }

    /// <summary>Whether <paramref name="activation"/> is one the decoder can apply.</summary>
    public static bool IsSupported(ActivationFunction activation) => activation is ActivationFunction.ReLU
        or ActivationFunction.Swish or ActivationFunction.GELU or ActivationFunction.Softplus or ActivationFunction.Tanh
        or ActivationFunction.Sigmoid or ActivationFunction.Identity or ActivationFunction.Linear;

    /// <summary>Applies <paramref name="activation"/> element-wise.</summary>
    public static Tensor<T> Apply(ActivationFunction activation, Tensor<T> x) => activation switch
    {
        ActivationFunction.ReLU => Engine.TensorReLU(x),
        ActivationFunction.Swish => Engine.TensorSiLU(x),
        ActivationFunction.GELU => Gelu(x),
        ActivationFunction.Softplus => Engine.Softplus(x),
        ActivationFunction.Tanh => Engine.TensorTanh(x),
        ActivationFunction.Sigmoid => Engine.TensorSigmoid(x),
        ActivationFunction.Identity or ActivationFunction.Linear => x,
        _ => throw new NotSupportedException($"The TripoSR decoder does not support the {activation} activation.")
    };
}

/// <summary>
/// torch <c>nn.Linear</c>: <c>y = x W^T + b</c> over the last axis, with the weight stored in torch's
/// <c>[out, in]</c> layout so a checkpoint loads without a transpose.
/// </summary>
internal sealed class TsrLinear<T> : KlModule<T>
{
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T>? _bias;

    /// <param name="inFeatures">The input width.</param>
    /// <param name="outFeatures">The output width.</param>
    /// <param name="bias">Whether the layer has a bias.</param>
    /// <param name="weightBound">The weight's uniform bound; torch's default 1/sqrt(in) when null.</param>
    public TsrLinear(int inFeatures, int outFeatures, bool bias = true, double? weightBound = null)
    {
        if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures), "The input width must be positive.");
        if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures), "The output width must be positive.");
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        var shape = new[] { outFeatures, inFeatures };
        _weight = weightBound is double bound ? TsrInit<T>.Uniform(shape, bound) : LazyWeight<T>.Uniform(shape, inFeatures);
        _bias = bias ? LazyWeight<T>.Uniform(new[] { outFeatures }, inFeatures) : null;
    }

    public int InFeatures { get; }

    public int OutFeatures { get; }

    /// <summary>Applies the layer to the last axis of a contiguous <paramref name="x"/>.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        if (shape[shape.Length - 1] != InFeatures)
            throw new ArgumentException($"Expected a last axis of {InFeatures}, got {shape[shape.Length - 1]}.", nameof(x));
        int rows = x.Length / InFeatures;
        var y = Engine.TensorMatMulTransposed(Engine.Reshape(x, new[] { rows, InFeatures }), _weight.Value);
        if (_bias is not null)
            y = Engine.TensorAdd(y, Engine.TensorBroadcastTo(Engine.Reshape(_bias.Value, new[] { 1, OutFeatures }), new[] { rows, OutFeatures }));
        shape[shape.Length - 1] = OutFeatures;
        return Engine.Reshape(y, shape);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "weight"), _weight.Value);
        if (_bias is not null) yield return (Join(prefix, "bias"), _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        if (_bias is not null) yield return _bias;
    }
}

/// <summary>torch <c>nn.LayerNorm</c> over the last axis, with an affine scale and shift.</summary>
internal sealed class TsrLayerNorm<T> : KlModule<T>
{
    private readonly LazyWeight<T> _gamma;
    private readonly LazyWeight<T> _beta;
    private readonly double _epsilon;

    public TsrLayerNorm(int width, double epsilon)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "The width must be positive.");
        _epsilon = epsilon;
        _gamma = LazyWeight<T>.Constant(new[] { width }, 1.0);
        _beta = LazyWeight<T>.Constant(new[] { width }, 0.0);
    }

    public Tensor<T> Forward(Tensor<T> x) => Engine.LayerNorm(x, _gamma.Value, _beta.Value, _epsilon, out _, out _);

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "weight"), _gamma.Value);
        yield return (Join(prefix, "bias"), _beta.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _gamma;
        yield return _beta;
    }
}

/// <summary>
/// Multi-head scaled dot-product attention with separate query, key, value and output projections (diffusers'
/// <c>Attention</c> and HF ViT's self-attention share this shape; only their state-dict names differ). Without a
/// context it is self-attention.
/// </summary>
internal sealed class TsrAttention<T> : KlModule<T>
{
    private readonly TsrLinear<T> _query;
    private readonly TsrLinear<T> _key;
    private readonly TsrLinear<T> _value;
    private readonly TsrLinear<T> _output;
    private readonly string[] _names;
    private readonly int _heads;
    private readonly int _headDim;

    /// <param name="queryDim">The width of the attending tokens.</param>
    /// <param name="contextDim">The width of the attended tokens (the query width for self-attention).</param>
    /// <param name="heads">The number of heads.</param>
    /// <param name="headDim">The width of one head.</param>
    /// <param name="projectionBias">Whether the query, key and value projections have a bias.</param>
    /// <param name="names">The state-dict names of the query, key, value and output projections.</param>
    public TsrAttention(int queryDim, int contextDim, int heads, int headDim, bool projectionBias, string[] names)
    {
        if (heads <= 0) throw new ArgumentOutOfRangeException(nameof(heads), "The head count must be positive.");
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim), "The head width must be positive.");
        if (names is null || names.Length != 4) throw new ArgumentException("Four projection names are required.", nameof(names));
        _heads = heads;
        _headDim = headDim;
        _names = names;
        int inner = heads * headDim;
        _query = new TsrLinear<T>(queryDim, inner, projectionBias);
        _key = new TsrLinear<T>(contextDim, inner, projectionBias);
        _value = new TsrLinear<T>(contextDim, inner, projectionBias);
        _output = new TsrLinear<T>(inner, queryDim);
    }

    /// <summary>Attends <paramref name="x"/> [B, Nq, Dq] to <paramref name="context"/> [B, Nk, Dc], or to itself.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? context = null)
    {
        var source = context ?? x;
        int batch = x.Shape[0];
        int queryLength = x.Shape[1];
        int keyLength = source.Shape[1];
        var q = SplitHeads(_query.Forward(x), batch, queryLength);
        var k = SplitHeads(_key.Forward(source), batch, keyLength);
        var v = SplitHeads(_value.Forward(source), batch, keyLength);
        // softmax(Q K^T / sqrt(d_head)) V without materialising the score matrix (the 3072 triplane tokens).
        var attended = Engine.ScaledDotProductAttention(q, k, v, mask: null, scale: null, isCausal: false, softcap: 0.0);
        attended = Engine.TensorPermute(attended, new[] { 0, 2, 1, 3 }).Contiguous();
        return _output.Forward(Engine.Reshape(attended, new[] { batch, queryLength, _heads * _headDim }));
    }

    private Tensor<T> SplitHeads(Tensor<T> projected, int batch, int length)
        => Engine.TensorPermute(Engine.Reshape(projected, new[] { batch, length, _heads, _headDim }), new[] { 0, 2, 1, 3 });

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        foreach (var named in _query.NamedTensors(Join(prefix, _names[0]))) yield return named;
        foreach (var named in _key.NamedTensors(Join(prefix, _names[1]))) yield return named;
        foreach (var named in _value.NamedTensors(Join(prefix, _names[2]))) yield return named;
        foreach (var named in _output.NamedTensors(Join(prefix, _names[3]))) yield return named;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _query;
        yield return _key;
        yield return _value;
        yield return _output;
    }
}

/// <summary>
/// One HF <c>ViTLayer</c> (pre-norm): <c>x + Attn(LN(x))</c>, then <c>x + MLP(LN(x))</c> with an exact-GELU MLP.
/// </summary>
internal sealed class TsrVitLayer<T> : KlModule<T>
{
    private static readonly string[] AttentionNames =
        { "attention.attention.query", "attention.attention.key", "attention.attention.value", "attention.output.dense" };

    private readonly TsrLayerNorm<T> _normBefore;
    private readonly TsrAttention<T> _attention;
    private readonly TsrLayerNorm<T> _normAfter;
    private readonly TsrLinear<T> _intermediate;
    private readonly TsrLinear<T> _output;

    public TsrVitLayer(int width, int heads, int headDim, int mlpWidth, double epsilon)
    {
        _normBefore = new TsrLayerNorm<T>(width, epsilon);
        // HF ViTSelfAttention: query/key/value with bias (qkv_bias=True).
        _attention = new TsrAttention<T>(width, width, heads, headDim, projectionBias: true, AttentionNames);
        _normAfter = new TsrLayerNorm<T>(width, epsilon);
        _intermediate = new TsrLinear<T>(width, mlpWidth);
        _output = new TsrLinear<T>(mlpWidth, width);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        x = Engine.TensorAdd(x, _attention.Forward(_normBefore.Forward(x)));
        return Engine.TensorAdd(x, _output.Forward(TsrActivations<T>.Gelu(_intermediate.Forward(_normAfter.Forward(x)))));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        foreach (var named in _normBefore.NamedTensors(Join(prefix, "layernorm_before"))) yield return named;
        foreach (var named in _attention.NamedTensors(prefix)) yield return named;
        foreach (var named in _normAfter.NamedTensors(Join(prefix, "layernorm_after"))) yield return named;
        foreach (var named in _intermediate.NamedTensors(Join(prefix, "intermediate.dense"))) yield return named;
        foreach (var named in _output.NamedTensors(Join(prefix, "output.dense"))) yield return named;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _normBefore;
        yield return _attention;
        yield return _normAfter;
        yield return _intermediate;
        yield return _output;
    }
}

/// <summary>
/// The DINO ViT image encoder TripoSR's <c>DINOSingleImageTokenizer</c> wraps (HF <c>ViTModel</c>, no pooler):
/// a patch-embedding convolution, a CLS token, learned position embeddings resized bicubically to the patch grid,
/// pre-norm transformer layers and a final layer norm. Camera-free: unlike LRM, no camera features modulate it.
/// </summary>
internal sealed class TsrDinoEncoder<T> : KlModule<T>
{
    private readonly KlConv2d<T> _patchEmbedding;
    private readonly LazyWeight<T> _clsToken;
    private readonly LazyWeight<T> _positionEmbeddings;
    private readonly List<TsrVitLayer<T>> _layers = new();
    private readonly TsrLayerNorm<T> _norm;
    private readonly int _width;
    private readonly int _positionGrid;

    public TsrDinoEncoder(int patchSize, int width, int heads, int headDim, int mlpWidth, int layers, int positionGrid, double epsilon)
    {
        if (positionGrid <= 0) throw new ArgumentOutOfRangeException(nameof(positionGrid), "The position grid must be positive.");
        _width = width;
        _positionGrid = positionGrid;
        _patchEmbedding = new KlConv2d<T>(3, width, patchSize, stride: patchSize);
        // HF ViTPreTrainedModel._init_weights: CLS token and position embeddings ~ trunc_normal(std = 0.02).
        _clsToken = TsrInit<T>.Normal(new[] { 1, 1, width }, 0.02);
        _positionEmbeddings = TsrInit<T>.Normal(new[] { 1, (positionGrid * positionGrid) + 1, width }, 0.02);
        for (int i = 0; i < layers; i++)
            _layers.Add(new TsrVitLayer<T>(width, heads, headDim, mlpWidth, epsilon));
        _norm = new TsrLayerNorm<T>(width, epsilon);
    }

    /// <summary>The encoder width.</summary>
    public int Width => _width;

    /// <summary>Encodes normalized images [B, 3, H, W] to tokens [B, 1 + (H/P)(W/P), D] (CLS first).</summary>
    public Tensor<T> Forward(Tensor<T> images)
    {
        int batch = images.Shape[0];
        var patches = _patchEmbedding.Forward(images);
        int gridHeight = patches.Shape[2];
        int gridWidth = patches.Shape[3];
        int count = gridHeight * gridWidth;
        var tokens = Engine.TensorPermute(Engine.Reshape(patches, new[] { batch, _width, count }), new[] { 0, 2, 1 }).Contiguous();
        var cls = Engine.TensorBroadcastTo(_clsToken.Value, new[] { batch, 1, _width }).Contiguous();
        var x = Engine.TensorConcatenate(new[] { cls, tokens }, 1);
        var positions = PositionEmbeddings(gridHeight, gridWidth);
        x = Engine.TensorAdd(x, Engine.TensorBroadcastTo(positions, new[] { batch, count + 1, _width }));
        foreach (var layer in _layers)
            x = layer.Forward(x);
        return _norm.Forward(x);
    }

    // HF ViTEmbeddings.interpolate_pos_encoding: the patch embeddings as a [1, D, g, g] image, resized bicubically
    // (align_corners=False) to the input's patch grid; the CLS embedding is kept as is.
    private Tensor<T> PositionEmbeddings(int gridHeight, int gridWidth)
    {
        var all = _positionEmbeddings.Value;
        if (gridHeight == _positionGrid && gridWidth == _positionGrid) return all;
        var cls = Engine.TensorNarrow(all, 1, 0, 1).Contiguous();
        var patch = Engine.Reshape(Engine.TensorNarrow(all, 1, 1, _positionGrid * _positionGrid).Contiguous(),
            new[] { 1, _positionGrid, _positionGrid, _width });
        var image = Engine.TensorPermute(patch, new[] { 0, 3, 1, 2 }).Contiguous();
        var resized = Engine.Interpolate(image, new[] { gridHeight, gridWidth }, InterpolateMode.Bicubic, false);
        var back = Engine.Reshape(Engine.TensorPermute(resized, new[] { 0, 2, 3, 1 }).Contiguous(),
            new[] { 1, gridHeight * gridWidth, _width });
        return Engine.TensorConcatenate(new[] { cls, back }, 1);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "embeddings.cls_token"), _clsToken.Value);
        yield return (Join(prefix, "embeddings.position_embeddings"), _positionEmbeddings.Value);
        foreach (var named in _patchEmbedding.NamedTensors(Join(prefix, "embeddings.patch_embeddings.projection"))) yield return named;
        for (int i = 0; i < _layers.Count; i++)
            foreach (var named in _layers[i].NamedTensors(Join(prefix, "encoder.layer." + i))) yield return named;
        foreach (var named in _norm.NamedTensors(Join(prefix, "layernorm"))) yield return named;
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _clsToken;
        yield return _positionEmbeddings;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _patchEmbedding;
        foreach (var layer in _layers) yield return layer;
        yield return _norm;
    }
}

/// <summary>
/// TripoSR's <c>BasicTransformerBlock</c> (diffusers layout, layer-norm type): self-attention, cross-attention to the
/// image tokens and a GEGLU feed-forward, each pre-normed and residual. Projections have no bias except the outputs.
/// </summary>
internal sealed class TsrTransformerBlock<T> : KlModule<T>
{
    private static readonly string[] AttentionNames = { "to_q", "to_k", "to_v", "to_out.0" };

    // torch nn.LayerNorm's default epsilon, which BasicTransformerBlock leaves in place.
    private const double NormEpsilon = 1e-5;

    private readonly TsrLayerNorm<T> _norm1;
    private readonly TsrAttention<T> _selfAttention;
    private readonly TsrLayerNorm<T> _norm2;
    private readonly TsrAttention<T> _crossAttention;
    private readonly TsrLayerNorm<T> _norm3;
    private readonly TsrLinear<T> _gegluProjection;
    private readonly TsrLinear<T> _feedForwardOut;
    private readonly int _feedForwardWidth;

    public TsrTransformerBlock(int width, int heads, int headDim, int contextWidth)
    {
        _feedForwardWidth = 4 * width;
        _norm1 = new TsrLayerNorm<T>(width, NormEpsilon);
        _selfAttention = new TsrAttention<T>(width, width, heads, headDim, projectionBias: false, AttentionNames);
        _norm2 = new TsrLayerNorm<T>(width, NormEpsilon);
        _crossAttention = new TsrAttention<T>(width, contextWidth, heads, headDim, projectionBias: false, AttentionNames);
        _norm3 = new TsrLayerNorm<T>(width, NormEpsilon);
        // diffusers FeedForward(activation_fn="geglu", mult=4): GEGLU projects to twice the inner width.
        _gegluProjection = new TsrLinear<T>(width, 2 * _feedForwardWidth);
        _feedForwardOut = new TsrLinear<T>(_feedForwardWidth, width);
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T> context)
    {
        x = Engine.TensorAdd(x, _selfAttention.Forward(_norm1.Forward(x)));
        x = Engine.TensorAdd(x, _crossAttention.Forward(_norm2.Forward(x), context));
        var projected = _gegluProjection.Forward(_norm3.Forward(x));
        int axis = projected.Rank - 1;
        // GEGLU: hidden, gate = proj(x).chunk(2, dim=-1); hidden * gelu(gate).
        var hidden = Engine.TensorNarrow(projected, axis, 0, _feedForwardWidth);
        var gate = Engine.TensorNarrow(projected, axis, _feedForwardWidth, _feedForwardWidth);
        var gated = Engine.TensorMultiply(hidden.Contiguous(), TsrActivations<T>.Gelu(gate.Contiguous()));
        return Engine.TensorAdd(x, _feedForwardOut.Forward(gated));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        foreach (var named in _norm1.NamedTensors(Join(prefix, "norm1"))) yield return named;
        foreach (var named in _selfAttention.NamedTensors(Join(prefix, "attn1"))) yield return named;
        foreach (var named in _norm2.NamedTensors(Join(prefix, "norm2"))) yield return named;
        foreach (var named in _crossAttention.NamedTensors(Join(prefix, "attn2"))) yield return named;
        foreach (var named in _norm3.NamedTensors(Join(prefix, "norm3"))) yield return named;
        foreach (var named in _gegluProjection.NamedTensors(Join(prefix, "ff.net.0.proj"))) yield return named;
        foreach (var named in _feedForwardOut.NamedTensors(Join(prefix, "ff.net.2"))) yield return named;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _norm1;
        yield return _selfAttention;
        yield return _norm2;
        yield return _crossAttention;
        yield return _norm3;
        yield return _gegluProjection;
        yield return _feedForwardOut;
    }
}

/// <summary>
/// TripoSR's <c>Transformer1D</c> over [B, C, N] tokens: GroupNorm (eps 1e-6), a projection to the inner width, the
/// blocks, a projection back, and a residual connection to the un-normalized input.
/// </summary>
internal sealed class TsrTransformer1D<T> : KlModule<T>
{
    private readonly KlGroupNorm<T> _norm;
    private readonly TsrLinear<T> _projectIn;
    private readonly List<TsrTransformerBlock<T>> _blocks = new();
    private readonly TsrLinear<T> _projectOut;

    public TsrTransformer1D(int channels, int normGroups, int heads, int headDim, int layers, int contextWidth)
    {
        int inner = heads * headDim;
        _norm = new KlGroupNorm<T>(normGroups, channels, 1e-6);
        _projectIn = new TsrLinear<T>(channels, inner);
        for (int i = 0; i < layers; i++)
            _blocks.Add(new TsrTransformerBlock<T>(inner, heads, headDim, contextWidth));
        _projectOut = new TsrLinear<T>(inner, channels);
    }

    /// <summary>Transforms tokens [B, C, N] attending to <paramref name="context"/> [B, M, Dc].</summary>
    public Tensor<T> Forward(Tensor<T> tokens, Tensor<T> context)
    {
        int batch = tokens.Shape[0];
        int channels = tokens.Shape[1];
        int count = tokens.Shape[2];
        var normalized = Engine.Reshape(_norm.Forward(Engine.Reshape(tokens, new[] { batch, channels, count, 1 })),
            new[] { batch, channels, count });
        var h = _projectIn.Forward(Engine.TensorPermute(normalized, new[] { 0, 2, 1 }).Contiguous());
        foreach (var block in _blocks)
            h = block.Forward(h, context);
        h = _projectOut.Forward(h);
        return Engine.TensorAdd(tokens, Engine.TensorPermute(h, new[] { 0, 2, 1 }).Contiguous());
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        foreach (var named in _norm.NamedTensors(Join(prefix, "norm"))) yield return named;
        foreach (var named in _projectIn.NamedTensors(Join(prefix, "proj_in"))) yield return named;
        for (int i = 0; i < _blocks.Count; i++)
            foreach (var named in _blocks[i].NamedTensors(Join(prefix, "transformer_blocks." + i))) yield return named;
        foreach (var named in _projectOut.NamedTensors(Join(prefix, "proj_out"))) yield return named;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _norm;
        yield return _projectIn;
        foreach (var block in _blocks) yield return block;
        yield return _projectOut;
    }
}

/// <summary>
/// TripoSR's <c>Triplane1DTokenizer</c>: learnable embeddings [3, C, R, R] (initialized N(0, 1/C)) flattened to
/// tokens [C, 3 R R] in plane-major order, and the inverse rearrangement of the transformer output.
/// </summary>
internal sealed class TsrTriplaneTokenizer<T> : KlModule<T>
{
    private readonly LazyWeight<T> _embeddings;
    private readonly int _channels;
    private readonly int _resolution;

    public TsrTriplaneTokenizer(int resolution, int channels)
    {
        if (resolution <= 0) throw new ArgumentOutOfRangeException(nameof(resolution), "The plane size must be positive.");
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels), "The token width must be positive.");
        _resolution = resolution;
        _channels = channels;
        _embeddings = TsrInit<T>.Normal(new[] { 3, channels, resolution, resolution }, 1.0 / Math.Sqrt(channels));
    }

    public int TokenCount => 3 * _resolution * _resolution;

    /// <summary>"Np Ct Hp Wp -> B Ct (Np Hp Wp)": the tokens [B, C, 3 R R].</summary>
    public Tensor<T> Tokens(int batch)
    {
        var flat = Engine.Reshape(Engine.TensorPermute(_embeddings.Value, new[] { 1, 0, 2, 3 }).Contiguous(),
            new[] { 1, _channels, TokenCount });
        return Engine.TensorBroadcastTo(flat, new[] { batch, _channels, TokenCount }).Contiguous();
    }

    /// <summary>"B Ct (Np Hp Wp) -> B Np Ct Hp Wp": the planes [B, 3, C, R, R].</summary>
    public Tensor<T> Detokenize(Tensor<T> tokens)
    {
        int batch = tokens.Shape[0];
        return Engine.TensorPermute(Engine.Reshape(tokens, new[] { batch, _channels, 3, _resolution, _resolution }),
            new[] { 0, 2, 1, 3, 4 }).Contiguous();
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "embeddings"), _embeddings.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _embeddings;
    }
}

/// <summary>
/// TripoSR's <c>TriplaneUpsampleNetwork</c>: one <c>ConvTranspose2d(C_in, C_out, kernel 2, stride 2)</c> shared by
/// the three planes, doubling their resolution.
/// </summary>
internal sealed class TsrTriplaneUpsampler<T> : KlModule<T>
{
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T> _bias;
    private readonly int _outChannels;

    public TsrTriplaneUpsampler(int inChannels, int outChannels)
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels), "The input channels must be positive.");
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels), "The output channels must be positive.");
        _outChannels = outChannels;
        // torch computes a transposed convolution's fan-in from weight.size(1) * k * k = out * 4.
        int fanIn = outChannels * 4;
        _weight = LazyWeight<T>.Uniform(new[] { inChannels, outChannels, 2, 2 }, fanIn);
        _bias = LazyWeight<T>.Uniform(new[] { outChannels }, fanIn);
    }

    /// <summary>Upsamples planes [B, 3, C_in, R, R] to [B, 3, C_out, 2R, 2R].</summary>
    public Tensor<T> Forward(Tensor<T> planes)
    {
        int batch = planes.Shape[0];
        int height = planes.Shape[3];
        int width = planes.Shape[4];
        var packed = Engine.Reshape(planes, new[] { batch * 3, planes.Shape[2], height, width });
        var up = Engine.ConvTranspose2D(packed, _weight.Value, new[] { 2, 2 }, new[] { 0, 0 }, new[] { 0, 0 });
        up = AddChannelBias(up, _bias.Value);
        return Engine.Reshape(up, new[] { batch, 3, _outChannels, 2 * height, 2 * width });
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "upsample.weight"), _weight.Value);
        yield return (Join(prefix, "upsample.bias"), _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>
/// TripoSR's <c>NeRFMLP</c>: <c>Linear(in, w)</c>, then <c>n - 1</c> more <c>Linear(w, w)</c>, each followed by the
/// activation, then <c>Linear(w, 4)</c> giving the raw density and three colour features. Weights are
/// kaiming-uniform for ReLU (bound sqrt(6 / fan_in)), as <c>make_linear</c> initializes them.
/// </summary>
internal sealed class TsrNeRFDecoder<T> : KlModule<T>
{
    private readonly List<TsrLinear<T>> _layers = new();
    private readonly ActivationFunction _activation;

    public TsrNeRFDecoder(int inChannels, int width, int hiddenLayers, ActivationFunction activation)
    {
        if (hiddenLayers <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenLayers), "The decoder needs a hidden layer.");
        _activation = activation;
        _layers.Add(new TsrLinear<T>(inChannels, width, weightBound: Math.Sqrt(6.0 / inChannels)));
        for (int i = 0; i < hiddenLayers - 1; i++)
            _layers.Add(new TsrLinear<T>(width, width, weightBound: Math.Sqrt(6.0 / width)));
        _layers.Add(new TsrLinear<T>(width, 4, weightBound: Math.Sqrt(6.0 / width)));
    }

    /// <summary>The decoder's input width.</summary>
    public int InChannels => _layers[0].InFeatures;

    /// <summary>Decodes features [N, in] to raw outputs [N, 4]: density, then three colour features.</summary>
    public Tensor<T> Forward(Tensor<T> features)
    {
        var x = features;
        int last = _layers.Count - 1;
        for (int i = 0; i < last; i++)
            x = TsrActivations<T>.Apply(_activation, _layers[i].Forward(x));
        return _layers[last].Forward(x);
    }

    // nn.Sequential indices: a linear layer every second slot, the activations between them.
    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        for (int i = 0; i < _layers.Count; i++)
            foreach (var named in _layers[i].NamedTensors(Join(prefix, "layers." + (2 * i)))) yield return named;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        foreach (var layer in _layers) yield return layer;
    }
}

/// <summary>
/// The whole TripoSR network (<c>TSR</c>): image tokenizer, triplane tokenizer, backbone, post-processor and decoder,
/// named as the official checkpoint names them.
/// </summary>
internal sealed class TripoSRNetwork<T> : KlModule<T>
{
    private readonly double[] _mean;
    private readonly double[] _std;
    private readonly int _imageSize;

    public TripoSRNetwork(TripoSROptions options)
    {
        _mean = (double[])options.ImageMean.Clone();
        _std = (double[])options.ImageStd.Clone();
        _imageSize = options.ImageSize;
        Encoder = new TsrDinoEncoder<T>(options.PatchSize, options.EncoderWidth, options.EncoderHeads, options.EncoderHeadDim,
            options.EncoderMlpRatio * options.EncoderWidth, options.EncoderLayers, options.EncoderPositionGridSize,
            options.EncoderLayerNormEpsilon);
        Tokenizer = new TsrTriplaneTokenizer<T>(options.TriplaneTokenResolution, options.TriplaneTokenChannels);
        Backbone = new TsrTransformer1D<T>(options.TriplaneTokenChannels, options.TransformerNormGroups,
            options.TransformerHeads, options.TransformerHeadDim, options.TransformerLayers, options.EncoderWidth);
        Upsampler = new TsrTriplaneUpsampler<T>(options.TriplaneTokenChannels, options.TriplaneChannels);
        int decoderInput = options.FeatureReduction == TriplaneFeatureReduction.Concat
            ? 3 * options.TriplaneChannels
            : options.TriplaneChannels;
        Decoder = new TsrNeRFDecoder<T>(decoderInput, options.DecoderHiddenWidth, options.DecoderHiddenLayers,
            options.DecoderActivation);
    }

    public TsrDinoEncoder<T> Encoder { get; }

    public TsrTriplaneTokenizer<T> Tokenizer { get; }

    public TsrTransformer1D<T> Backbone { get; }

    public TsrTriplaneUpsampler<T> Upsampler { get; }

    public TsrNeRFDecoder<T> Decoder { get; }

    /// <summary>
    /// The image tokens [B, M, D] of RGB images [B, 3, H, W] in [0, 1]: resized to the conditioning size (bilinear),
    /// ImageNet-normalized and encoded.
    /// </summary>
    public Tensor<T> EncodeImage(Tensor<T> images)
    {
        var x = images;
        if (x.Shape[2] != _imageSize || x.Shape[3] != _imageSize)
            x = Engine.Interpolate(x, new[] { _imageSize, _imageSize }, InterpolateMode.Bilinear, false);
        var shape = x.Shape.ToArray();
        var mean = Engine.TensorBroadcastTo(ChannelConstant(_mean), shape);
        var std = Engine.TensorBroadcastTo(ChannelConstant(_std), shape);
        return Encoder.Forward(Engine.TensorDivide(Engine.TensorSubtract(x, mean), std));
    }

    /// <summary>The scene codes (triplanes) [B, 3, C, 2R, 2R] of RGB images [B, 3, H, W] in [0, 1].</summary>
    public Tensor<T> SceneCodes(Tensor<T> images)
    {
        int batch = images.Shape[0];
        var imageTokens = EncodeImage(images);
        var tokens = Backbone.Forward(Tokenizer.Tokens(batch), imageTokens);
        return Upsampler.Forward(Tokenizer.Detokenize(tokens));
    }

    private static Tensor<T> ChannelConstant(double[] values)
        => new Tensor<double>(new[] { 1, 3, 1, 1 }, new Vector<double>(values)).Cast<T>();

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        foreach (var named in Encoder.NamedTensors(Join(prefix, "image_tokenizer.model"))) yield return named;
        foreach (var named in Tokenizer.NamedTensors(Join(prefix, "tokenizer"))) yield return named;
        foreach (var named in Backbone.NamedTensors(Join(prefix, "backbone"))) yield return named;
        foreach (var named in Upsampler.NamedTensors(Join(prefix, "post_processor"))) yield return named;
        foreach (var named in Decoder.NamedTensors(Join(prefix, "decoder"))) yield return named;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return Encoder;
        yield return Tokenizer;
        yield return Backbone;
        yield return Upsampler;
        yield return Decoder;
    }
}
