using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>How a MGLD-VSR weight starts.</summary>
internal enum MgldInit
{
    /// <summary>torch's default for Conv and Linear: uniform in ±1/√fan_in.</summary>
    Default,

    /// <summary><c>zero_module</c>: every value zero.</summary>
    Zero,

    /// <summary>taming's <c>weights_init</c> for the discriminator's convolutions: N(0, 0.02).</summary>
    Normal002,

    /// <summary>BasicSR's <c>default_init_weights(scale=0.1)</c>: Kaiming normal (fan in) times 0.1.</summary>
    ScaledKaiming,
}

/// <summary>Weights and tensor helpers shared by MGLD-VSR's networks.</summary>
internal static class MgldOps
{
    private static double Gaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    internal static LazyWeight<T> Weight<T>(int[] shape, int fanIn, MgldInit init)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return init switch
        {
            MgldInit.Zero => LazyWeight<T>.Constant(shape, 0.0),
            MgldInit.Normal002 => new LazyWeight<T>(shape, (tensor, random) =>
            {
                for (int i = 0; i < tensor.Length; i++) tensor[i] = ops.FromDouble(0.02 * Gaussian(random));
            }),
            MgldInit.ScaledKaiming => new LazyWeight<T>(shape, (tensor, random) =>
            {
                double std = 0.1 * Math.Sqrt(2.0 / fanIn);
                for (int i = 0; i < tensor.Length; i++) tensor[i] = ops.FromDouble(std * Gaussian(random));
            }),
            _ => LazyWeight<T>.Uniform(shape, fanIn),
        };
    }

    /// <summary>Adds a per-channel bias to [N, C, ...].</summary>
    internal static Tensor<T> ChannelBias<T>(IEngine engine, Tensor<T> x, Tensor<T> bias)
    {
        var shape = Enumerable.Repeat(1, x.Rank).ToArray();
        shape[1] = bias.Length;
        return engine.TensorAdd(x, engine.TensorBroadcastTo(engine.Reshape(bias, shape), x.Shape.ToArray()));
    }

    /// <summary>Scales or shifts [..., D] by a [D] vector along the last axis.</summary>
    internal static Tensor<T> LastAxis<T>(IEngine engine, Tensor<T> x, Tensor<T> vector, bool multiply)
    {
        var shape = Enumerable.Repeat(1, x.Rank).ToArray();
        shape[x.Rank - 1] = vector.Length;
        var broadcast = engine.TensorBroadcastTo(engine.Reshape(vector, shape), x.Shape.ToArray());
        return multiply ? engine.TensorMultiply(x, broadcast) : engine.TensorAdd(x, broadcast);
    }

    /// <summary>
    /// openaimodel's <c>timestep_embedding</c>: [cos, sin] of t·10000^(−i/half) for i below half the width, one row a
    /// timestep.
    /// </summary>
    internal static Tensor<T> TimestepEmbedding<T>(IReadOnlyList<double> timesteps, int dim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int half = dim / 2;
        var embedding = new Tensor<T>(new[] { timesteps.Count, dim });
        for (int b = 0; b < timesteps.Count; b++)
            for (int i = 0; i < half; i++)
            {
                double angle = timesteps[b] * Math.Exp(-Math.Log(10000.0) * i / half);
                embedding[b, i] = ops.FromDouble(Math.Cos(angle));
                embedding[b, half + i] = ops.FromDouble(Math.Sin(angle));
            }
        return embedding;
    }

    /// <summary>Repeats each of B rows T times: one row a frame.</summary>
    internal static Tensor<T> PerFrame<T>(IEngine engine, Tensor<T> rows, int frames)
    {
        int b = rows.Shape[0];
        var tail = rows._shape.Skip(1).ToArray();
        var expanded = engine.TensorBroadcastTo(engine.Reshape(rows, new[] { b, 1 }.Concat(tail).ToArray()), new[] { b, frames }.Concat(tail).ToArray());
        return engine.Reshape(expanded, new[] { b * frames }.Concat(tail).ToArray());
    }
}

/// <summary>A set of modules registered together, so a network's frozen and trained weights can be two components.</summary>
internal sealed class LazyModuleGroup<T> : LazyWeightModule<T>
{
    private readonly List<LazyWeightModule<T>> _modules = new();

    public LazyModuleGroup(MgldInit zeroModuleInit = MgldInit.Zero) => ZeroModuleInit = zeroModuleInit;

    /// <summary>
    /// The initialization of the layers LDM wraps in <c>zero_module</c>: zero where the weights are trained from
    /// scratch, the default where they stand in for Stable Diffusion's pretrained ones, which are not zero (a zero
    /// output convolution in a frozen UNet would make ε̂ ≡ 0 and leave nothing upstream a gradient).
    /// </summary>
    public MgldInit ZeroModuleInit { get; }

    /// <summary>Adds <paramref name="module"/> and returns it.</summary>
    public TModule Add<TModule>(TModule module) where TModule : LazyWeightModule<T>
    {
        _modules.Add(module);
        return module;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _modules;
}

/// <summary>Where a network's leaves go: the weights it keeps from Stable Diffusion and the weights MGLD-VSR trains.</summary>
internal sealed class MgldParts<T>
{
    public LazyModuleGroup<T> Frozen { get; } = new(MgldInit.Default);

    public LazyModuleGroup<T> Trained { get; } = new();

    public LazyModuleGroup<T> Of(bool trained) => trained ? Trained : Frozen;
}

/// <summary>nn.Conv2d (bias optional).</summary>
internal sealed class MgldConv2d<T> : LazyWeightModule<T>
{
    private readonly int _stride, _padH, _padW;
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T>? _bias;

    public MgldConv2d(int inChannels, int outChannels, int kernel, int stride = 1, int padding = 0, MgldInit init = MgldInit.Default, bool bias = true)
        : this(inChannels, outChannels, kernel, kernel, padding, padding, stride, init, bias)
    {
    }

    /// <summary>A rectangular kernel (RAFT's separable GRU uses 1x5 and 5x1).</summary>
    public MgldConv2d(int inChannels, int outChannels, int kernelH, int kernelW, int padH, int padW, int stride = 1,
        MgldInit init = MgldInit.Default, bool bias = true)
    {
        _stride = stride;
        _padH = padH;
        _padW = padW;
        int fanIn = inChannels * kernelH * kernelW;
        _weight = MgldOps.Weight<T>(new[] { outChannels, inChannels, kernelH, kernelW }, fanIn, init);
        if (bias) _bias = init == MgldInit.Default ? LazyWeight<T>.Uniform(new[] { outChannels }, fanIn) : LazyWeight<T>.Constant(new[] { outChannels }, 0.0);
    }

    /// <summary>The kernel, for the adaptive GAN weight's last-layer gradient.</summary>
    public Tensor<T> Kernel => _weight.Value;

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = Engine.Conv2D(x, _weight.Value, new[] { _stride, _stride }, new[] { _padH, _padW }, new[] { 1, 1 });
        return _bias is null ? y : MgldOps.ChannelBias(Engine, y, _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        if (_bias is not null) yield return _bias;
    }
}

/// <summary>nn.Linear (bias optional) over the last axis.</summary>
internal sealed class MgldLinear<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T>? _bias;

    public MgldLinear(int inFeatures, int outFeatures, bool bias = true, MgldInit init = MgldInit.Default)
    {
        _weight = MgldOps.Weight<T>(new[] { outFeatures, inFeatures }, inFeatures, init);
        if (bias) _bias = init == MgldInit.Default ? LazyWeight<T>.Uniform(new[] { outFeatures }, inFeatures) : LazyWeight<T>.Constant(new[] { outFeatures }, 0.0);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int inFeatures = shape[shape.Length - 1];
        var flat = Engine.Reshape(x, new[] { x.Length / inFeatures, inFeatures });
        var y = Engine.TensorMatMul(flat, Engine.TensorPermute(_weight.Value, new[] { 1, 0 }));
        if (_bias is not null) y = MgldOps.LastAxis(Engine, y, _bias.Value, multiply: false);
        shape[shape.Length - 1] = _weight.Shape[0];
        return Engine.Reshape(y, shape);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        if (_bias is not null) yield return _bias;
    }
}

/// <summary>nn.GroupNorm with an affine scale and shift over [N, C, ...].</summary>
internal sealed class MgldGroupNorm<T> : LazyWeightModule<T>
{
    private readonly int _groups;
    private readonly double _epsilon;
    private readonly LazyWeight<T> _gamma, _beta;

    public MgldGroupNorm(int channels, double epsilon, int groups = 32)
    {
        if (channels % groups != 0) throw new ArgumentException($"{channels} channels do not split into {groups} groups.", nameof(channels));
        _groups = groups;
        _epsilon = epsilon;
        _gamma = LazyWeight<T>.Constant(new[] { channels }, 1.0);
        _beta = LazyWeight<T>.Constant(new[] { channels }, 0.0);
    }

    public Tensor<T> Forward(Tensor<T> x) => Engine.GroupNorm(x, _groups, _gamma.Value, _beta.Value, _epsilon, out _, out _);

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _gamma;
        yield return _beta;
    }
}

/// <summary>nn.LayerNorm (affine, epsilon 1e-5) over the last axis.</summary>
internal sealed class MgldLayerNorm<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _gamma, _beta;

    public MgldLayerNorm(int dim)
    {
        _gamma = LazyWeight<T>.Constant(new[] { dim }, 1.0);
        _beta = LazyWeight<T>.Constant(new[] { dim }, 0.0);
    }

    public Tensor<T> Forward(Tensor<T> x)
        => MgldOps.LastAxis(Engine, MgldOps.LastAxis(Engine, WanOps.LayerNorm(Engine, x, 1e-5), _gamma.Value, multiply: true), _beta.Value, multiply: false);

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _gamma;
        yield return _beta;
    }
}

/// <summary>
/// The learned blend α of the temporal modules: α·temporal + (1 − α)·input. The release declares it uninitialized
/// (<c>torch.Tensor(1)</c>) and loads it; it starts at zero here, so a new temporal module begins as the identity.
/// </summary>
internal sealed class MgldAlpha<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _alpha = LazyWeight<T>.Constant(new[] { 1 }, 0.0);

    public Tensor<T> Blend(Tensor<T> temporal, Tensor<T> input)
    {
        var alpha = Engine.TensorBroadcastTo(Engine.Reshape(_alpha.Value, Enumerable.Repeat(1, input.Rank).ToArray()), input.Shape.ToArray());
        return Engine.TensorAdd(input, Engine.TensorMultiply(alpha, Engine.TensorSubtract(temporal, input)));
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _alpha;
    }
}

/// <summary>
/// <c>SpatialTemporalConv</c>: a (3, 1, 1) convolution across each pixel's frames, blended into the input by α. Every
/// weight is a "temporal" one, so MGLD-VSR trains it.
/// </summary>
internal sealed class MgldTemporalConv<T>
{
    private readonly LazyWeight<T> _weight, _bias;
    private readonly MgldAlpha<T> _alpha;

    public MgldTemporalConv(MgldParts<T> parts, int channels)
    {
        int fanIn = channels * 3;
        _weight = LazyWeight<T>.Uniform(new[] { channels, channels, 3, 1, 1 }, fanIn);
        _bias = LazyWeight<T>.Uniform(new[] { channels }, fanIn);
        parts.Trained.Add(new MgldWeights<T>(_weight, _bias));
        _alpha = parts.Trained.Add(new MgldAlpha<T>());
    }

    /// <summary>[B·T, C, H, W] frames of B clips.</summary>
    public Tensor<T> Forward(IEngine engine, Tensor<T> x, int frames)
    {
        int bt = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3], b = bt / frames;
        var clip = engine.TensorPermute(engine.Reshape(x, new[] { b, frames, c, h, w }), new[] { 0, 2, 1, 3, 4 });
        var y = MgldOps.ChannelBias(engine, engine.Conv3D(clip, _weight.Value, new[] { 1, 1, 1 }, new[] { 1, 0, 0 }, new[] { 1, 1, 1 }), _bias.Value);
        var back = engine.Reshape(engine.TensorPermute(y, new[] { 0, 2, 1, 3, 4 }), new[] { bt, c, h, w });
        return _alpha.Blend(back, x);
    }
}

/// <summary>A module that only owns the given weights, for blocks that compute with them directly.</summary>
internal sealed class MgldWeights<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T>[] _weights;

    public MgldWeights(params LazyWeight<T>[] weights) => _weights = weights;

    protected override IEnumerable<LazyWeight<T>> OwnWeights() => _weights;
}

/// <summary>
/// The latent diffusion's multi-head attention (<c>CrossAttention</c>): bias-free q, k and v (k and v from the context
/// when one is given), softmax(q·kᵀ/√d)·v per head, and a biased output projection.
/// </summary>
internal sealed class MgldAttention<T>
{
    private readonly int _heads;
    private readonly MgldLinear<T> _q, _k, _v, _out;

    public MgldAttention(LazyModuleGroup<T> group, int queryDim, int heads, int headDim, int? contextDim = null)
    {
        _heads = heads;
        int inner = heads * headDim, context = contextDim ?? queryDim;
        _q = group.Add(new MgldLinear<T>(queryDim, inner, bias: false));
        _k = group.Add(new MgldLinear<T>(context, inner, bias: false));
        _v = group.Add(new MgldLinear<T>(context, inner, bias: false));
        _out = group.Add(new MgldLinear<T>(inner, queryDim));
    }

    /// <summary>[N, L, D] queries on an optional [N, Lc, Dc] context.</summary>
    public Tensor<T> Forward(IEngine engine, Tensor<T> x, Tensor<T>? context = null)
    {
        var source = context ?? x;
        var attended = WanOps.Attend(engine, WanOps.Heads(engine, _q.Forward(x), _heads),
            WanOps.Heads(engine, _k.Forward(source), _heads), WanOps.Heads(engine, _v.Forward(source), _heads), null);
        return _out.Forward(WanOps.Merge(engine, attended, _heads));
    }
}

/// <summary>
/// <c>TemporalAttention</c>: each pixel's frames attend to each other after a LayerNorm, blended into the input by α.
/// The release trains the parameters whose names hold "temporal" (the attention and α), not the LayerNorm.
/// </summary>
internal sealed class MgldTemporalAttention<T>
{
    private readonly MgldLayerNorm<T> _norm;
    private readonly MgldAttention<T> _attention;
    private readonly MgldAlpha<T> _alpha;

    public MgldTemporalAttention(MgldParts<T> parts, int channels, int heads, int headDim)
    {
        _attention = new MgldAttention<T>(parts.Trained, channels, heads, headDim);
        _norm = parts.Frozen.Add(new MgldLayerNorm<T>(channels));
        _alpha = parts.Trained.Add(new MgldAlpha<T>());
    }

    public Tensor<T> Forward(IEngine engine, Tensor<T> x, int frames)
    {
        int bt = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3], b = bt / frames;
        // (b t) c h w -> (b h w) t c
        var sequence = engine.Reshape(engine.TensorPermute(engine.Reshape(x, new[] { b, frames, c, h, w }), new[] { 0, 3, 4, 1, 2 }),
            new[] { b * h * w, frames, c });
        var attended = _attention.Forward(engine, _norm.Forward(sequence));
        var back = engine.Reshape(engine.TensorPermute(engine.Reshape(attended, new[] { b, h, w, frames, c }), new[] { 0, 3, 4, 1, 2 }),
            new[] { bt, c, h, w });
        return _alpha.Blend(back, x);
    }
}

/// <summary>
/// <c>SpatialTransformerV2</c> with linear projections (SD 2.1): GroupNorm (32, 1e-6), a linear map in, one
/// <c>BasicTransformerBlockV2</c> (self-attention, cross-attention on the text context, a GEGLU feed-forward, each after
/// a LayerNorm and added back), a zero-initialized linear map out, and the input added back.
/// </summary>
internal sealed class MgldSpatialTransformer<T>
{
    private readonly MgldGroupNorm<T> _norm;
    private readonly MgldLinear<T> _projIn, _projOut, _ffIn, _ffOut;
    private readonly MgldLayerNorm<T> _norm1, _norm2, _norm3;
    private readonly MgldAttention<T> _attn1, _attn2;

    public MgldSpatialTransformer(LazyModuleGroup<T> group, int channels, int heads, int headDim, int contextDim)
    {
        int inner = heads * headDim;
        _norm = group.Add(new MgldGroupNorm<T>(channels, 1e-6));
        _projIn = group.Add(new MgldLinear<T>(channels, inner));
        _norm1 = group.Add(new MgldLayerNorm<T>(inner));
        _attn1 = new MgldAttention<T>(group, inner, heads, headDim);
        _norm2 = group.Add(new MgldLayerNorm<T>(inner));
        _attn2 = new MgldAttention<T>(group, inner, heads, headDim, contextDim);
        _norm3 = group.Add(new MgldLayerNorm<T>(inner));
        _ffIn = group.Add(new MgldLinear<T>(inner, inner * 4 * 2));
        _ffOut = group.Add(new MgldLinear<T>(inner * 4, inner));
        _projOut = group.Add(new MgldLinear<T>(inner, channels, init: group.ZeroModuleInit));
    }

    public Tensor<T> Forward(IEngine engine, Tensor<T> x, Tensor<T> context)
    {
        int n = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        var tokens = engine.Reshape(engine.TensorPermute(_norm.Forward(x), new[] { 0, 2, 3, 1 }), new[] { n, h * w, c });
        var y = _projIn.Forward(tokens);
        y = engine.TensorAdd(y, _attn1.Forward(engine, _norm1.Forward(y)));
        y = engine.TensorAdd(y, _attn2.Forward(engine, _norm2.Forward(y), context));
        // GEGLU: the first half times GELU of the second.
        var projected = _ffIn.Forward(_norm3.Forward(y));
        int hidden = projected.Shape[2] / 2;
        var gated = engine.TensorMultiply(engine.TensorNarrow(projected, 2, 0, hidden), engine.GELU(engine.TensorNarrow(projected, 2, hidden, hidden)));
        y = engine.TensorAdd(y, _ffOut.Forward(gated));
        var image = engine.TensorPermute(engine.Reshape(_projOut.Forward(y), new[] { n, h, w, c }), new[] { 0, 3, 1, 2 });
        return engine.TensorAdd(image, x);
    }
}

/// <summary>
/// SPADE (Park et al. 2019) as StableSR uses it: GroupNorm32 of the features, modulated by γ and β from a 3x3 conv, ReLU
/// and two 3x3 convs on the structure condition of the same width: norm(x)·(1 + γ) + β.
/// </summary>
internal sealed class MgldSpade<T>
{
    private const int Hidden = 128;
    private readonly MgldGroupNorm<T> _norm;
    private readonly MgldConv2d<T> _shared, _gamma, _beta;

    public MgldSpade(LazyModuleGroup<T> group, int channels, int conditionChannels)
    {
        _norm = group.Add(new MgldGroupNorm<T>(channels, 1e-5));
        _shared = group.Add(new MgldConv2d<T>(conditionChannels, Hidden, 3, padding: 1));
        _gamma = group.Add(new MgldConv2d<T>(Hidden, channels, 3, padding: 1));
        _beta = group.Add(new MgldConv2d<T>(Hidden, channels, 3, padding: 1));
    }

    public Tensor<T> Forward(IEngine engine, Tensor<T> x, IReadOnlyDictionary<int, Tensor<T>> condition)
    {
        if (!condition.TryGetValue(x.Shape[3], out var segmap))
            throw new InvalidOperationException($"No structure condition is {x.Shape[3]} wide.");
        var actv = engine.ReLU(_shared.Forward(segmap));
        var scale = engine.TensorAddScalar(_gamma.Forward(actv), MathHelper.GetNumericOperations<T>().One);
        return engine.TensorAdd(engine.TensorMultiply(_norm.Forward(x), scale), _beta.Forward(actv));
    }
}

/// <summary>
/// openaimodel's <c>ResBlock</c> (or <c>ResBlockDual</c> with SPADE): GroupNorm32, SiLU and a 3x3 conv; the time
/// embedding through SiLU and a linear map added; GroupNorm32, SiLU and a zero-initialized 3x3 conv; SPADE on the
/// structure condition when dual; and the input (through a 1x1 conv when the width changes) added back. Only SPADE is
/// trained in a dual block.
/// </summary>
internal sealed class MgldResBlock<T>
{
    private readonly MgldGroupNorm<T> _inNorm, _outNorm;
    private readonly MgldConv2d<T> _inConv, _outConv;
    private readonly MgldConv2d<T>? _skip;
    private readonly MgldLinear<T> _emb;
    private readonly MgldSpade<T>? _spade;

    public MgldResBlock(MgldParts<T> parts, bool trained, int channels, int embedChannels, int outChannels, int? spadeChannels = null)
    {
        var group = parts.Of(trained);
        _inNorm = group.Add(new MgldGroupNorm<T>(channels, 1e-5));
        _inConv = group.Add(new MgldConv2d<T>(channels, outChannels, 3, padding: 1));
        _emb = group.Add(new MgldLinear<T>(embedChannels, outChannels));
        _outNorm = group.Add(new MgldGroupNorm<T>(outChannels, 1e-5));
        _outConv = group.Add(new MgldConv2d<T>(outChannels, outChannels, 3, padding: 1, init: group.ZeroModuleInit));
        if (channels != outChannels) _skip = group.Add(new MgldConv2d<T>(channels, outChannels, 1));
        if (spadeChannels is int s) _spade = new MgldSpade<T>(parts.Trained, outChannels, s);
    }

    /// <summary>[N, C, H, W] with the time embedding [N, E] and, for a dual block, the structure condition by width.</summary>
    public Tensor<T> Forward(IEngine engine, Tensor<T> x, Tensor<T> emb, IReadOnlyDictionary<int, Tensor<T>>? condition = null)
    {
        var h = _inConv.Forward(engine.Swish(_inNorm.Forward(x)));
        var e = _emb.Forward(engine.Swish(emb));
        h = engine.TensorAdd(h, engine.TensorBroadcastTo(engine.Reshape(e, new[] { e.Shape[0], e.Shape[1], 1, 1 }), h.Shape.ToArray()));
        h = _outConv.Forward(engine.Swish(_outNorm.Forward(h)));
        if (_spade is not null)
            h = _spade.Forward(engine, h, condition ?? throw new ArgumentNullException(nameof(condition), "A dual block needs the structure condition."));
        return engine.TensorAdd(_skip is null ? x : _skip.Forward(x), h);
    }
}

/// <summary>
/// openaimodel's <c>AttentionBlock</c> with the legacy head order: GroupNorm32, a 1x1 qkv projection split per head,
/// softmax(q·kᵀ/√d)·v, and a zero-initialized 1x1 projection added back.
/// </summary>
internal sealed class MgldAttentionBlock<T>
{
    private readonly int _heads;
    private readonly MgldGroupNorm<T> _norm;
    private readonly MgldConv2d<T> _qkv, _proj;

    public MgldAttentionBlock(LazyModuleGroup<T> group, int channels, int heads)
    {
        _heads = heads;
        _norm = group.Add(new MgldGroupNorm<T>(channels, 1e-5));
        _qkv = group.Add(new MgldConv2d<T>(channels, 3 * channels, 1));
        _proj = group.Add(new MgldConv2d<T>(channels, channels, 1, init: group.ZeroModuleInit));
    }

    public Tensor<T> Forward(IEngine engine, Tensor<T> x)
    {
        int n = x.Shape[0], c = x.Shape[1], l = x.Shape[2] * x.Shape[3], d = c / _heads;
        // QKVAttentionLegacy: [N, 3C, L] -> [N·H, 3d, L], then q, k, v of [N·H, d, L].
        var qkv = engine.Reshape(_qkv.Forward(_norm.Forward(x)), new[] { n * _heads, 3 * d, l });
        Tensor<T> Part(int i) => engine.TensorPermute(engine.TensorNarrow(qkv, 1, i * d, d), new[] { 0, 2, 1 });
        var attended = WanOps.Attend(engine, Part(0), Part(1), Part(2), null);
        var merged = engine.Reshape(engine.TensorPermute(attended, new[] { 0, 2, 1 }), new[] { n, c, x.Shape[2], x.Shape[3] });
        return engine.TensorAdd(x, _proj.Forward(merged));
    }
}

/// <summary>The time embedding: a sinusoid of the model width through Linear, SiLU and Linear to four times it.</summary>
internal sealed class MgldTimeEmbedding<T>
{
    private readonly int _channels;
    private readonly MgldLinear<T> _first, _second;

    public MgldTimeEmbedding(LazyModuleGroup<T> group, int channels)
    {
        _channels = channels;
        _first = group.Add(new MgldLinear<T>(channels, 4 * channels));
        _second = group.Add(new MgldLinear<T>(4 * channels, 4 * channels));
    }

    public Tensor<T> Forward(IEngine engine, IReadOnlyList<double> timesteps)
        => _second.Forward(engine.Swish(_first.Forward(MgldOps.TimestepEmbedding<T>(timesteps, _channels))));
}

/// <summary>
/// MGLD-VSR's denoiser (<c>InflatedUNetModelDualcondV2</c>): Stable Diffusion 2.1's UNet (320 wide, multipliers 1, 2, 4,
/// 4, two residual blocks a level, linear spatial transformers of 64-wide heads at the first three levels, cross-attending
/// to the 1,024-wide text context) whose residual blocks are SPADE-modulated by the structure condition, with a temporal
/// convolution, temporal attention and a second temporal convolution in the middle block. SD's weights are frozen; SPADE
/// and the temporal modules train.
/// </summary>
internal sealed class MgldUNet<T>
{
    private readonly int _channels;
    private readonly MgldTimeEmbedding<T> _time;
    private readonly MgldConv2d<T> _convIn, _convOut;
    private readonly MgldGroupNorm<T> _outNorm;
    private readonly List<(MgldResBlock<T>? Res, MgldSpatialTransformer<T>? Attn, MgldConv2d<T>? Down)> _input = new();
    private readonly MgldResBlock<T> _mid1, _mid2;
    private readonly MgldTemporalConv<T> _midConv1, _midConv2;
    private readonly MgldSpatialTransformer<T> _midAttn;
    private readonly MgldTemporalAttention<T> _midTemporal;
    private readonly List<(MgldResBlock<T> Res, MgldSpatialTransformer<T>? Attn, MgldConv2d<T>? Up)> _output = new();

    public MgldUNet(MgldParts<T> parts, int latentChannels, int channels, IReadOnlyList<int> multipliers, int resBlocks,
        int attentionLevels, int headChannels, int contextDim, int structureChannels)
    {
        _channels = channels;
        var frozen = parts.Frozen;
        int emb = 4 * channels;
        _time = new MgldTimeEmbedding<T>(frozen, channels);
        _convIn = frozen.Add(new MgldConv2d<T>(latentChannels, channels, 3, padding: 1));
        var skips = new List<int> { channels };
        int ch = channels;
        for (int level = 0; level < multipliers.Count; level++)
        {
            int width = channels * multipliers[level];
            for (int i = 0; i < resBlocks; i++)
            {
                var res = new MgldResBlock<T>(parts, false, ch, emb, width, structureChannels);
                ch = width;
                var attn = level < attentionLevels ? new MgldSpatialTransformer<T>(frozen, ch, ch / headChannels, headChannels, contextDim) : null;
                _input.Add((res, attn, null));
                skips.Add(ch);
            }
            if (level < multipliers.Count - 1)
            {
                _input.Add((null, null, frozen.Add(new MgldConv2d<T>(ch, ch, 3, stride: 2, padding: 1))));
                skips.Add(ch);
            }
        }
        _mid1 = new MgldResBlock<T>(parts, false, ch, emb, ch, structureChannels);
        _midConv1 = new MgldTemporalConv<T>(parts, ch);
        _midAttn = new MgldSpatialTransformer<T>(frozen, ch, ch / headChannels, headChannels, contextDim);
        _midTemporal = new MgldTemporalAttention<T>(parts, ch, ch / headChannels, headChannels);
        _mid2 = new MgldResBlock<T>(parts, false, ch, emb, ch, structureChannels);
        _midConv2 = new MgldTemporalConv<T>(parts, ch);
        for (int level = multipliers.Count - 1; level >= 0; level--)
        {
            int width = channels * multipliers[level];
            for (int i = 0; i <= resBlocks; i++)
            {
                int skip = skips[skips.Count - 1];
                skips.RemoveAt(skips.Count - 1);
                var res = new MgldResBlock<T>(parts, false, ch + skip, emb, width, structureChannels);
                ch = width;
                var attn = level < attentionLevels ? new MgldSpatialTransformer<T>(frozen, ch, ch / headChannels, headChannels, contextDim) : null;
                var up = level > 0 && i == resBlocks ? frozen.Add(new MgldConv2d<T>(ch, ch, 3, padding: 1)) : null;
                _output.Add((res, attn, up));
            }
        }
        _outNorm = frozen.Add(new MgldGroupNorm<T>(ch, 1e-5));
        _convOut = frozen.Add(new MgldConv2d<T>(ch, latentChannels, 3, padding: 1, init: frozen.ZeroModuleInit));
    }

    /// <summary>
    /// ε for noisy latents [B·T, 4, h, w] at per-frame timesteps, the text context [B·T, Lc, Dc] and the structure condition
    /// by width.
    /// </summary>
    public Tensor<T> Forward(IEngine engine, Tensor<T> x, IReadOnlyList<double> timesteps, Tensor<T> context,
        IReadOnlyDictionary<int, Tensor<T>> condition, int frames)
    {
        var emb = _time.Forward(engine, timesteps);
        var skips = new List<Tensor<T>>();
        var h = _convIn.Forward(x);
        skips.Add(h);
        foreach (var (res, attn, down) in _input)
        {
            if (down is not null) h = down.Forward(h);
            else if (res is not null)
            {
                h = res.Forward(engine, h, emb, condition);
                if (attn is not null) h = attn.Forward(engine, h, context);
            }
            skips.Add(h);
        }
        h = _midConv1.Forward(engine, _mid1.Forward(engine, h, emb, condition), frames);
        h = _midTemporal.Forward(engine, _midAttn.Forward(engine, h, context), frames);
        h = _midConv2.Forward(engine, _mid2.Forward(engine, h, emb, condition), frames);
        foreach (var (res, attn, up) in _output)
        {
            var skip = skips[skips.Count - 1];
            skips.RemoveAt(skips.Count - 1);
            h = res.Forward(engine, engine.TensorConcatenate(new[] { h, skip }, 1), emb, condition);
            if (attn is not null) h = attn.Forward(engine, h, context);
            if (up is not null) h = up.Forward(engine.Upsample(h, 2, 2));
        }
        return _convOut.Forward(engine.Swish(_outNorm.Forward(h)));
    }

    /// <summary>The model width the sinusoidal timestep embedding spans.</summary>
    public int Channels => _channels;
}

/// <summary>
/// StableSR's time-aware structure encoder (<c>InflatedEncoderUNetModelWT</c>): the LR latent through a UNet encoder
/// (256 wide, multipliers 1, 1, 2, 2, two residual blocks a level, four-head attention at the first three levels, 3x3
/// stride-2 downsampling) and a residual-attention-residual middle; the input of every downsampling and the middle's
/// output each pass a residual block to 256 channels, keyed by width. Trained throughout.
/// </summary>
internal sealed class MgldStructureEncoder<T>
{
    private readonly MgldTimeEmbedding<T> _time;
    private readonly MgldConv2d<T> _convIn;
    private readonly List<(MgldResBlock<T>? Res, MgldAttentionBlock<T>? Attn, MgldConv2d<T>? Down)> _blocks = new();
    private readonly MgldResBlock<T> _mid1, _mid2;
    private readonly MgldAttentionBlock<T> _midAttn;
    private readonly List<MgldResBlock<T>> _transforms = new();

    public MgldStructureEncoder(MgldParts<T> parts, int inChannels, int channels, int outChannels, IReadOnlyList<int> multipliers,
        int resBlocks, int attentionLevels, int heads)
    {
        var group = parts.Trained;
        int emb = 4 * channels;
        _time = new MgldTimeEmbedding<T>(group, channels);
        _convIn = group.Add(new MgldConv2d<T>(inChannels, channels, 3, padding: 1));
        var tapped = new List<int>();
        int ch = channels;
        for (int level = 0; level < multipliers.Count; level++)
        {
            for (int i = 0; i < resBlocks; i++)
            {
                var res = new MgldResBlock<T>(parts, true, ch, emb, channels * multipliers[level]);
                ch = channels * multipliers[level];
                _blocks.Add((res, level < attentionLevels ? new MgldAttentionBlock<T>(group, ch, heads) : null, null));
            }
            if (level < multipliers.Count - 1)
            {
                _blocks.Add((null, null, group.Add(new MgldConv2d<T>(ch, ch, 3, stride: 2, padding: 1))));
                tapped.Add(ch);
            }
        }
        _mid1 = new MgldResBlock<T>(parts, true, ch, emb, ch);
        _midAttn = new MgldAttentionBlock<T>(group, ch, heads);
        _mid2 = new MgldResBlock<T>(parts, true, ch, emb, ch);
        tapped.Add(ch);
        foreach (int width in tapped) _transforms.Add(new MgldResBlock<T>(parts, true, width, emb, outChannels));
    }

    /// <summary>The structure condition of LR latents [N, 4, h, w] at per-frame timesteps, keyed by width.</summary>
    public IReadOnlyDictionary<int, Tensor<T>> Forward(IEngine engine, Tensor<T> x, IReadOnlyList<double> timesteps)
    {
        var emb = _time.Forward(engine, timesteps);
        var taps = new List<Tensor<T>>();
        var h = _convIn.Forward(x);
        foreach (var (res, attn, down) in _blocks)
        {
            if (down is not null)
            {
                taps.Add(h);
                h = down.Forward(h);
                continue;
            }
            if (res is not null) h = res.Forward(engine, h, emb);
            if (attn is not null) h = attn.Forward(engine, h);
        }
        h = _mid2.Forward(engine, _midAttn.Forward(engine, _mid1.Forward(engine, h, emb)), emb);
        taps.Add(h);
        var result = new Dictionary<int, Tensor<T>>();
        for (int i = 0; i < taps.Count; i++) result[taps[i].Shape[3]] = _transforms[i].Forward(engine, taps[i], emb);
        return result;
    }
}
