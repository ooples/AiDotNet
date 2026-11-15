using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>Shared tensor helpers of the SyncDreamer blocks.</summary>
internal static class SdOps<T>
{
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Group normalization over [B, C, ...] of any rank (the spatial axes are folded into two).</summary>
    public static Tensor<T> GroupNorm(Tensor<T> x, int groups, Tensor<T> gamma, Tensor<T> beta, double epsilon)
    {
        if (x.Rank == 4) return Engine.GroupNorm(x, groups, gamma, beta, epsilon, out _, out _);
        var shape = x.Shape.ToArray();
        int last = shape[shape.Length - 1];
        int spatial = x.Length / (shape[0] * shape[1] * last);
        var folded = Engine.Reshape(x, new[] { shape[0], shape[1], spatial, last });
        return Engine.Reshape(Engine.GroupNorm(folded, groups, gamma, beta, epsilon, out _, out _), shape);
    }

    /// <summary>Broadcasts a per-sample vector [B, C] over the spatial axes of <paramref name="like"/> ([B, ., ...]).</summary>
    public static Tensor<T> BroadcastVector(Tensor<T> vector, Tensor<T> like)
    {
        var target = like.Shape.ToArray();
        var unit = new int[target.Length];
        for (int i = 0; i < unit.Length; i++) unit[i] = 1;
        unit[0] = vector.Shape[0];
        unit[1] = vector.Shape[1];
        target[1] = vector.Shape[1];
        return Engine.TensorBroadcastTo(Engine.Reshape(vector, unit), target);
    }

    /// <summary>Multiplies [B, C, ...] by a per-channel vector [C] and adds a per-channel vector [C].</summary>
    public static Tensor<T> ChannelAffine(Tensor<T> x, Tensor<T> scale, Tensor<T> shift)
    {
        var target = x.Shape.ToArray();
        var unit = new int[target.Length];
        for (int i = 0; i < unit.Length; i++) unit[i] = 1;
        unit[1] = scale.Length;
        var s = Engine.TensorBroadcastTo(Engine.Reshape(scale, unit), target);
        var b = Engine.TensorBroadcastTo(Engine.Reshape(shift, unit), target);
        return Engine.TensorAdd(Engine.TensorMultiply(x, s), b);
    }

    /// <summary>
    /// The latent-diffusion sinusoidal timestep embedding <c>timestep_embedding(t, dim)</c>: [cos(t f), sin(t f)] with
    /// <c>f_i = exp(-ln(10000) i / (dim / 2))</c>, as [1, dim].
    /// </summary>
    public static Tensor<T> TimestepEmbedding(int timestep, int dim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int half = dim / 2;
        var frequencies = Engine.TensorExp(Engine.TensorMultiplyScalar(
            Engine.TensorArange<T>(0.0, half, 1.0), ops.FromDouble(-Math.Log(10000.0) / half)));
        var phase = Engine.TensorMultiplyScalar(frequencies, ops.FromDouble(timestep));
        var embedding = Engine.TensorConcatenate(new[] { Engine.TensorCos(phase), Engine.TensorSin(phase) }, 0);
        return Engine.Reshape(embedding, new[] { 1, dim });
    }
}

/// <summary>nn.Linear over the last axis, weight [out, in].</summary>
internal sealed class SdLinear<T> : KlModule<T>
{
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T>? _bias;

    public SdLinear(int inFeatures, int outFeatures, bool bias = true)
        : this(inFeatures, outFeatures, LazyWeight<T>.Uniform(new[] { outFeatures, inFeatures }, inFeatures),
            bias ? LazyWeight<T>.Uniform(new[] { outFeatures }, inFeatures) : null)
    {
    }

    public SdLinear(int inFeatures, int outFeatures, LazyWeight<T> weight, LazyWeight<T>? bias)
    {
        if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
        if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _weight = weight;
        _bias = bias;
    }

    public int InFeatures { get; }

    public int OutFeatures { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int rows = x.Length / InFeatures;
        var y = Engine.TensorMatMul(Engine.Reshape(x, new[] { rows, InFeatures }), Engine.TensorTranspose(_weight.Value));
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

    /// <summary>
    /// Zero123's <c>cc_projection</c> initialization: the leading [out, out] block of the weight is the identity, the
    /// rest of the weight and the bias are zero, so the projected CLIP embedding starts as the embedding itself.
    /// </summary>
    public static SdLinear<T> IdentityInitialized(int inFeatures, int outFeatures)
    {
        var weight = new LazyWeight<T>(new[] { outFeatures, inFeatures }, (tensor, _) => FillIdentity(tensor, outFeatures, inFeatures));
        return new SdLinear<T>(inFeatures, outFeatures, weight, LazyWeight<T>.Constant(new[] { outFeatures }, 0.0));
    }

    private static void FillIdentity(Tensor<T> tensor, int rows, int columns)
    {
        var span = tensor.AsWritableSpan();
        T one = NumOps.One;
        int diagonal = Math.Min(rows, columns);
        for (int i = 0; i < diagonal; i++) span[i * columns + i] = one;
    }
}

/// <summary>nn.Conv2d or nn.Conv3d (by <c>dimensions</c>), with an optional bias.</summary>
internal sealed class SdConv<T> : KlModule<T>
{
    private readonly int _dimensions;
    private readonly int _stride;
    private readonly int _padding;
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T>? _bias;

    public SdConv(int dimensions, int inChannels, int outChannels, int kernel, int stride = 1, int padding = 0,
        bool bias = true, bool zeroInitialized = false)
    {
        if (dimensions != 2 && dimensions != 3) throw new ArgumentOutOfRangeException(nameof(dimensions));
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        _dimensions = dimensions;
        _stride = stride;
        _padding = padding;
        InChannels = inChannels;
        OutChannels = outChannels;
        int fanIn = inChannels * kernel * kernel * (dimensions == 3 ? kernel : 1);
        var shape = dimensions == 2
            ? new[] { outChannels, inChannels, kernel, kernel }
            : new[] { outChannels, inChannels, kernel, kernel, kernel };
        _weight = zeroInitialized ? LazyWeight<T>.Constant(shape, 0.0) : LazyWeight<T>.Uniform(shape, fanIn);
        if (bias) _bias = zeroInitialized ? LazyWeight<T>.Constant(new[] { outChannels }, 0.0) : LazyWeight<T>.Uniform(new[] { outChannels }, fanIn);
    }

    public int InChannels { get; }

    public int OutChannels { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = _dimensions == 2
            ? Engine.Conv2D(x, _weight.Value, new[] { _stride, _stride }, new[] { _padding, _padding }, new[] { 1, 1 })
            : Engine.Conv3D(x, _weight.Value, new[] { _stride, _stride, _stride }, new[] { _padding, _padding, _padding }, new[] { 1, 1, 1 });
        return _bias is null ? y : AddChannelBias(y, _bias.Value);
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

/// <summary>nn.ConvTranspose3d(in, out, 3, stride 2, padding 1, output_padding 1): doubles every spatial side.</summary>
internal sealed class SdConvTranspose3d<T> : KlModule<T>
{
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T> _bias;

    public SdConvTranspose3d(int inChannels, int outChannels)
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        // torch computes a transposed convolution's fan-in from weight.size(1) = out_channels.
        int fanIn = outChannels * 27;
        _weight = LazyWeight<T>.Uniform(new[] { inChannels, outChannels, 3, 3, 3 }, fanIn);
        _bias = LazyWeight<T>.Uniform(new[] { outChannels }, fanIn);
    }

    public Tensor<T> Forward(Tensor<T> x) => AddChannelBias(
        Engine.ConvTranspose3D(x, _weight.Value, new[] { 2, 2, 2 }, new[] { 1, 1, 1 }, new[] { 1, 1, 1 }), _bias.Value);

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "weight"), _weight.Value);
        yield return (Join(prefix, "bias"), _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>nn.GroupNorm(groups, channels) with torch's epsilon 1e-5, over [B, C, ...] of any rank.</summary>
internal sealed class SdGroupNorm<T> : KlModule<T>
{
    private const double Epsilon = 1e-5;
    private readonly int _groups;
    private readonly LazyWeight<T> _gamma;
    private readonly LazyWeight<T> _beta;

    public SdGroupNorm(int groups, int channels)
    {
        if (groups <= 0 || channels % groups != 0)
            throw new ArgumentException($"{channels} channels do not split into {groups} groups.", nameof(groups));
        _groups = groups;
        _gamma = LazyWeight<T>.Constant(new[] { channels }, 1.0);
        _beta = LazyWeight<T>.Constant(new[] { channels }, 0.0);
    }

    public Tensor<T> Forward(Tensor<T> x) => SdOps<T>.GroupNorm(x, _groups, _gamma.Value, _beta.Value, Epsilon);

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
/// nn.BatchNorm3d in evaluation mode, as sampling runs it: <c>(x - running_mean) / sqrt(running_var + 1e-5) * weight
/// + bias</c>. The running statistics are stored with the weights so a loaded checkpoint survives a clone.
/// </summary>
internal sealed class SdBatchNorm<T> : KlModule<T>
{
    private const double Epsilon = 1e-5;
    private readonly LazyWeight<T> _gamma;
    private readonly LazyWeight<T> _beta;
    private readonly LazyWeight<T> _mean;
    private readonly LazyWeight<T> _variance;

    public SdBatchNorm(int channels)
    {
        _gamma = LazyWeight<T>.Constant(new[] { channels }, 1.0);
        _beta = LazyWeight<T>.Constant(new[] { channels }, 0.0);
        _mean = LazyWeight<T>.Constant(new[] { channels }, 0.0);
        _variance = LazyWeight<T>.Constant(new[] { channels }, 1.0);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var scale = Engine.TensorDivide(_gamma.Value, Engine.TensorSqrt(Engine.TensorAddScalar(_variance.Value, NumOps.FromDouble(Epsilon))));
        var shift = Engine.TensorSubtract(_beta.Value, Engine.TensorMultiply(_mean.Value, scale));
        return SdOps<T>.ChannelAffine(x, scale, shift);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "weight"), _gamma.Value);
        yield return (Join(prefix, "bias"), _beta.Value);
        yield return (Join(prefix, "running_mean"), _mean.Value);
        yield return (Join(prefix, "running_var"), _variance.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _gamma;
        yield return _beta;
        yield return _mean;
        yield return _variance;
    }
}

/// <summary>nn.LayerNorm over the last axis with epsilon 1e-5.</summary>
internal sealed class SdLayerNorm<T> : KlModule<T>
{
    private readonly LazyWeight<T> _gamma;
    private readonly LazyWeight<T> _beta;

    public SdLayerNorm(int width)
    {
        _gamma = LazyWeight<T>.Constant(new[] { width }, 1.0);
        _beta = LazyWeight<T>.Constant(new[] { width }, 0.0);
    }

    public Tensor<T> Forward(Tensor<T> x) => Engine.LayerNorm(x, _gamma.Value, _beta.Value, 1e-5, out _, out _);

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
/// <c>Image2DResBlockWithTV</c>: <c>x + conv(x + time_embed(t) + view_embed(v))</c>, the convolution being group norm,
/// SiLU, 3x3 conv, group norm, SiLU, 3x3 conv, and the embeddings 1x1 convolutions of the timestep and viewpoint.
/// </summary>
internal sealed class SdImageResBlockTV<T> : KlModule<T>
{
    private readonly SdConv<T> _timeEmbed;
    private readonly SdConv<T> _viewEmbed;
    private readonly SdGroupNorm<T> _norm1;
    private readonly SdConv<T> _conv1;
    private readonly SdGroupNorm<T> _norm2;
    private readonly SdConv<T> _conv2;

    public SdImageResBlockTV(int dim, int timeDim, int viewDim, int groups)
    {
        _timeEmbed = new SdConv<T>(2, timeDim, dim, 1);
        _viewEmbed = new SdConv<T>(2, viewDim, dim, 1);
        _norm1 = new SdGroupNorm<T>(groups, dim);
        _conv1 = new SdConv<T>(2, dim, dim, 3, padding: 1);
        _norm2 = new SdGroupNorm<T>(groups, dim);
        _conv2 = new SdConv<T>(2, dim, dim, 3, padding: 1);
    }

    /// <param name="x">[B, dim, H, W].</param>
    /// <param name="t">[B, timeDim, 1, 1].</param>
    /// <param name="v">[B, viewDim, 1, 1].</param>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> t, Tensor<T> v)
    {
        var condition = Engine.TensorAdd(_timeEmbed.Forward(t), _viewEmbed.Forward(v));
        var h = Engine.TensorAdd(x, Engine.TensorBroadcastTo(condition, x.Shape.ToArray()));
        h = _conv1.Forward(Engine.Swish(_norm1.Forward(h)));
        h = _conv2.Forward(Engine.Swish(_norm2.Forward(h)));
        return Engine.TensorAdd(x, h);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) =>
        _timeEmbed.NamedTensors(Join(prefix, "time_embed"))
            .Concat(_viewEmbed.NamedTensors(Join(prefix, "view_embed")))
            .Concat(_norm1.NamedTensors(Join(prefix, "conv.0")))
            .Concat(_conv1.NamedTensors(Join(prefix, "conv.2")))
            .Concat(_norm2.NamedTensors(Join(prefix, "conv.3")))
            .Concat(_conv2.NamedTensors(Join(prefix, "conv.5")));

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _timeEmbed;
        yield return _viewEmbed;
        yield return _norm1;
        yield return _conv1;
        yield return _norm2;
        yield return _conv2;
    }
}

/// <summary>
/// <c>NoisyTargetViewEncoder</c>: a 3x3 conv from the 4 latent channels, three timestep- and viewpoint-conditioned
/// residual blocks, then group norm, SiLU and a 3x3 conv to the feature width that is unprojected into the volume.
/// </summary>
internal sealed class SdNoisyTargetViewEncoder<T> : KlModule<T>
{
    private readonly SdConv<T> _initConv;
    private readonly SdImageResBlockTV<T>[] _blocks;
    private readonly SdGroupNorm<T> _finalNorm;
    private readonly SdConv<T> _finalConv;

    public SdNoisyTargetViewEncoder(int latentChannels, int timeDim, int viewDim, int width, int outputDim, int groups)
    {
        _initConv = new SdConv<T>(2, latentChannels, width, 3, padding: 1);
        _blocks = new[]
        {
            new SdImageResBlockTV<T>(width, timeDim, viewDim, groups),
            new SdImageResBlockTV<T>(width, timeDim, viewDim, groups),
            new SdImageResBlockTV<T>(width, timeDim, viewDim, groups),
        };
        _finalNorm = new SdGroupNorm<T>(groups, width);
        _finalConv = new SdConv<T>(2, width, outputDim, 3, padding: 1);
        OutputDim = outputDim;
    }

    public int OutputDim { get; }

    /// <param name="x">Noisy latents [B, 4, H, W].</param>
    /// <param name="t">Timestep embeddings [B, timeDim].</param>
    /// <param name="v">Viewpoint embeddings [B, viewDim].</param>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> t, Tensor<T> v)
    {
        var t4 = Engine.Reshape(t, new[] { t.Shape[0], t.Shape[1], 1, 1 });
        var v4 = Engine.Reshape(v, new[] { v.Shape[0], v.Shape[1], 1, 1 });
        var h = _initConv.Forward(x);
        foreach (var block in _blocks) h = block.Forward(h, t4, v4);
        return _finalConv.Forward(Engine.Swish(_finalNorm.Forward(h)));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        var all = _initConv.NamedTensors(Join(prefix, "init_conv"));
        for (int i = 0; i < _blocks.Length; i++) all = all.Concat(_blocks[i].NamedTensors(Join(prefix, "out_conv" + i)));
        return all.Concat(_finalNorm.NamedTensors(Join(prefix, "final_out.0")))
            .Concat(_finalConv.NamedTensors(Join(prefix, "final_out.2")));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _initConv;
        foreach (var block in _blocks) yield return block;
        yield return _finalNorm;
        yield return _finalConv;
    }
}

/// <summary>
/// <c>SpatialTimeBlock</c> and <c>FrustumTVBlock</c>: group norm, SiLU, the conditioning vectors broadcast over the
/// volume and concatenated as channels, then a 3x3x3 convolution with the given stride.
/// </summary>
internal sealed class SdConditionedVolumeBlock<T> : KlModule<T>
{
    private readonly SdGroupNorm<T> _norm;
    private readonly SdConv<T> _conv;

    public SdConditionedVolumeBlock(int inChannels, int conditionChannels, int outChannels, int stride, int groups)
    {
        _norm = new SdGroupNorm<T>(groups, inChannels);
        _conv = new SdConv<T>(3, inChannels + conditionChannels, outChannels, 3, stride, 1);
    }

    /// <param name="x">[B, C, D, H, W].</param>
    /// <param name="conditions">Per-sample vectors [B, c], concatenated in order after the features.</param>
    public Tensor<T> Forward(Tensor<T> x, params Tensor<T>[] conditions)
    {
        var h = Engine.Swish(_norm.Forward(x));
        var parts = new Tensor<T>[conditions.Length + 1];
        parts[0] = h;
        for (int i = 0; i < conditions.Length; i++) parts[i + 1] = SdOps<T>.BroadcastVector(conditions[i], h);
        return _conv.Forward(Engine.TensorConcatenate(parts, 1));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) =>
        _norm.NamedTensors(Join(prefix, "bn")).Concat(_conv.NamedTensors(Join(prefix, "conv")));

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _norm;
        yield return _conv;
    }
}

/// <summary>A 3D upsampling step: transposed convolution, batch norm, ReLU.</summary>
internal sealed class SdVolumeUp<T> : KlModule<T>
{
    private readonly SdConvTranspose3d<T> _deconv;
    private readonly SdBatchNorm<T> _norm;

    public SdVolumeUp(int inChannels, int outChannels)
    {
        _deconv = new SdConvTranspose3d<T>(inChannels, outChannels);
        _norm = new SdBatchNorm<T>(outChannels);
    }

    public Tensor<T> Forward(Tensor<T> x) => Engine.ReLU(_norm.Forward(_deconv.Forward(x)));

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) =>
        _deconv.NamedTensors(Join(prefix, "0")).Concat(_norm.NamedTensors(Join(prefix, "1")));

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _deconv;
        yield return _norm;
    }
}

/// <summary>
/// <c>SpatialTime3DNet</c>: the 3D U-Net that turns the unprojected per-view features into the spatial feature volume.
/// Three stride-2 stages of timestep-conditioned blocks down to 1/8 resolution, then transposed convolutions back up
/// with additive skips; the output has <c>dims[0]</c> channels at the input resolution.
/// </summary>
internal sealed class SdSpatialTime3DNet<T> : KlModule<T>
{
    private readonly SdConv<T> _initConv;
    private readonly SdConditionedVolumeBlock<T> _conv0, _conv1, _conv2a, _conv2b, _conv3, _conv4a, _conv4b, _conv5, _conv6a, _conv6b;
    private readonly SdVolumeUp<T> _conv7, _conv8, _conv9;

    public SdSpatialTime3DNet(int inputDim, int timeDim, IReadOnlyList<int> dims, int groups)
    {
        int d0 = dims[0], d1 = dims[1], d2 = dims[2], d3 = dims[3];
        _initConv = new SdConv<T>(3, inputDim, d0, 3, 1, 1);
        _conv0 = new SdConditionedVolumeBlock<T>(d0, timeDim, d0, 1, groups);
        _conv1 = new SdConditionedVolumeBlock<T>(d0, timeDim, d1, 2, groups);
        _conv2a = new SdConditionedVolumeBlock<T>(d1, timeDim, d1, 1, groups);
        _conv2b = new SdConditionedVolumeBlock<T>(d1, timeDim, d1, 1, groups);
        _conv3 = new SdConditionedVolumeBlock<T>(d1, timeDim, d2, 2, groups);
        _conv4a = new SdConditionedVolumeBlock<T>(d2, timeDim, d2, 1, groups);
        _conv4b = new SdConditionedVolumeBlock<T>(d2, timeDim, d2, 1, groups);
        _conv5 = new SdConditionedVolumeBlock<T>(d2, timeDim, d3, 2, groups);
        _conv6a = new SdConditionedVolumeBlock<T>(d3, timeDim, d3, 1, groups);
        _conv6b = new SdConditionedVolumeBlock<T>(d3, timeDim, d3, 1, groups);
        _conv7 = new SdVolumeUp<T>(d3, d2);
        _conv8 = new SdVolumeUp<T>(d2, d1);
        _conv9 = new SdVolumeUp<T>(d1, d0);
    }

    /// <param name="x">[B, inputDim, V, V, V].</param>
    /// <param name="t">[B, timeDim].</param>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> t)
    {
        x = _initConv.Forward(x);
        var c0 = _conv0.Forward(x, t);
        x = _conv1.Forward(c0, t);
        x = _conv2a.Forward(x, t);
        var c2 = _conv2b.Forward(x, t);
        x = _conv3.Forward(c2, t);
        x = _conv4a.Forward(x, t);
        var c4 = _conv4b.Forward(x, t);
        x = _conv5.Forward(c4, t);
        x = _conv6a.Forward(x, t);
        x = _conv6b.Forward(x, t);
        x = Engine.TensorAdd(c4, _conv7.Forward(x));
        x = Engine.TensorAdd(c2, _conv8.Forward(x));
        return Engine.TensorAdd(c0, _conv9.Forward(x));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) =>
        _initConv.NamedTensors(Join(prefix, "init_conv"))
            .Concat(_conv0.NamedTensors(Join(prefix, "conv0")))
            .Concat(_conv1.NamedTensors(Join(prefix, "conv1")))
            .Concat(_conv2a.NamedTensors(Join(prefix, "conv2_0")))
            .Concat(_conv2b.NamedTensors(Join(prefix, "conv2_1")))
            .Concat(_conv3.NamedTensors(Join(prefix, "conv3")))
            .Concat(_conv4a.NamedTensors(Join(prefix, "conv4_0")))
            .Concat(_conv4b.NamedTensors(Join(prefix, "conv4_1")))
            .Concat(_conv5.NamedTensors(Join(prefix, "conv5")))
            .Concat(_conv6a.NamedTensors(Join(prefix, "conv6_0")))
            .Concat(_conv6b.NamedTensors(Join(prefix, "conv6_1")))
            .Concat(_conv7.NamedTensors(Join(prefix, "conv7")))
            .Concat(_conv8.NamedTensors(Join(prefix, "conv8")))
            .Concat(_conv9.NamedTensors(Join(prefix, "conv9")));

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _initConv;
        yield return _conv0;
        yield return _conv1;
        yield return _conv2a;
        yield return _conv2b;
        yield return _conv3;
        yield return _conv4a;
        yield return _conv4b;
        yield return _conv5;
        yield return _conv6a;
        yield return _conv6b;
        yield return _conv7;
        yield return _conv8;
        yield return _conv9;
    }
}

/// <summary>
/// <c>FrustumTV3DNet</c>: the 3D U-Net over one view's frustum volume, conditioned on the timestep and the viewpoint.
/// It returns the volume at four scales (full, 1/2, 1/4, 1/8 of the latent side), the contexts of the U-Net's
/// depth-wise attention at those resolutions.
/// </summary>
internal sealed class SdFrustumTV3DNet<T> : KlModule<T>
{
    private readonly SdConv<T> _conv0;
    private readonly SdConditionedVolumeBlock<T> _conv1, _conv2, _conv3, _conv4, _conv5, _conv6;
    private readonly SdVolumeUp<T> _up0, _up1, _up2;

    public SdFrustumTV3DNet(int inputDim, int timeDim, int viewDim, IReadOnlyList<int> dims, int groups)
    {
        int c = timeDim + viewDim;
        _conv0 = new SdConv<T>(3, inputDim, dims[0], 3, 1, 1);
        _conv1 = new SdConditionedVolumeBlock<T>(dims[0], c, dims[1], 2, groups);
        _conv2 = new SdConditionedVolumeBlock<T>(dims[1], c, dims[1], 1, groups);
        _conv3 = new SdConditionedVolumeBlock<T>(dims[1], c, dims[2], 2, groups);
        _conv4 = new SdConditionedVolumeBlock<T>(dims[2], c, dims[2], 1, groups);
        _conv5 = new SdConditionedVolumeBlock<T>(dims[2], c, dims[3], 2, groups);
        _conv6 = new SdConditionedVolumeBlock<T>(dims[3], c, dims[3], 1, groups);
        _up0 = new SdVolumeUp<T>(dims[3], dims[2]);
        _up1 = new SdVolumeUp<T>(dims[2], dims[1]);
        _up2 = new SdVolumeUp<T>(dims[1], dims[0]);
    }

    /// <param name="x">[B, inputDim, D, S, S].</param>
    /// <param name="t">[B, timeDim].</param>
    /// <param name="v">[B, viewDim].</param>
    /// <returns>The four scales, finest first.</returns>
    public Tensor<T>[] Forward(Tensor<T> x, Tensor<T> t, Tensor<T> v)
    {
        var x0 = _conv0.Forward(x);
        var x1 = _conv2.Forward(_conv1.Forward(x0, t, v), t, v);
        var x2 = _conv4.Forward(_conv3.Forward(x1, t, v), t, v);
        var x3 = _conv6.Forward(_conv5.Forward(x2, t, v), t, v);
        x2 = Engine.TensorAdd(_up0.Forward(x3), x2);
        x1 = Engine.TensorAdd(_up1.Forward(x2), x1);
        x0 = Engine.TensorAdd(_up2.Forward(x1), x0);
        return new[] { x0, x1, x2, x3 };
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) =>
        _conv0.NamedTensors(Join(prefix, "conv0"))
            .Concat(_conv1.NamedTensors(Join(prefix, "conv1")))
            .Concat(_conv2.NamedTensors(Join(prefix, "conv2")))
            .Concat(_conv3.NamedTensors(Join(prefix, "conv3")))
            .Concat(_conv4.NamedTensors(Join(prefix, "conv4")))
            .Concat(_conv5.NamedTensors(Join(prefix, "conv5")))
            .Concat(_conv6.NamedTensors(Join(prefix, "conv6")))
            .Concat(_up0.NamedTensors(Join(prefix, "up0")))
            .Concat(_up1.NamedTensors(Join(prefix, "up1")))
            .Concat(_up2.NamedTensors(Join(prefix, "up2")));

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _conv0;
        yield return _conv1;
        yield return _conv2;
        yield return _conv3;
        yield return _conv4;
        yield return _conv5;
        yield return _conv6;
        yield return _up0;
        yield return _up1;
        yield return _up2;
    }
}

/// <summary>
/// <c>DepthAttention</c>: every pixel of the U-Net feature map attends only to the frustum features along its own ray.
/// <c>q = to_q(x)</c> ([b, h, d_h, H, W]), <c>k, v = to_k(c), to_v(c)</c> ([b, h, d_h, D, H, W]),
/// <c>a = softmax_D(sum_{d_h} q k / sqrt(d_h))</c>, <c>out = to_out(sum_D v a)</c>.
/// </summary>
internal sealed class SdDepthAttention<T> : KlModule<T>
{
    private readonly int _heads;
    private readonly int _headDim;
    private readonly SdConv<T> _toQ;
    private readonly SdConv<T> _toK;
    private readonly SdConv<T> _toV;
    private readonly SdConv<T> _toOut;

    public SdDepthAttention(int queryDim, int contextDim, int heads, int headDim, bool outputBias)
    {
        _heads = heads;
        _headDim = headDim;
        int inner = heads * headDim;
        _toQ = new SdConv<T>(2, queryDim, inner, 1, bias: false);
        _toK = new SdConv<T>(3, contextDim, inner, 1, bias: false);
        _toV = new SdConv<T>(3, contextDim, inner, 1, bias: false);
        _toOut = new SdConv<T>(2, inner, queryDim, 1, bias: outputBias);
    }

    /// <param name="x">[b, queryDim, H, W].</param>
    /// <param name="context">[b, contextDim, D, H, W].</param>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> context)
    {
        int b = x.Shape[0], h = x.Shape[2], w = x.Shape[3], d = context.Shape[2];
        if (context.Shape[0] != b || context.Shape[3] != h || context.Shape[4] != w)
            throw new ArgumentException(
                $"The frustum context [{string.Join(", ", context.Shape.ToArray())}] does not match the feature map " +
                $"[{string.Join(", ", x.Shape.ToArray())}].", nameof(context));
        var full = new[] { b, _heads, _headDim, d, h, w };
        var q = Engine.TensorBroadcastTo(Engine.Reshape(_toQ.Forward(x), new[] { b, _heads, _headDim, 1, h, w }), full);
        var k = Engine.Reshape(_toK.Forward(context), full);
        var v = Engine.Reshape(_toV.Forward(context), full);
        var similarity = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(Engine.TensorMultiply(q, k), new[] { 2 }, false), NumOps.FromDouble(1.0 / Math.Sqrt(_headDim)));
        var attention = Engine.Softmax(similarity, 2);                                                  // [b, h, D, H, W]
        var weights = Engine.TensorBroadcastTo(Engine.Reshape(attention, new[] { b, _heads, 1, d, h, w }), full);
        var output = Engine.ReduceSum(Engine.TensorMultiply(v, weights), new[] { 3 }, false);          // [b, h, d_h, H, W]
        return _toOut.Forward(Engine.Reshape(output, new[] { b, _heads * _headDim, h, w }));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) =>
        _toQ.NamedTensors(Join(prefix, "to_q"))
            .Concat(_toK.NamedTensors(Join(prefix, "to_k")))
            .Concat(_toV.NamedTensors(Join(prefix, "to_v")))
            .Concat(_toOut.NamedTensors(Join(prefix, "to_out")));

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _toQ;
        yield return _toK;
        yield return _toV;
        yield return _toOut;
    }
}

/// <summary>
/// <c>DepthTransformer</c>: <c>x + proj_out(depth_attn(proj_in(x), proj_context(c)))</c>, with proj_in a 1x1 conv,
/// group norm and SiLU; proj_context a bias-free 1x1x1 conv, group norm and ReLU; and proj_out group norm, ReLU, a
/// bias-free 3x3 conv, group norm, ReLU and a zero-initialized bias-free 3x3 conv, so the block starts as the identity.
/// </summary>
internal sealed class SdDepthTransformer<T> : KlModule<T>
{
    private readonly SdConv<T> _projIn;
    private readonly SdGroupNorm<T> _projInNorm;
    private readonly SdConv<T> _projContext;
    private readonly SdGroupNorm<T> _projContextNorm;
    private readonly SdDepthAttention<T> _attention;
    private readonly SdGroupNorm<T> _outNorm1;
    private readonly SdConv<T> _outConv1;
    private readonly SdGroupNorm<T> _outNorm2;
    private readonly SdConv<T> _outConv2;

    public SdDepthTransformer(int dim, int heads, int headDim, int contextDim, int groups)
    {
        int inner = heads * headDim;
        Dim = dim;
        ContextDim = contextDim;
        _projIn = new SdConv<T>(2, dim, inner, 1);
        _projInNorm = new SdGroupNorm<T>(groups, inner);
        _projContext = new SdConv<T>(3, contextDim, contextDim, 1, bias: false);
        _projContextNorm = new SdGroupNorm<T>(groups, contextDim);
        _attention = new SdDepthAttention<T>(inner, contextDim, heads, headDim, outputBias: false);
        _outNorm1 = new SdGroupNorm<T>(groups, inner);
        _outConv1 = new SdConv<T>(2, inner, inner, 3, padding: 1, bias: false);
        _outNorm2 = new SdGroupNorm<T>(groups, inner);
        _outConv2 = new SdConv<T>(2, inner, dim, 3, padding: 1, bias: false, zeroInitialized: true);
    }

    public int Dim { get; }

    public int ContextDim { get; }

    public SdDepthAttention<T> Attention => _attention;

    /// <param name="x">[b, dim, H, W].</param>
    /// <param name="context">[b, contextDim, D, H, W].</param>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> context)
    {
        var h = Engine.Swish(_projInNorm.Forward(_projIn.Forward(x)));
        var c = Engine.ReLU(_projContextNorm.Forward(_projContext.Forward(context)));
        h = _attention.Forward(h, c);
        h = _outConv1.Forward(Engine.ReLU(_outNorm1.Forward(h)));
        h = _outConv2.Forward(Engine.ReLU(_outNorm2.Forward(h)));
        return Engine.TensorAdd(h, x);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) =>
        _projIn.NamedTensors(Join(prefix, "proj_in.0"))
            .Concat(_projInNorm.NamedTensors(Join(prefix, "proj_in.1")))
            .Concat(_projContext.NamedTensors(Join(prefix, "proj_context.0")))
            .Concat(_projContextNorm.NamedTensors(Join(prefix, "proj_context.1")))
            .Concat(_attention.NamedTensors(Join(prefix, "depth_attn")))
            .Concat(_outNorm1.NamedTensors(Join(prefix, "proj_out.0")))
            .Concat(_outConv1.NamedTensors(Join(prefix, "proj_out.2")))
            .Concat(_outNorm2.NamedTensors(Join(prefix, "proj_out.3")))
            .Concat(_outConv2.NamedTensors(Join(prefix, "proj_out.5")));

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _projIn;
        yield return _projInNorm;
        yield return _projContext;
        yield return _projContextNorm;
        yield return _attention;
        yield return _outNorm1;
        yield return _outConv1;
        yield return _outNorm2;
        yield return _outConv2;
    }
}

/// <summary>
/// One pre-norm block of OpenAI CLIP's vision transformer: <c>x + attn(ln_1(x))</c>, then <c>x + mlp(ln_2(x))</c>, with
/// fused query-key-value projection weights and a QuickGELU MLP of width 4x.
/// </summary>
internal sealed class SdClipBlock<T> : KlModule<T>
{
    private readonly int _width;
    private readonly int _heads;
    private readonly SdLayerNorm<T> _ln1;
    private readonly LazyWeight<T> _inProjWeight;
    private readonly LazyWeight<T> _inProjBias;
    private readonly SdLinear<T> _outProj;
    private readonly SdLayerNorm<T> _ln2;
    private readonly SdLinear<T> _fc;
    private readonly SdLinear<T> _proj;

    public SdClipBlock(int width, int heads)
    {
        if (width % heads != 0) throw new ArgumentException("The CLIP width must divide by the head count.", nameof(heads));
        _width = width;
        _heads = heads;
        _ln1 = new SdLayerNorm<T>(width);
        _inProjWeight = LazyWeight<T>.Uniform(new[] { 3 * width, width }, width);
        _inProjBias = LazyWeight<T>.Constant(new[] { 3 * width }, 0.0);
        _outProj = new SdLinear<T>(width, width);
        _ln2 = new SdLayerNorm<T>(width);
        _fc = new SdLinear<T>(width, 4 * width);
        _proj = new SdLinear<T>(4 * width, width);
    }

    /// <param name="x">[B, N, width].</param>
    public Tensor<T> Forward(Tensor<T> x)
    {
        int b = x.Shape[0], n = x.Shape[1], hd = _width / _heads;
        var normed = Engine.Reshape(_ln1.Forward(x), new[] { b * n, _width });
        var qkv = Engine.TensorAdd(Engine.TensorMatMul(normed, Engine.TensorTranspose(_inProjWeight.Value)),
            Engine.TensorBroadcastTo(Engine.Reshape(_inProjBias.Value, new[] { 1, 3 * _width }), new[] { b * n, 3 * _width }));
        var split = Engine.TensorPermute(Engine.Reshape(qkv, new[] { b, n, 3, _heads, hd }), new[] { 2, 0, 3, 1, 4 });
        Tensor<T> Part(int index) => Engine.Reshape(
            Engine.TensorSlice(split, new[] { index, 0, 0, 0, 0 }, new[] { 1, b, _heads, n, hd }), new[] { b * _heads, n, hd });
        var q = Engine.TensorMultiplyScalar(Part(0), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var k = Part(1);
        var v = Part(2);
        var probabilities = Engine.Softmax(Engine.BatchMatMul(q, Engine.TensorPermute(k, new[] { 0, 2, 1 })), 2);
        var context = Engine.Reshape(Engine.TensorPermute(
            Engine.Reshape(Engine.BatchMatMul(probabilities, v), new[] { b, _heads, n, hd }), new[] { 0, 2, 1, 3 }),
            new[] { b, n, _width });
        x = Engine.TensorAdd(x, _outProj.Forward(context));
        var hidden = _fc.Forward(_ln2.Forward(x));
        // QuickGELU: x * sigmoid(1.702 x).
        hidden = Engine.TensorMultiply(hidden, Engine.Sigmoid(Engine.TensorMultiplyScalar(hidden, NumOps.FromDouble(1.702))));
        return Engine.TensorAdd(x, _proj.Forward(hidden));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        foreach (var t in _ln1.NamedTensors(Join(prefix, "ln_1"))) yield return t;
        yield return (Join(prefix, "attn.in_proj_weight"), _inProjWeight.Value);
        yield return (Join(prefix, "attn.in_proj_bias"), _inProjBias.Value);
        foreach (var t in _outProj.NamedTensors(Join(prefix, "attn.out_proj"))) yield return t;
        foreach (var t in _ln2.NamedTensors(Join(prefix, "ln_2"))) yield return t;
        foreach (var t in _fc.NamedTensors(Join(prefix, "mlp.c_fc"))) yield return t;
        foreach (var t in _proj.NamedTensors(Join(prefix, "mlp.c_proj"))) yield return t;
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _inProjWeight;
        yield return _inProjBias;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _ln1;
        yield return _outProj;
        yield return _ln2;
        yield return _fc;
        yield return _proj;
    }
}

/// <summary>
/// OpenAI CLIP's image encoder (<c>VisionTransformer.forward</c>, Radford et al. 2021): a bias-free patch convolution,
/// a class token and position embeddings, <c>ln_pre</c>, the transformer, <c>ln_post</c> on the class token and the
/// projection to the joint embedding (<c>encode_image</c>). Parameter names follow the OpenAI checkpoint.
/// </summary>
internal sealed class SdClipImageEncoder<T> : KlModule<T>
{
    private static readonly double[] Mean = { 0.48145466, 0.4578275, 0.40821073 };
    private static readonly double[] Std = { 0.26862954, 0.26130258, 0.27577711 };
    private readonly int _imageSize;
    private readonly int _patch;
    private readonly int _width;
    private readonly LazyWeight<T> _conv;
    private readonly LazyWeight<T> _classEmbedding;
    private readonly LazyWeight<T> _positionalEmbedding;
    private readonly SdLayerNorm<T> _lnPre;
    private readonly SdClipBlock<T>[] _blocks;
    private readonly SdLayerNorm<T> _lnPost;
    private readonly LazyWeight<T> _projection;

    public SdClipImageEncoder(int imageSize, int patchSize, int width, int layers, int heads, int embedDim)
    {
        if (imageSize % patchSize != 0) throw new ArgumentException("The CLIP image side must divide by the patch side.", nameof(patchSize));
        _imageSize = imageSize;
        _patch = patchSize;
        _width = width;
        int grid = imageSize / patchSize;
        _conv = LazyWeight<T>.Uniform(new[] { width, 3, patchSize, patchSize }, 3 * patchSize * patchSize);
        _classEmbedding = LazyWeight<T>.Uniform(new[] { width }, width);
        _positionalEmbedding = LazyWeight<T>.Uniform(new[] { grid * grid + 1, width }, width);
        _lnPre = new SdLayerNorm<T>(width);
        _blocks = new SdClipBlock<T>[layers];
        for (int i = 0; i < layers; i++) _blocks[i] = new SdClipBlock<T>(width, heads);
        _lnPost = new SdLayerNorm<T>(width);
        _projection = LazyWeight<T>.Uniform(new[] { width, embedDim }, width);
        EmbedDim = embedDim;
    }

    public int EmbedDim { get; }

    /// <summary>
    /// <c>FrozenCLIPImageEmbedder.preprocess</c> of images [B, 3, H, W] in [0, 1]: kornia's bicubic resize to the CLIP
    /// side with <c>align_corners=True</c> and <c>antialias=False</c> (the embedder's default), then CLIP's mean and std.
    /// The reference resizes the [-1, 1] image and maps it to [0, 1] afterwards; the resize is linear, so the order does
    /// not change the result.
    /// </summary>
    public Tensor<T> Preprocess(Tensor<T> images)
    {
        var x = images.Shape[2] == _imageSize && images.Shape[3] == _imageSize
            ? images
            : Engine.Interpolate(images, new[] { _imageSize, _imageSize }, InterpolateMode.Bicubic, true);
        var mean = new Tensor<double>(Mean, new[] { 3 }).Cast<T>();
        var inverseStd = new Tensor<double>(new[] { 1.0 / Std[0], 1.0 / Std[1], 1.0 / Std[2] }, new[] { 3 }).Cast<T>();
        return SdOps<T>.ChannelAffine(Engine.TensorSubtract(x, Engine.TensorBroadcastTo(Engine.Reshape(mean, new[] { 1, 3, 1, 1 }), x.Shape.ToArray())),
            inverseStd, new Tensor<T>(new[] { 3 }));
    }

    /// <summary>
    /// The CLIP image embedding [B, embedDim] of images [B, 3, H, W] in [0, 1]: <see cref="Preprocess"/>, then the
    /// transformer.
    /// </summary>
    public Tensor<T> Encode(Tensor<T> images)
    {
        int b = images.Shape[0];
        var x = Preprocess(images);
        int grid = _imageSize / _patch, patches = grid * grid;
        var tokens = Engine.Conv2D(x, _conv.Value, new[] { _patch, _patch }, new[] { 0, 0 }, new[] { 1, 1 });   // [B, W, g, g]
        tokens = Engine.TensorPermute(Engine.Reshape(tokens, new[] { b, _width, patches }), new[] { 0, 2, 1 }); // [B, P, W]
        var cls = Engine.TensorBroadcastTo(Engine.Reshape(_classEmbedding.Value, new[] { 1, 1, _width }), new[] { b, 1, _width });
        var h = Engine.TensorConcatenate(new[] { cls, tokens }, 1);
        h = Engine.TensorAdd(h, Engine.TensorBroadcastTo(Engine.Reshape(_positionalEmbedding.Value, new[] { 1, patches + 1, _width }), h.Shape.ToArray()));
        h = _lnPre.Forward(h);
        foreach (var block in _blocks) h = block.Forward(h);
        var pooled = _lnPost.Forward(Engine.Reshape(Engine.TensorSlice(h, new[] { 0, 0, 0 }, new[] { b, 1, _width }), new[] { b, _width }));
        return Engine.TensorMatMul(pooled, _projection.Value);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "conv1.weight"), _conv.Value);
        yield return (Join(prefix, "class_embedding"), _classEmbedding.Value);
        yield return (Join(prefix, "positional_embedding"), _positionalEmbedding.Value);
        foreach (var t in _lnPre.NamedTensors(Join(prefix, "ln_pre"))) yield return t;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (var t in _blocks[i].NamedTensors(Join(prefix, "transformer.resblocks." + i))) yield return t;
        foreach (var t in _lnPost.NamedTensors(Join(prefix, "ln_post"))) yield return t;
        yield return (Join(prefix, "proj"), _projection.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _conv;
        yield return _classEmbedding;
        yield return _positionalEmbedding;
        yield return _projection;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _lnPre;
        foreach (var block in _blocks) yield return block;
        yield return _lnPost;
    }
}

/// <summary>
/// Everything SyncDreamer adds to the Zero123 U-Net and the VAE: the CLIP image encoder and Zero123's
/// <c>cc_projection</c>, the timestep embedding of the volume networks, the spatial volume network
/// (<c>SpatialVolumeNet</c>: noisy target-view encoder, spatial and frustum 3D U-Nets) and the depth-wise attention
/// blocks of the U-Net (<c>middle_conditions</c> and <c>output_conditions</c>).
/// </summary>
internal sealed class SyncDreamerNetwork<T> : KlModule<T>
{
    /// <summary>The viewpoint embedding width: (d_elevation, sin d_azimuth, cos d_azimuth, d_radius).</summary>
    public const int ViewpointDim = 4;

    private readonly SdDepthTransformer<T>?[] _siteConditions;
    private readonly List<SdDepthTransformer<T>> _outputConditions = new();

    public SyncDreamerNetwork(SyncDreamerOptions options, int latentChannels, IReadOnlyList<UNetFeatureSite> sites, int unetLevels)
    {
        var o = options;
        Clip = new SdClipImageEncoder<T>(o.ClipImageSize, o.ClipPatchSize, o.ClipWidth, o.ClipLayers, o.ClipHeads, o.CrossAttentionDim);
        CcProjection = SdLinear<T>.IdentityInitialized(o.CrossAttentionDim + ViewpointDim, o.CrossAttentionDim);
        TimeEmbed1 = new SdLinear<T>(o.TimeEmbedDim, o.TimeEmbedDim);
        TimeEmbed2 = new SdLinear<T>(o.TimeEmbedDim, o.TimeEmbedDim);
        TargetEncoder = new SdNoisyTargetViewEncoder<T>(latentChannels, o.TimeEmbedDim, ViewpointDim, o.TargetEncoderWidth,
            o.TargetEncoderOutputDim, o.NormGroups);
        SpatialNet = new SdSpatialTime3DNet<T>(o.TargetEncoderOutputDim * o.ViewCount, o.TimeEmbedDim, o.SpatialVolumeDims, o.NormGroups);
        FrustumNet = new SdFrustumTV3DNet<T>(o.SpatialVolumeDims[0], o.TimeEmbedDim, ViewpointDim, o.FrustumVolumeDims, o.NormGroups);
        TimeEmbedDim = o.TimeEmbedDim;

        // Depth-wise attention after the middle block and after every LDM output block of the levels above the deepest
        // (DepthWiseAttention: middle_conditions and output_b2c 3..11 for four levels), each reading the frustum volume
        // at the resolution the block outputs: DepthTransformer(ch, heads, d / 2, context_dim=d).
        _siteConditions = new SdDepthTransformer<T>?[sites.Count];
        SdDepthTransformer<T>? middle = null;
        for (int i = 0; i < sites.Count; i++)
        {
            var site = sites[i];
            bool conditioned = site.IsMiddle || site.Level <= unetLevels - 2;
            if (!conditioned) continue;
            if (site.Scale < 0 || site.Scale >= o.FrustumVolumeDims.Length)
                throw new ArgumentException(
                    $"The U-Net reaches 1/{1 << site.Scale} of the latent side, below the frustum volume's coarsest scale.",
                    nameof(sites));
            int context = o.FrustumVolumeDims[site.Scale];
            var transformer = new SdDepthTransformer<T>(site.Channels, o.DepthAttentionHeads, context / 2, context, o.NormGroups);
            _siteConditions[i] = transformer;
            if (site.IsMiddle) middle = transformer;
            else _outputConditions.Add(transformer);
        }

        MiddleCondition = middle ?? throw new ArgumentException("The U-Net has no middle block.", nameof(sites));
    }

    public SdClipImageEncoder<T> Clip { get; }

    public SdLinear<T> CcProjection { get; }

    public SdLinear<T> TimeEmbed1 { get; }

    public SdLinear<T> TimeEmbed2 { get; }

    public SdNoisyTargetViewEncoder<T> TargetEncoder { get; }

    public SdSpatialTime3DNet<T> SpatialNet { get; }

    public SdFrustumTV3DNet<T> FrustumNet { get; }

    public SdDepthTransformer<T> MiddleCondition { get; }

    public IReadOnlyList<SdDepthTransformer<T>> OutputConditions => _outputConditions;

    public int TimeEmbedDim { get; }

    /// <summary>The depth transformer at U-Net site <paramref name="siteIndex"/>, or null when the site is not conditioned.</summary>
    public SdDepthTransformer<T>? ConditionAt(int siteIndex)
        => siteIndex >= 0 && siteIndex < _siteConditions.Length ? _siteConditions[siteIndex] : null;

    /// <summary><c>embed_time</c>: the sinusoidal embedding of <paramref name="timestep"/> through Linear, SiLU, Linear; [1, dim].</summary>
    public Tensor<T> EmbedTime(int timestep)
        => TimeEmbed2.Forward(Engine.Swish(TimeEmbed1.Forward(SdOps<T>.TimestepEmbedding(timestep, TimeEmbedDim))));

    /// <summary>
    /// Every view's encoded noisy latent unprojected into the spatial volume, [N, C, V, V, V]
    /// (<c>construct_spatial_volume</c> before the 3D U-Net).
    /// </summary>
    /// <param name="noisy">The noisy target latents, [N, 4, S, S].</param>
    /// <param name="timeEmbedding">[1, timeDim].</param>
    /// <param name="viewEmbedding">[N, 4].</param>
    /// <param name="warps">Each view's voxel warp, [N, V, V * V, 2].</param>
    public Tensor<T> UnprojectViews(Tensor<T> noisy, Tensor<T> timeEmbedding, Tensor<T> viewEmbedding, Tensor<T> warps)
    {
        int n = noisy.Shape[0], v = warps.Shape[1];
        var t = Engine.TensorBroadcastTo(timeEmbedding, new[] { n, TimeEmbedDim });
        var features = TargetEncoder.Forward(noisy, t, viewEmbedding);                                       // [N, C, S, S]
        var sampled = Engine.GridSample(features, warps, GridSampleMode.Bilinear, GridSamplePadding.Zeros, true);  // [N, C, V, V*V]
        return Engine.Reshape(sampled, new[] { n, TargetEncoder.OutputDim, v, v, v });
    }

    /// <summary>The spatial feature volume [1, dims[0], V, V, V] (<c>construct_spatial_volume</c>).</summary>
    public Tensor<T> ConstructSpatialVolume(Tensor<T> noisy, Tensor<T> timeEmbedding, Tensor<T> viewEmbedding, Tensor<T> warps)
    {
        var perView = UnprojectViews(noisy, timeEmbedding, viewEmbedding, warps);
        int n = perView.Shape[0], c = perView.Shape[1], v = perView.Shape[2];
        // stack(1).view(B, N * C, V, V, V): view-major channels.
        var stacked = Engine.Reshape(perView, new[] { 1, n * c, v, v, v });
        return SpatialNet.Forward(stacked, timeEmbedding);
    }

    /// <summary>
    /// One view's frustum features at the four scales (<c>construct_view_frustum_volume</c>): the spatial volume
    /// sampled trilinearly at the frustum's points, then the frustum 3D U-Net.
    /// </summary>
    /// <param name="spatialVolume">[1, C, V, V, V].</param>
    /// <param name="timeEmbedding">[1, timeDim].</param>
    /// <param name="viewEmbedding">The view's embedding, [1, 4].</param>
    /// <param name="frustumGrid">[1, D, S, S, 3] normalized spatial-volume coordinates.</param>
    public Tensor<T>[] ConstructFrustumVolume(Tensor<T> spatialVolume, Tensor<T> timeEmbedding, Tensor<T> viewEmbedding, Tensor<T> frustumGrid)
    {
        var sampled = Engine.TensorGridSample3D(spatialVolume, frustumGrid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, true);
        return FrustumNet.Forward(sampled, timeEmbedding, viewEmbedding);
    }

    /// <summary>The U-Net's cross-attention context for one view: <c>cc_projection(cat(clip, v))</c>, [1, 1, dim].</summary>
    public Tensor<T> ViewContext(Tensor<T> clipEmbedding, Tensor<T> viewEmbedding)
    {
        var joined = Engine.TensorConcatenate(new[] { clipEmbedding, viewEmbedding }, 1);
        var projected = CcProjection.Forward(joined);
        return Engine.Reshape(projected, new[] { 1, 1, CcProjection.OutFeatures });
    }

    /// <summary>The feature map after site <paramref name="site"/>'s depth-wise attention (unchanged when the site has none).</summary>
    public Tensor<T> Condition(int siteIndex, UNetFeatureSite site, Tensor<T> features, Tensor<T>[] frustum)
    {
        var transformer = ConditionAt(siteIndex);
        return transformer is null ? features : transformer.Forward(features, frustum[site.Scale]);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        var all = Clip.NamedTensors(Join(prefix, "clip_image_encoder.model.visual"))
            .Concat(CcProjection.NamedTensors(Join(prefix, "cc_projection")))
            .Concat(TimeEmbed1.NamedTensors(Join(prefix, "time_embed.0")))
            .Concat(TimeEmbed2.NamedTensors(Join(prefix, "time_embed.2")))
            .Concat(TargetEncoder.NamedTensors(Join(prefix, "spatial_volume.target_encoder")))
            .Concat(SpatialNet.NamedTensors(Join(prefix, "spatial_volume.spatial_volume_feats")))
            .Concat(FrustumNet.NamedTensors(Join(prefix, "spatial_volume.frustum_volume_feats")))
            .Concat(MiddleCondition.NamedTensors(Join(prefix, "model.diffusion_model.middle_conditions")));
        for (int i = 0; i < _outputConditions.Count; i++)
            all = all.Concat(_outputConditions[i].NamedTensors(Join(prefix, "model.diffusion_model.output_conditions." + i)));
        return all;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return Clip;
        yield return CcProjection;
        yield return TimeEmbed1;
        yield return TimeEmbed2;
        yield return TargetEncoder;
        yield return SpatialNet;
        yield return FrustumNet;
        yield return MiddleCondition;
        foreach (var condition in _outputConditions) yield return condition;
    }
}
