using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// A building block of the latent-diffusion KL autoencoder (Rombach et al. 2022) as diffusers' <c>AutoencoderKL</c>
/// builds it, over [B, C, H, W]. Every block reads its weights as live tensors through the engine, so the gradient tape
/// reaches them, and names each weight as diffusers' state dict does.
/// </summary>
/// <typeparam name="T">The numeric type of the weights.</typeparam>
internal abstract class KlModule<T> : LazyWeightModule<T>
{
    /// <summary>Each weight under its diffusers state-dict name, <paramref name="prefix"/> first.</summary>
    public abstract IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix);

    // Adds a per-channel bias to [B, C, ...].
    protected static Tensor<T> AddChannelBias(Tensor<T> x, Tensor<T> bias)
    {
        var shape = new int[x.Rank];
        for (int i = 0; i < shape.Length; i++) shape[i] = 1;
        shape[1] = bias.Length;
        return Engine.TensorAdd(x, Engine.TensorBroadcastTo(Engine.Reshape(bias, shape), x.Shape.ToArray()));
    }

    protected static string Join(string prefix, string name) => prefix.Length == 0 ? name : prefix + "." + name;

    /// <summary>Writes each named weight from <paramref name="read"/>, which is given its name and shape.</summary>
    /// <returns>The names read, in order.</returns>
    public static IReadOnlyCollection<string> Load(IEnumerable<(string Name, Tensor<T> Tensor)> named, Func<string, int[], double[]> read)
    {
        if (read is null) throw new ArgumentNullException(nameof(read));
        var names = new List<string>();
        foreach (var (name, tensor) in named)
        {
            var values = read(name, tensor.Shape.ToArray());
            if (values is null || values.Length != tensor.Length)
                throw new ArgumentException($"'{name}' needs {tensor.Length} values; {values?.Length ?? 0} were given.", nameof(read));
            for (int i = 0; i < values.Length; i++) tensor[i] = NumOps.FromDouble(values[i]);
            names.Add(name);
        }
        return names;
    }
}

/// <summary>nn.Conv2d with a bias.</summary>
internal sealed class KlConv2d<T> : KlModule<T>
{
    private readonly int _stride;
    private readonly int _padding;

    public KlConv2d(int inChannels, int outChannels, int kernel, int stride = 1, int padding = 0)
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        _stride = stride;
        _padding = padding;
        int fanIn = inChannels * kernel * kernel;
        _weight = LazyWeight<T>.Uniform(new[] { outChannels, inChannels, kernel, kernel }, fanIn);
        _bias = LazyWeight<T>.Uniform(new[] { outChannels }, fanIn);
    }

    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T> _bias;

    public Tensor<T> Forward(Tensor<T> x) =>
        AddChannelBias(Engine.Conv2D(x, _weight.Value, new[] { _stride, _stride }, new[] { _padding, _padding }, new[] { 1, 1 }), _bias.Value);

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

/// <summary>
/// nn.GroupNorm with an affine scale and shift over [B, C, ...] (epsilon 1e-6, as the KL autoencoder uses, unless
/// given).
/// </summary>
internal sealed class KlGroupNorm<T> : KlModule<T>
{
    private readonly int _groups;
    private readonly double _epsilon;

    public KlGroupNorm(int groups, int channels, double epsilon = 1e-6)
    {
        if (groups <= 0 || channels % groups != 0)
            throw new ArgumentException($"{channels} channels do not split into {groups} groups.", nameof(groups));
        _groups = groups;
        _epsilon = epsilon;
        _gamma = LazyWeight<T>.Constant(new[] { channels }, 1.0);
        _beta = LazyWeight<T>.Constant(new[] { channels }, 0.0);
    }

    private readonly LazyWeight<T> _gamma;
    private readonly LazyWeight<T> _beta;

    public Tensor<T> Forward(Tensor<T> x) => Engine.GroupNorm(x, _groups, _gamma.Value, _beta.Value, _epsilon, out _, out _);

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
/// ResnetBlock2D without a time embedding: norm, SiLU, 3x3 conv, twice, plus the input (through a 1x1 conv when the
/// width changes).
/// </summary>
internal sealed class KlResnetBlock<T> : KlModule<T>
{
    private readonly KlGroupNorm<T> _norm1;
    private readonly KlConv2d<T> _conv1;
    private readonly KlGroupNorm<T> _norm2;
    private readonly KlConv2d<T> _conv2;
    private readonly KlConv2d<T>? _shortcut;

    public KlResnetBlock(int inChannels, int outChannels, int groups)
    {
        _norm1 = new KlGroupNorm<T>(groups, inChannels);
        _conv1 = new KlConv2d<T>(inChannels, outChannels, 3, padding: 1);
        _norm2 = new KlGroupNorm<T>(groups, outChannels);
        _conv2 = new KlConv2d<T>(outChannels, outChannels, 3, padding: 1);
        if (inChannels != outChannels) _shortcut = new KlConv2d<T>(inChannels, outChannels, 1);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var h = _conv1.Forward(Engine.Swish(_norm1.Forward(x)));
        h = _conv2.Forward(Engine.Swish(_norm2.Forward(h)));
        return Engine.TensorAdd(_shortcut is null ? x : _shortcut.Forward(x), h);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        var all = _norm1.NamedTensors(Join(prefix, "norm1"))
            .Concat(_conv1.NamedTensors(Join(prefix, "conv1")))
            .Concat(_norm2.NamedTensors(Join(prefix, "norm2")))
            .Concat(_conv2.NamedTensors(Join(prefix, "conv2")));
        return _shortcut is null ? all : all.Concat(_shortcut.NamedTensors(Join(prefix, "conv_shortcut")));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _norm1;
        yield return _conv1;
        yield return _norm2;
        yield return _conv2;
        yield return _shortcut;
    }
}

/// <summary>
/// The middle block's single-head self-attention over the spatial positions: group norm, query, key and value
/// projections, softmax(Q K^T / sqrt(C)) V, an output projection, and the input added back.
/// </summary>
internal sealed class KlAttention<T> : KlModule<T>
{
    private readonly int _channels;
    private readonly KlGroupNorm<T> _norm;

    public KlAttention(int channels, int groups)
    {
        _channels = channels;
        _norm = new KlGroupNorm<T>(groups, channels);
        _queryWeight = LazyWeight<T>.Uniform(new[] { channels, channels }, channels);
        _queryBias = LazyWeight<T>.Uniform(new[] { channels }, channels);
        _keyWeight = LazyWeight<T>.Uniform(new[] { channels, channels }, channels);
        _keyBias = LazyWeight<T>.Uniform(new[] { channels }, channels);
        _valueWeight = LazyWeight<T>.Uniform(new[] { channels, channels }, channels);
        _valueBias = LazyWeight<T>.Uniform(new[] { channels }, channels);
        _outWeight = LazyWeight<T>.Uniform(new[] { channels, channels }, channels);
        _outBias = LazyWeight<T>.Uniform(new[] { channels }, channels);
    }

    private readonly LazyWeight<T> _queryWeight;
    private readonly LazyWeight<T> _queryBias;
    private readonly LazyWeight<T> _keyWeight;
    private readonly LazyWeight<T> _keyBias;
    private readonly LazyWeight<T> _valueWeight;
    private readonly LazyWeight<T> _valueBias;
    private readonly LazyWeight<T> _outWeight;
    private readonly LazyWeight<T> _outBias;

    public Tensor<T> Forward(Tensor<T> x)
    {
        var s = x.Shape;
        int batch = s[0], positions = s[2] * s[3];
        // [B, C, H, W] -> [B, HW, C].
        var tokens = Engine.TensorPermute(Engine.Reshape(_norm.Forward(x), new[] { batch, _channels, positions }), new[] { 0, 2, 1 });
        var q = Linear(tokens, _queryWeight.Value, _queryBias.Value);
        var k = Linear(tokens, _keyWeight.Value, _keyBias.Value);
        var v = Linear(tokens, _valueWeight.Value, _valueBias.Value);
        var scores = Engine.TensorMultiplyScalar(Engine.BatchMatMul(q, Engine.TensorPermute(k, new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(_channels)));
        var attended = Linear(Engine.BatchMatMul(Engine.Softmax(scores, 2), v), _outWeight.Value, _outBias.Value);
        var back = Engine.Reshape(Engine.TensorPermute(attended, new[] { 0, 2, 1 }), new[] { batch, _channels, s[2], s[3] });
        return Engine.TensorAdd(back, x);
    }

    // [B, N, C] times a PyTorch [out, in] weight plus a bias.
    private static Tensor<T> Linear(Tensor<T> x, Tensor<T> weight, Tensor<T> bias)
    {
        var s = x.Shape;
        int rows = s[0] * s[1], outFeatures = weight.Shape[0];
        var y = Engine.TensorMatMul(Engine.Reshape(x, new[] { rows, s[2] }), Engine.TensorPermute(weight, new[] { 1, 0 }));
        y = Engine.TensorAdd(y, Engine.TensorBroadcastTo(Engine.Reshape(bias, new[] { 1, outFeatures }), new[] { rows, outFeatures }));
        return Engine.Reshape(y, new[] { s[0], s[1], outFeatures });
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        foreach (var named in _norm.NamedTensors(Join(prefix, "group_norm"))) yield return named;
        yield return (Join(prefix, "to_q.weight"), _queryWeight.Value);
        yield return (Join(prefix, "to_q.bias"), _queryBias.Value);
        yield return (Join(prefix, "to_k.weight"), _keyWeight.Value);
        yield return (Join(prefix, "to_k.bias"), _keyBias.Value);
        yield return (Join(prefix, "to_v.weight"), _valueWeight.Value);
        yield return (Join(prefix, "to_v.bias"), _valueBias.Value);
        yield return (Join(prefix, "to_out.0.weight"), _outWeight.Value);
        yield return (Join(prefix, "to_out.0.bias"), _outBias.Value);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _norm;
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _queryWeight;
        yield return _queryBias;
        yield return _keyWeight;
        yield return _keyBias;
        yield return _valueWeight;
        yield return _valueBias;
        yield return _outWeight;
        yield return _outBias;
    }
}

/// <summary>UNetMidBlock2D as the autoencoder uses it: residual block, self-attention, residual block.</summary>
internal sealed class KlMidBlock<T> : KlModule<T>
{
    private readonly KlResnetBlock<T> _resnet0;
    private readonly KlAttention<T> _attention;
    private readonly KlResnetBlock<T> _resnet1;

    public KlMidBlock(int channels, int groups)
    {
        _resnet0 = new KlResnetBlock<T>(channels, channels, groups);
        _attention = new KlAttention<T>(channels, groups);
        _resnet1 = new KlResnetBlock<T>(channels, channels, groups);
    }

    public Tensor<T> Forward(Tensor<T> x) => _resnet1.Forward(_attention.Forward(_resnet0.Forward(x)));

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) =>
        _attention.NamedTensors(Join(prefix, "attentions.0"))
            .Concat(_resnet0.NamedTensors(Join(prefix, "resnets.0")))
            .Concat(_resnet1.NamedTensors(Join(prefix, "resnets.1")));

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _resnet0;
        yield return _attention;
        yield return _resnet1;
    }
}

/// <summary>
/// The KL encoder: a 3x3 conv in, one stage of residual blocks per width (each but the last halved by a stride-2 conv
/// after padding the right and bottom by one), the middle block, GroupNorm, SiLU and a 3x3 conv to the posterior's
/// mean and log-variance.
/// </summary>
internal sealed class KlEncoder<T> : KlModule<T>
{
    private readonly KlConv2d<T> _convIn;
    private readonly List<(List<KlResnetBlock<T>> Resnets, KlConv2d<T>? Downsample)> _stages = new();
    private readonly KlMidBlock<T> _mid;
    private readonly KlGroupNorm<T> _normOut;
    private readonly KlConv2d<T> _convOut;

    public KlEncoder(int inChannels, int latentChannels, IReadOnlyList<int> widths, int layersPerBlock, int groups)
    {
        _convIn = new KlConv2d<T>(inChannels, widths[0], 3, padding: 1);
        int previous = widths[0];
        for (int i = 0; i < widths.Count; i++)
        {
            var resnets = new List<KlResnetBlock<T>>();
            for (int j = 0; j < layersPerBlock; j++) resnets.Add(new KlResnetBlock<T>(j == 0 ? previous : widths[i], widths[i], groups));
            var down = i < widths.Count - 1 ? new KlConv2d<T>(widths[i], widths[i], 3, stride: 2) : null;
            _stages.Add((resnets, down));
            previous = widths[i];
        }
        _mid = new KlMidBlock<T>(previous, groups);
        _normOut = new KlGroupNorm<T>(groups, previous);
        _convOut = new KlConv2d<T>(previous, 2 * latentChannels, 3, padding: 1);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        x = _convIn.Forward(x);
        foreach (var (resnets, down) in _stages)
        {
            foreach (var resnet in resnets) x = resnet.Forward(x);
            // Downsample2D pads (0, 1, 0, 1) and convolves with stride 2 and no padding.
            if (down is not null) x = down.Forward(Engine.PadNd(x, new[] { 0, 1, 0, 1 }, PadMode.Constant, NumOps.Zero));
        }
        x = _mid.Forward(x);
        return _convOut.Forward(Engine.Swish(_normOut.Forward(x)));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        var all = _convIn.NamedTensors(Join(prefix, "conv_in"));
        for (int i = 0; i < _stages.Count; i++)
        {
            for (int j = 0; j < _stages[i].Resnets.Count; j++)
                all = all.Concat(_stages[i].Resnets[j].NamedTensors(Join(prefix, $"down_blocks.{i}.resnets.{j}")));
            if (_stages[i].Downsample is { } down) all = all.Concat(down.NamedTensors(Join(prefix, $"down_blocks.{i}.downsamplers.0.conv")));
        }
        return all
            .Concat(_mid.NamedTensors(Join(prefix, "mid_block")))
            .Concat(_normOut.NamedTensors(Join(prefix, "conv_norm_out")))
            .Concat(_convOut.NamedTensors(Join(prefix, "conv_out")));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _convIn;
        foreach (var (resnets, down) in _stages)
        {
            foreach (var resnet in resnets) yield return resnet;
            yield return down;
        }
        yield return _mid;
        yield return _normOut;
        yield return _convOut;
    }
}

/// <summary>
/// The KL decoder: a 3x3 conv in, the middle block, one stage per width from the deepest (each with one more residual
/// block than the encoder's and, but the last, a nearest 2x upsample and 3x3 conv), GroupNorm, SiLU and a 3x3 conv out.
/// </summary>
internal sealed class KlDecoder<T> : KlModule<T>
{
    private readonly KlConv2d<T> _convIn;
    private readonly KlMidBlock<T> _mid;
    private readonly List<(List<KlResnetBlock<T>> Resnets, KlConv2d<T>? Upsample)> _stages = new();
    private readonly KlGroupNorm<T> _normOut;
    private readonly KlConv2d<T> _convOut;

    public KlDecoder(int latentChannels, int outChannels, IReadOnlyList<int> widths, int layersPerBlock, int groups)
    {
        var reversed = widths.Reverse().ToArray();
        _convIn = new KlConv2d<T>(latentChannels, reversed[0], 3, padding: 1);
        _mid = new KlMidBlock<T>(reversed[0], groups);
        int previous = reversed[0];
        for (int i = 0; i < reversed.Length; i++)
        {
            var resnets = new List<KlResnetBlock<T>>();
            for (int j = 0; j < layersPerBlock + 1; j++) resnets.Add(new KlResnetBlock<T>(j == 0 ? previous : reversed[i], reversed[i], groups));
            var up = i < reversed.Length - 1 ? new KlConv2d<T>(reversed[i], reversed[i], 3, padding: 1) : null;
            _stages.Add((resnets, up));
            previous = reversed[i];
        }
        _normOut = new KlGroupNorm<T>(groups, previous);
        _convOut = new KlConv2d<T>(previous, outChannels, 3, padding: 1);
    }

    public Tensor<T> Forward(Tensor<T> z)
    {
        var x = _mid.Forward(_convIn.Forward(z));
        foreach (var (resnets, up) in _stages)
        {
            foreach (var resnet in resnets) x = resnet.Forward(x);
            if (up is not null) x = up.Forward(Engine.Upsample(x, 2, 2));
        }
        return _convOut.Forward(Engine.Swish(_normOut.Forward(x)));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        var all = _convIn.NamedTensors(Join(prefix, "conv_in"))
            .Concat(_mid.NamedTensors(Join(prefix, "mid_block")));
        for (int i = 0; i < _stages.Count; i++)
        {
            for (int j = 0; j < _stages[i].Resnets.Count; j++)
                all = all.Concat(_stages[i].Resnets[j].NamedTensors(Join(prefix, $"up_blocks.{i}.resnets.{j}")));
            if (_stages[i].Upsample is { } up) all = all.Concat(up.NamedTensors(Join(prefix, $"up_blocks.{i}.upsamplers.0.conv")));
        }
        return all
            .Concat(_normOut.NamedTensors(Join(prefix, "conv_norm_out")))
            .Concat(_convOut.NamedTensors(Join(prefix, "conv_out")));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _convIn;
        yield return _mid;
        foreach (var (resnets, up) in _stages)
        {
            foreach (var resnet in resnets) yield return resnet;
            yield return up;
        }
        yield return _normOut;
        yield return _convOut;
    }
}
