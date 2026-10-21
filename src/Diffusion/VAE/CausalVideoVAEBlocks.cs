using AiDotNet.ComputerVision;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// A building block of CogVideoX's causal 3D VAE (Yang et al. 2024) as diffusers' <c>AutoencoderKLCogVideoX</c> builds
/// it. Every block reads its weights as live tensors through the engine, so the gradient tape reaches them, and names
/// each weight as diffusers' state dict does.
/// </summary>
/// <typeparam name="T">The numeric type of the weights.</typeparam>
internal abstract class CausalVideoModule<T> : CvParameterModule<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    protected static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Each weight under its diffusers state-dict name, <paramref name="prefix"/> first.</summary>
    public abstract IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix);

    protected override IEnumerable<Tensor<T>> OwnParameterTensors() => Array.Empty<Tensor<T>>();

    // torch.nn.Conv*d's default initialization: uniform in +-1/sqrt(fan_in) for the weight and the bias.
    protected static Tensor<T> Uniform(int[] shape, int fanIn)
    {
        var tensor = new Tensor<T>(shape);
        var random = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.NextRandom();
        double bound = 1.0 / Math.Sqrt(fanIn);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * bound);
        return tensor;
    }

    // Adds a per-channel bias to a [B, C, ...] tensor.
    protected static Tensor<T> AddChannelBias(Tensor<T> x, Tensor<T> bias)
    {
        var shape = new int[x.Rank];
        for (int i = 0; i < shape.Length; i++) shape[i] = 1;
        shape[1] = bias.Length;
        return Engine.TensorAdd(x, Engine.TensorBroadcastTo(Engine.Reshape(bias, shape), x.Shape.ToArray()));
    }

    // Frames [start, start + count) of a [B, C, T, H, W] tensor.
    protected static Tensor<T> Frames(Tensor<T> x, int start, int count)
    {
        var s = x.Shape;
        return Engine.TensorSlice(x, new[] { 0, 0, start, 0, 0 }, new[] { s[0], s[1], count, s[3], s[4] });
    }

    protected static string Join(string prefix, string name) => prefix.Length == 0 ? name : prefix + "." + name;
}

/// <summary>
/// <c>CogVideoXCausalConv3d</c> with its default <c>pad_mode="constant"</c>: the first frame is repeated
/// <c>kernel - 1</c> times in front, so an output frame reads only its own and earlier frames, and height and width
/// are zero-padded by <c>(kernel - 1) / 2</c>.
/// </summary>
internal sealed class CausalConv3d<T> : CausalVideoModule<T>
{
    private readonly int _kernel;
    private readonly int _timeStride;
    private readonly bool _nested;

    /// <summary>Creates a causal convolution with a kernel of the given (odd) size in time, height and width.</summary>
    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Output channels.</param>
    /// <param name="kernel">Kernel size, odd.</param>
    /// <param name="timeStride">Stride in time.</param>
    /// <param name="nested">diffusers keeps the convolution as a child named <c>conv</c>; the 1x1x1 residual shortcut
    /// is a bare <c>CogVideoXSafeConv3d</c> with no child.</param>
    public CausalConv3d(int inChannels, int outChannels, int kernel, int timeStride = 1, bool nested = true)
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        if (kernel <= 0 || kernel % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernel), "The kernel must be odd.");
        if (timeStride <= 0) throw new ArgumentOutOfRangeException(nameof(timeStride));
        _kernel = kernel;
        _timeStride = timeStride;
        _nested = nested;
        int fanIn = inChannels * kernel * kernel * kernel;
        Weight = Uniform(new[] { outChannels, inChannels, kernel, kernel, kernel }, fanIn);
        Bias = Uniform(new[] { outChannels }, fanIn);
    }

    public Tensor<T> Weight { get; }

    public Tensor<T> Bias { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        if (_kernel > 1)
        {
            var first = Frames(x, 0, 1);
            var parts = new Tensor<T>[_kernel];
            for (int i = 0; i < _kernel - 1; i++) parts[i] = first;
            parts[_kernel - 1] = x;
            x = Engine.TensorConcatenate(parts, 2);
        }
        int pad = (_kernel - 1) / 2;
        var y = Engine.Conv3D(x, Weight, new[] { _timeStride, 1, 1 }, new[] { 0, pad, pad }, new[] { 1, 1, 1 });
        return AddChannelBias(y, Bias);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        string conv = _nested ? Join(prefix, "conv") : prefix;
        yield return (Join(conv, "weight"), Weight);
        yield return (Join(conv, "bias"), Bias);
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren() => Array.Empty<IParameterSource<T>?>();

    protected override IEnumerable<Tensor<T>> OwnParameterTensors()
    {
        yield return Weight;
        yield return Bias;
    }
}

/// <summary><c>nn.GroupNorm</c> over [B, C, T, H, W]: statistics per group over its channels and the whole clip.</summary>
internal sealed class VideoGroupNorm<T> : CausalVideoModule<T>
{
    private readonly int _groups;
    private readonly double _epsilon;

    public VideoGroupNorm(int groups, int channels, double epsilon)
    {
        if (groups <= 0 || channels % groups != 0)
            throw new ArgumentException($"{channels} channels do not split into {groups} groups.", nameof(groups));
        _groups = groups;
        _epsilon = epsilon;
        Gamma = new Tensor<T>(new[] { channels });
        Beta = new Tensor<T>(new[] { channels });
        Gamma.Fill(NumOps.One);
    }

    public Tensor<T> Gamma { get; }

    public Tensor<T> Beta { get; }

    public Tensor<T> Forward(Tensor<T> x) => Engine.GroupNorm(x, _groups, Gamma, Beta, _epsilon, out _, out _);

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "weight"), Gamma);
        yield return (Join(prefix, "bias"), Beta);
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren() => Array.Empty<IParameterSource<T>?>();

    protected override IEnumerable<Tensor<T>> OwnParameterTensors()
    {
        yield return Gamma;
        yield return Beta;
    }
}

/// <summary>An <c>nn.Conv2d</c> applied to every frame of [B, C, T, H, W] independently.</summary>
internal sealed class FrameConv2d<T> : CausalVideoModule<T>
{
    private readonly int _stride;
    private readonly int _padding;

    public FrameConv2d(int inChannels, int outChannels, int kernel, int stride, int padding)
    {
        _stride = stride;
        _padding = padding;
        int fanIn = inChannels * kernel * kernel;
        Weight = Uniform(new[] { outChannels, inChannels, kernel, kernel }, fanIn);
        Bias = Uniform(new[] { outChannels }, fanIn);
    }

    public Tensor<T> Weight { get; }

    public Tensor<T> Bias { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var s = x.Shape;
        int batch = s[0], channels = s[1], frames = s[2];
        var flat = Engine.Reshape(Engine.TensorPermute(x, new[] { 0, 2, 1, 3, 4 }), new[] { batch * frames, channels, s[3], s[4] });
        var y = AddChannelBias(
            Engine.Conv2D(flat, Weight, new[] { _stride, _stride }, new[] { _padding, _padding }, new[] { 1, 1 }), Bias);
        var ys = y.Shape;
        return Engine.TensorPermute(Engine.Reshape(y, new[] { batch, frames, ys[1], ys[2], ys[3] }), new[] { 0, 2, 1, 3, 4 });
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "conv.weight"), Weight);
        yield return (Join(prefix, "conv.bias"), Bias);
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren() => Array.Empty<IParameterSource<T>?>();

    protected override IEnumerable<Tensor<T>> OwnParameterTensors()
    {
        yield return Weight;
        yield return Bias;
    }
}

/// <summary>
/// <c>CogVideoXSpatialNorm3D</c> (the decoder's normalization, after MoVQ): group norm modulated by the latent,
/// <c>norm(f) * conv_y(zq) + conv_b(zq)</c>, with zq resized (nearest) to f's frames, height and width. For an odd
/// frame count above one, the first frame and the rest resize separately, as the causal layout keeps them apart.
/// </summary>
internal sealed class SpatialNorm3D<T> : CausalVideoModule<T>
{
    public SpatialNorm3D(int channels, int latentChannels, int groups)
    {
        Norm = new VideoGroupNorm<T>(groups, channels, 1e-6);
        ConvY = new CausalConv3d<T>(latentChannels, channels, 1);
        ConvB = new CausalConv3d<T>(latentChannels, channels, 1);
    }

    public VideoGroupNorm<T> Norm { get; }

    public CausalConv3d<T> ConvY { get; }

    public CausalConv3d<T> ConvB { get; }

    public Tensor<T> Forward(Tensor<T> f, Tensor<T> zq)
    {
        var fs = f.Shape;
        int frames = fs[2];
        if (frames > 1 && frames % 2 == 1)
        {
            var first = Resize(Frames(zq, 0, 1), 1, fs[3], fs[4]);
            var rest = Resize(Frames(zq, 1, zq.Shape[2] - 1), frames - 1, fs[3], fs[4]);
            zq = Engine.TensorConcatenate(new[] { first, rest }, 2);
        }
        else
        {
            zq = Resize(zq, frames, fs[3], fs[4]);
        }
        return Engine.TensorAdd(Engine.TensorMultiply(Norm.Forward(f), ConvY.Forward(zq)), ConvB.Forward(zq));
    }

    private static Tensor<T> Resize(Tensor<T> x, int frames, int height, int width)
    {
        var s = x.Shape;
        if (s[2] == frames && s[3] == height && s[4] == width) return x;
        return Engine.Interpolate(x, new[] { frames, height, width }, InterpolateMode.Nearest, false);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) =>
        Norm.NamedTensors(Join(prefix, "norm_layer"))
            .Concat(ConvY.NamedTensors(Join(prefix, "conv_y")))
            .Concat(ConvB.NamedTensors(Join(prefix, "conv_b")));

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return Norm;
        yield return ConvY;
        yield return ConvB;
    }
}

/// <summary>
/// <c>CogVideoXResnetBlock3D</c> without a time embedding (the VAE passes none): norm, SiLU, causal 3x3x3 conv, twice,
/// plus the input (through a 1x1x1 conv when the width changes). The decoder's blocks normalize with
/// <see cref="SpatialNorm3D{T}"/>.
/// </summary>
internal sealed class CausalResnetBlock3D<T> : CausalVideoModule<T>
{
    private readonly VideoGroupNorm<T>? _groupNorm1;
    private readonly VideoGroupNorm<T>? _groupNorm2;
    private readonly SpatialNorm3D<T>? _spatialNorm1;
    private readonly SpatialNorm3D<T>? _spatialNorm2;
    private readonly CausalConv3d<T> _conv1;
    private readonly CausalConv3d<T> _conv2;
    private readonly CausalConv3d<T>? _shortcut;

    /// <summary>Creates a residual block from <paramref name="inChannels"/> to <paramref name="outChannels"/>.</summary>
    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Output channels.</param>
    /// <param name="groups">GroupNorm groups.</param>
    /// <param name="epsilon">GroupNorm epsilon.</param>
    /// <param name="latentChannels">The latent's channels when the block normalizes with the latent (decoder), else
    /// null.</param>
    public CausalResnetBlock3D(int inChannels, int outChannels, int groups, double epsilon, int? latentChannels)
    {
        if (latentChannels is int zq)
        {
            _spatialNorm1 = new SpatialNorm3D<T>(inChannels, zq, groups);
            _spatialNorm2 = new SpatialNorm3D<T>(outChannels, zq, groups);
        }
        else
        {
            _groupNorm1 = new VideoGroupNorm<T>(groups, inChannels, epsilon);
            _groupNorm2 = new VideoGroupNorm<T>(groups, outChannels, epsilon);
        }
        _conv1 = new CausalConv3d<T>(inChannels, outChannels, 3);
        _conv2 = new CausalConv3d<T>(outChannels, outChannels, 3);
        if (inChannels != outChannels) _shortcut = new CausalConv3d<T>(inChannels, outChannels, 1, nested: false);
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? zq)
    {
        var h = Engine.Swish(Normalize(_groupNorm1, _spatialNorm1, x, zq));
        h = _conv1.Forward(h);
        h = Engine.Swish(Normalize(_groupNorm2, _spatialNorm2, h, zq));
        h = _conv2.Forward(h);
        var residual = _shortcut is null ? x : _shortcut.Forward(x);
        return Engine.TensorAdd(h, residual);
    }

    private static Tensor<T> Normalize(VideoGroupNorm<T>? group, SpatialNorm3D<T>? spatial, Tensor<T> x, Tensor<T>? zq)
    {
        if (spatial is not null)
        {
            if (zq is null) throw new InvalidOperationException("A decoder block normalizes with the latent; none was given.");
            return spatial.Forward(x, zq);
        }
        if (group is null) throw new InvalidOperationException("The block has no normalization.");
        return group.Forward(x);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        // diffusers registers norm1 and norm2 before the convolutions.
        IEnumerable<(string, Tensor<T>)> norms = _spatialNorm1 is not null && _spatialNorm2 is not null
            ? _spatialNorm1.NamedTensors(Join(prefix, "norm1")).Concat(_spatialNorm2.NamedTensors(Join(prefix, "norm2")))
            : (_groupNorm1?.NamedTensors(Join(prefix, "norm1")) ?? Enumerable.Empty<(string, Tensor<T>)>())
                .Concat(_groupNorm2?.NamedTensors(Join(prefix, "norm2")) ?? Enumerable.Empty<(string, Tensor<T>)>());
        var all = norms
            .Concat(_conv1.NamedTensors(Join(prefix, "conv1")))
            .Concat(_conv2.NamedTensors(Join(prefix, "conv2")));
        if (_shortcut is not null) all = all.Concat(_shortcut.NamedTensors(Join(prefix, "conv_shortcut")));
        return all;
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _groupNorm1;
        yield return _groupNorm2;
        yield return _spatialNorm1;
        yield return _spatialNorm2;
        yield return _conv1;
        yield return _conv2;
        yield return _shortcut;
    }
}

/// <summary>
/// <c>CogVideoXDownsample3D</c>: when it compresses time, frames after the first are averaged in pairs (the first
/// frame stays on its own for an odd count); then each frame is padded by one on the right and bottom and halved by a
/// stride-2 3x3 conv.
/// </summary>
internal sealed class CausalDownsample3D<T> : CausalVideoModule<T>
{
    private readonly bool _compressTime;

    public CausalDownsample3D(int channels, bool compressTime)
    {
        _compressTime = compressTime;
        Conv = new FrameConv2d<T>(channels, channels, 3, 2, 0);
    }

    public FrameConv2d<T> Conv { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        if (_compressTime)
        {
            int frames = x.Shape[2];
            if (frames % 2 == 1)
            {
                var first = Frames(x, 0, 1);
                x = frames > 1 ? Engine.TensorConcatenate(new[] { first, PairMean(Frames(x, 1, frames - 1)) }, 2) : first;
            }
            else
            {
                x = PairMean(x);
            }
        }
        x = Engine.PadNd(x, new[] { 0, 1, 0, 1 }, PadMode.Constant, NumOps.Zero);
        return Conv.Forward(x);
    }

    // avg_pool1d(kernel 2, stride 2) over the frames of [B, C, T, H, W], T even.
    private static Tensor<T> PairMean(Tensor<T> x)
    {
        var s = x.Shape;
        int pairs = s[2] / 2;
        var grouped = Engine.Reshape(x, new[] { s[0], s[1], pairs, 2, s[3] * s[4] });
        var even = Engine.TensorSlice(grouped, new[] { 0, 0, 0, 0, 0 }, new[] { s[0], s[1], pairs, 1, s[3] * s[4] });
        var odd = Engine.TensorSlice(grouped, new[] { 0, 0, 0, 1, 0 }, new[] { s[0], s[1], pairs, 1, s[3] * s[4] });
        var mean = Engine.TensorMultiplyScalar(Engine.TensorAdd(even, odd), NumOps.FromDouble(0.5));
        return Engine.Reshape(mean, new[] { s[0], s[1], pairs, s[3], s[4] });
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) => Conv.NamedTensors(prefix);

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return Conv;
    }
}

/// <summary>
/// <c>CogVideoXUpsample3D</c>: nearest 2x upsampling of height and width, and of time when it compresses time (the
/// first frame of an odd count stays a single frame), then a 3x3 conv per frame.
/// </summary>
internal sealed class CausalUpsample3D<T> : CausalVideoModule<T>
{
    private readonly bool _compressTime;

    public CausalUpsample3D(int channels, bool compressTime)
    {
        _compressTime = compressTime;
        Conv = new FrameConv2d<T>(channels, channels, 3, 1, 1);
    }

    public FrameConv2d<T> Conv { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        int frames = x.Shape[2];
        if (_compressTime && frames > 1 && frames % 2 == 1)
        {
            var first = Engine.Upsample3D(Frames(x, 0, 1), 1, 2, 2);
            var rest = Engine.Upsample3D(Frames(x, 1, frames - 1), 2, 2, 2);
            x = Engine.TensorConcatenate(new[] { first, rest }, 2);
        }
        else if (_compressTime && frames > 1)
        {
            x = Engine.Upsample3D(x, 2, 2, 2);
        }
        else
        {
            x = Engine.Upsample3D(x, 1, 2, 2);
        }
        return Conv.Forward(x);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix) => Conv.NamedTensors(prefix);

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return Conv;
    }
}

/// <summary>A run of residual blocks followed by an optional resampler (a down or up block).</summary>
internal sealed class CausalResnetStage<T> : CausalVideoModule<T>
{
    private readonly string _resamplerName;

    public CausalResnetStage(IReadOnlyList<CausalResnetBlock3D<T>> resnets, CausalDownsample3D<T>? downsampler,
        CausalUpsample3D<T>? upsampler)
    {
        Resnets = resnets;
        Downsampler = downsampler;
        Upsampler = upsampler;
        _resamplerName = downsampler is not null ? "downsamplers.0" : "upsamplers.0";
    }

    public IReadOnlyList<CausalResnetBlock3D<T>> Resnets { get; }

    public CausalDownsample3D<T>? Downsampler { get; }

    public CausalUpsample3D<T>? Upsampler { get; }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? zq)
    {
        foreach (var resnet in Resnets) x = resnet.Forward(x, zq);
        if (Downsampler is not null) x = Downsampler.Forward(x);
        if (Upsampler is not null) x = Upsampler.Forward(x);
        return x;
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        var all = Enumerable.Empty<(string, Tensor<T>)>();
        for (int i = 0; i < Resnets.Count; i++) all = all.Concat(Resnets[i].NamedTensors(Join(prefix, $"resnets.{i}")));
        if (Downsampler is not null) all = all.Concat(Downsampler.NamedTensors(Join(prefix, _resamplerName)));
        if (Upsampler is not null) all = all.Concat(Upsampler.NamedTensors(Join(prefix, _resamplerName)));
        return all;
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        foreach (var resnet in Resnets) yield return resnet;
        yield return Downsampler;
        yield return Upsampler;
    }
}

/// <summary>
/// <c>CogVideoXEncoder3D</c>: a causal conv in, one stage per width (each halving height and width but the last, the
/// first log2(temporal compression) also halving time), two residual blocks, group norm, SiLU, and a causal conv out
/// to the posterior's mean and log-variance.
/// </summary>
internal sealed class CausalVideoEncoder<T> : CausalVideoModule<T>
{
    private readonly CausalConv3d<T> _convIn;
    private readonly List<CausalResnetStage<T>> _downBlocks = new();
    private readonly CausalResnetStage<T> _midBlock;
    private readonly VideoGroupNorm<T> _normOut;
    private readonly CausalConv3d<T> _convOut;

    public CausalVideoEncoder(int inChannels, int latentChannels, IReadOnlyList<int> blockOutChannels, int layersPerBlock,
        int groups, double epsilon, int temporalCompressLevel)
    {
        _convIn = new CausalConv3d<T>(inChannels, blockOutChannels[0], 3);
        int outChannels = blockOutChannels[0];
        for (int i = 0; i < blockOutChannels.Count; i++)
        {
            int inStage = outChannels;
            outChannels = blockOutChannels[i];
            var resnets = new List<CausalResnetBlock3D<T>>();
            for (int j = 0; j < layersPerBlock; j++)
                resnets.Add(new CausalResnetBlock3D<T>(j == 0 ? inStage : outChannels, outChannels, groups, epsilon, null));
            bool last = i == blockOutChannels.Count - 1;
            var down = last ? null : new CausalDownsample3D<T>(outChannels, compressTime: i < temporalCompressLevel);
            _downBlocks.Add(new CausalResnetStage<T>(resnets, down, null));
        }
        int deepest = blockOutChannels[blockOutChannels.Count - 1];
        _midBlock = new CausalResnetStage<T>(new[]
        {
            new CausalResnetBlock3D<T>(deepest, deepest, groups, epsilon, null),
            new CausalResnetBlock3D<T>(deepest, deepest, groups, epsilon, null),
        }, null, null);
        _normOut = new VideoGroupNorm<T>(groups, deepest, 1e-6);
        _convOut = new CausalConv3d<T>(deepest, 2 * latentChannels, 3);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        x = _convIn.Forward(x);
        foreach (var block in _downBlocks) x = block.Forward(x, null);
        x = _midBlock.Forward(x, null);
        x = Engine.Swish(_normOut.Forward(x));
        return _convOut.Forward(x);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        var all = _convIn.NamedTensors(Join(prefix, "conv_in"));
        for (int i = 0; i < _downBlocks.Count; i++) all = all.Concat(_downBlocks[i].NamedTensors(Join(prefix, $"down_blocks.{i}")));
        return all
            .Concat(_midBlock.NamedTensors(Join(prefix, "mid_block")))
            .Concat(_normOut.NamedTensors(Join(prefix, "norm_out")))
            .Concat(_convOut.NamedTensors(Join(prefix, "conv_out")));
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _convIn;
        foreach (var block in _downBlocks) yield return block;
        yield return _midBlock;
        yield return _normOut;
        yield return _convOut;
    }
}

/// <summary>
/// <c>CogVideoXDecoder3D</c>: a causal conv in, two residual blocks, one stage per width from the deepest (each with
/// one more block than the encoder's, doubling height and width but the last, the first log2(temporal compression)
/// also doubling time), and a latent-modulated norm, SiLU and causal conv out. Every normalization reads the latent.
/// </summary>
internal sealed class CausalVideoDecoder<T> : CausalVideoModule<T>
{
    private readonly CausalConv3d<T> _convIn;
    private readonly CausalResnetStage<T> _midBlock;
    private readonly List<CausalResnetStage<T>> _upBlocks = new();
    private readonly SpatialNorm3D<T> _normOut;
    private readonly CausalConv3d<T> _convOut;

    public CausalVideoDecoder(int latentChannels, int outChannels, IReadOnlyList<int> blockOutChannels, int layersPerBlock,
        int groups, double epsilon, int temporalCompressLevel)
    {
        var reversed = blockOutChannels.Reverse().ToArray();
        _convIn = new CausalConv3d<T>(latentChannels, reversed[0], 3);
        _midBlock = new CausalResnetStage<T>(new[]
        {
            new CausalResnetBlock3D<T>(reversed[0], reversed[0], groups, epsilon, latentChannels),
            new CausalResnetBlock3D<T>(reversed[0], reversed[0], groups, epsilon, latentChannels),
        }, null, null);
        int stageOut = reversed[0];
        for (int i = 0; i < reversed.Length; i++)
        {
            int stageIn = stageOut;
            stageOut = reversed[i];
            var resnets = new List<CausalResnetBlock3D<T>>();
            for (int j = 0; j < layersPerBlock + 1; j++)
                resnets.Add(new CausalResnetBlock3D<T>(j == 0 ? stageIn : stageOut, stageOut, groups, epsilon, latentChannels));
            bool last = i == reversed.Length - 1;
            var up = last ? null : new CausalUpsample3D<T>(stageOut, compressTime: i < temporalCompressLevel);
            _upBlocks.Add(new CausalResnetStage<T>(resnets, null, up));
        }
        _normOut = new SpatialNorm3D<T>(reversed[reversed.Length - 1], latentChannels, groups);
        _convOut = new CausalConv3d<T>(reversed[reversed.Length - 1], outChannels, 3);
    }

    public Tensor<T> Forward(Tensor<T> z)
    {
        var x = _convIn.Forward(z);
        x = _midBlock.Forward(x, z);
        foreach (var block in _upBlocks) x = block.Forward(x, z);
        x = Engine.Swish(_normOut.Forward(x, z));
        return _convOut.Forward(x);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        var all = _convIn.NamedTensors(Join(prefix, "conv_in"))
            .Concat(_midBlock.NamedTensors(Join(prefix, "mid_block")));
        for (int i = 0; i < _upBlocks.Count; i++) all = all.Concat(_upBlocks[i].NamedTensors(Join(prefix, $"up_blocks.{i}")));
        return all
            .Concat(_normOut.NamedTensors(Join(prefix, "norm_out")))
            .Concat(_convOut.NamedTensors(Join(prefix, "conv_out")));
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _convIn;
        yield return _midBlock;
        foreach (var block in _upBlocks) yield return block;
        yield return _normOut;
        yield return _convOut;
    }
}
