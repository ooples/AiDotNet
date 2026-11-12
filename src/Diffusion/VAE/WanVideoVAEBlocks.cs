using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// The per-convolution feature cache Wan 2.1's VAE (Wan-Video/Wan2.1, wan/modules/vae.py) carries from one chunk of frames
/// to the next: one slot per cached convolution in call order, refilled as each chunk runs.
/// </summary>
internal sealed class WanFeatureCache<T>
{
    private readonly List<Slot> _slots = new();
    private int _index;

    internal sealed class Slot
    {
        internal Tensor<T>? Tensor { get; set; }

        // upsample3d's marker for "the first chunk went through without its time convolution" (the release's 'Rep').
        internal bool Repeat { get; set; }

        internal bool IsEmpty => Tensor is null && !Repeat;
    }

    /// <summary>Starts a chunk: the slots are visited again from the first.</summary>
    internal void BeginChunk() => _index = 0;

    /// <summary>The next slot, created on the first chunk.</summary>
    internal Slot Next()
    {
        if (_index == _slots.Count) _slots.Add(new Slot());
        return _slots[_index++];
    }
}

/// <summary>Shared pieces of Wan 2.1's causal video VAE.</summary>
internal abstract class WanVideoModule<T> : CausalVideoModule<T>
{
    // CACHE_T = 2: the last two frames of x, or the previous cache's last frame in front of x's single frame.
    protected static Tensor<T> NextCache(Tensor<T> x, Tensor<T>? previous)
    {
        int t = x.Shape[2];
        var cache = Frames(x, Math.Max(0, t - 2), Math.Min(2, t));
        if (cache.Shape[2] < 2 && previous is not null)
            cache = Engine.TensorConcatenate(new[] { Frames(previous, previous.Shape[2] - 1, 1), cache }, 2);
        return cache;
    }

    // A cached causal convolution as the release's encoder, decoder and residual blocks run one.
    protected static Tensor<T> Cached(WanCausalConv3d<T> conv, Tensor<T> x, WanFeatureCache<T>? cache)
    {
        if (cache is null) return conv.Forward(x, null);
        var slot = cache.Next();
        var next = NextCache(x, slot.Tensor);
        var y = conv.Forward(x, slot.Tensor);
        slot.Tensor = next;
        return y;
    }
}

/// <summary>
/// <c>CausalConv3d</c>: zero padding of twice the time padding in front only (or the cached frames in its place) and the
/// symmetric spatial padding, then an unpadded <c>nn.Conv3d</c>.
/// </summary>
internal sealed class WanCausalConv3d<T> : WanVideoModule<T>
{
    private readonly int[] _kernel;
    private readonly int _timeStride, _timePad, _spacePad;
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T> _bias;

    public WanCausalConv3d(int inChannels, int outChannels, int timeKernel, int spaceKernel, int timeStride = 1, int timePad = -1,
        int spacePad = -1, bool zeroWeight = false)
    {
        _kernel = new[] { timeKernel, spaceKernel, spaceKernel };
        _timeStride = timeStride;
        _timePad = timePad < 0 ? timeKernel / 2 : timePad;
        _spacePad = spacePad < 0 ? spaceKernel / 2 : spacePad;
        int fanIn = inChannels * timeKernel * spaceKernel * spaceKernel;
        var shape = new[] { outChannels, inChannels, timeKernel, spaceKernel, spaceKernel };
        _weight = zeroWeight ? LazyWeight<T>.Constant(shape, 0.0) : LazyWeight<T>.Uniform(shape, fanIn);
        _bias = LazyWeight<T>.Uniform(new[] { outChannels }, fanIn);
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? cache)
    {
        int front = 2 * _timePad;
        if (cache is not null && front > 0)
        {
            x = Engine.TensorConcatenate(new[] { cache, x }, 2);
            front -= cache.Shape[2];
        }
        if (front > 0 || _spacePad > 0)
            x = Engine.PadNd(x, new[] { _spacePad, _spacePad, _spacePad, _spacePad, Math.Max(0, front), 0 }, PadMode.Constant, NumOps.Zero);
        var y = Engine.Conv3D(x, _weight.Value, new[] { _timeStride, 1, 1 }, new[] { 0, 0, 0 }, new[] { 1, 1, 1 });
        return AddChannelBias(y, _bias.Value);
    }

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

/// <summary><c>RMS_norm</c>: F.normalize over the channels, times sqrt(channels) and a learned scale.</summary>
internal sealed class WanRmsNorm<T> : WanVideoModule<T>
{
    private readonly int _dim;
    private readonly LazyWeight<T> _gamma;

    public WanRmsNorm(int dim)
    {
        _dim = dim;
        _gamma = LazyWeight<T>.Constant(new[] { dim }, 1.0);
    }

    /// <summary>Normalizes axis 1 of x (any rank).</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        var norm = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(x), new[] { 1 }, keepDims: true));
        var unit = Engine.TensorDivide(x, Engine.TensorBroadcastTo(Engine.TensorMax(norm, NumOps.FromDouble(1e-12)), shape));
        return Engine.TensorMultiplyScalar(AddScale(unit, _gamma.Value), NumOps.FromDouble(Math.Sqrt(_dim)));
    }

    private static Tensor<T> AddScale(Tensor<T> x, Tensor<T> gamma)
    {
        var shape = new int[x.Rank];
        for (int i = 0; i < shape.Length; i++) shape[i] = 1;
        shape[1] = gamma.Length;
        return Engine.TensorMultiply(x, Engine.TensorBroadcastTo(Engine.Reshape(gamma, shape), x.Shape.ToArray()));
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        yield return (Join(prefix, "gamma"), _gamma.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _gamma;
    }
}

/// <summary><c>ResidualBlock</c>: (RMS norm, SiLU, causal 3x3x3 convolution) twice, plus a 1x1x1 shortcut when widths differ.</summary>
internal sealed class WanResidualBlock<T> : WanVideoModule<T>
{
    private readonly WanRmsNorm<T> _norm1, _norm2;
    private readonly WanCausalConv3d<T> _conv1, _conv2;
    private readonly WanCausalConv3d<T>? _shortcut;

    public WanResidualBlock(int inDim, int outDim)
    {
        _norm1 = new WanRmsNorm<T>(inDim);
        _conv1 = new WanCausalConv3d<T>(inDim, outDim, 3, 3);
        _norm2 = new WanRmsNorm<T>(outDim);
        _conv2 = new WanCausalConv3d<T>(outDim, outDim, 3, 3);
        if (inDim != outDim) _shortcut = new WanCausalConv3d<T>(inDim, outDim, 1, 1);
    }

    public Tensor<T> Forward(Tensor<T> x, WanFeatureCache<T>? cache)
    {
        var h = _shortcut is null ? x : _shortcut.Forward(x, null);
        var y = Cached(_conv1, Engine.Swish(_norm1.Forward(x)), cache);
        y = Cached(_conv2, Engine.Swish(_norm2.Forward(y)), cache);
        return Engine.TensorAdd(y, h);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
        => _norm1.NamedTensors(Join(prefix, "residual.0")).Concat(_conv1.NamedTensors(Join(prefix, "residual.2")))
            .Concat(_norm2.NamedTensors(Join(prefix, "residual.3"))).Concat(_conv2.NamedTensors(Join(prefix, "residual.6")))
            .Concat(_shortcut?.NamedTensors(Join(prefix, "shortcut")) ?? Enumerable.Empty<(string, Tensor<T>)>());

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _norm1;
        yield return _conv1;
        yield return _norm2;
        yield return _conv2;
        yield return _shortcut;
    }
}

/// <summary><c>AttentionBlock</c>: single-head self-attention over each frame's pixels, its output projection zero-initialized.</summary>
internal sealed class WanAttentionBlock<T> : WanVideoModule<T>
{
    private readonly WanRmsNorm<T> _norm;
    private readonly WanCausalConv3d<T> _qkv, _proj;

    public WanAttentionBlock(int dim)
    {
        _norm = new WanRmsNorm<T>(dim);
        _qkv = new WanCausalConv3d<T>(dim, 3 * dim, 1, 1);
        _proj = new WanCausalConv3d<T>(dim, dim, 1, 1, zeroWeight: true);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var s = x.Shape;
        int b = s[0], c = s[1], t = s[2], hw = s[3] * s[4];
        var qkv = _qkv.Forward(_norm.Forward(x), null);
        // [B, 3C, T, H, W] to [B·T, HW, 3C], then q, k and v along the channels.
        var tokens = Engine.Reshape(Engine.TensorPermute(qkv, new[] { 0, 2, 3, 4, 1 }), new[] { b * t, hw, 3 * c });
        var q = Engine.TensorNarrow(tokens, 2, 0, c);
        var k = Engine.TensorNarrow(tokens, 2, c, c);
        var v = Engine.TensorNarrow(tokens, 2, 2 * c, c);
        var scores = Engine.TensorMultiplyScalar(Engine.TensorBatchMatMul(q, Engine.TensorPermute(k, new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(c)));
        var attended = Engine.TensorBatchMatMul(Engine.TensorSoftmax(scores, 2), v);
        var back = Engine.TensorPermute(Engine.Reshape(attended, new[] { b, t, s[3], s[4], c }), new[] { 0, 4, 1, 2, 3 });
        return Engine.TensorAdd(_proj.Forward(back, null), x);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
        => _norm.NamedTensors(Join(prefix, "norm")).Concat(_qkv.NamedTensors(Join(prefix, "to_qkv")))
            .Concat(_proj.NamedTensors(Join(prefix, "proj")));

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _norm;
        yield return _qkv;
        yield return _proj;
    }
}

/// <summary>The four <c>Resample</c> modes.</summary>
internal enum WanResampleMode
{
    Upsample2d,
    Upsample3d,
    Downsample2d,
    Downsample3d,
}

/// <summary>
/// <c>Resample</c>: nearest x2 then a 3x3 convolution halving the width (upsampling), or a right/bottom zero pad and a
/// stride-2 3x3 convolution (downsampling), per frame; the 3D modes add a causal time convolution that doubles the frames
/// (from the second chunk on) or halves them against the previous chunk's last frame.
/// </summary>
internal sealed class WanResample<T> : WanVideoModule<T>
{
    private readonly WanResampleMode _mode;
    private readonly FrameConv2d<T> _conv;
    private readonly WanCausalConv3d<T>? _timeConv;

    public WanResample(int dim, WanResampleMode mode)
    {
        _mode = mode;
        bool up = mode is WanResampleMode.Upsample2d or WanResampleMode.Upsample3d;
        _conv = up ? new FrameConv2d<T>(dim, dim / 2, 3, 1, 1) : new FrameConv2d<T>(dim, dim, 3, 2, 0);
        if (mode == WanResampleMode.Upsample3d) _timeConv = new WanCausalConv3d<T>(dim, 2 * dim, 3, 1, 1, 1, 0);
        else if (mode == WanResampleMode.Downsample3d) _timeConv = new WanCausalConv3d<T>(dim, dim, 3, 1, 2, 0, 0);
    }

    public Tensor<T> Forward(Tensor<T> x, WanFeatureCache<T>? cache)
    {
        if (_mode == WanResampleMode.Upsample3d && cache is not null && _timeConv is not null)
        {
            var slot = cache.Next();
            if (slot.IsEmpty) slot.Repeat = true;
            else
            {
                var s = x.Shape;
                int t = s[2];
                var next = Frames(x, Math.Max(0, t - 2), Math.Min(2, t));
                if (next.Shape[2] < 2)
                {
                    var front = slot.Tensor is null
                        ? new Tensor<T>(next.Shape.ToArray())
                        : Frames(slot.Tensor, slot.Tensor.Shape[2] - 1, 1);
                    next = Engine.TensorConcatenate(new[] { front, next }, 2);
                }
                x = _timeConv.Forward(x, slot.Repeat ? null : slot.Tensor);
                slot.Tensor = next;
                slot.Repeat = false;
                // [B, 2C, T, H, W] as [B, 2, C, T, H, W], interleaved to [B, C, 2T, H, W].
                var pairs = Engine.Reshape(x, new[] { s[0], 2, s[1], t, s[3], s[4] });
                x = Engine.Reshape(Engine.TensorPermute(pairs, new[] { 0, 2, 3, 1, 4, 5 }), new[] { s[0], s[1], 2 * t, s[3], s[4] });
            }
        }

        if (_mode is WanResampleMode.Upsample2d or WanResampleMode.Upsample3d)
            x = _conv.Forward(Nearest2x(x));
        else
            x = _conv.Forward(Engine.PadNd(x, new[] { 0, 1, 0, 1 }, PadMode.Constant, NumOps.Zero));

        if (_mode == WanResampleMode.Downsample3d && cache is not null && _timeConv is not null)
        {
            var slot = cache.Next();
            if (slot.Tensor is null) slot.Tensor = x;
            else
            {
                var next = Frames(x, x.Shape[2] - 1, 1);
                var previous = Frames(slot.Tensor, slot.Tensor.Shape[2] - 1, 1);
                x = _timeConv.Forward(Engine.TensorConcatenate(new[] { previous, x }, 2), null);
                slot.Tensor = next;
            }
        }
        return x;
    }

    // Nearest-exact x2 in height and width (exact at an integer scale).
    private static Tensor<T> Nearest2x(Tensor<T> x)
    {
        var s = x.Shape;
        var wide = Engine.TensorBroadcastTo(Engine.Reshape(x, new[] { s[0], s[1], s[2], s[3], 1, s[4], 1 }),
            new[] { s[0], s[1], s[2], s[3], 2, s[4], 2 });
        return Engine.Reshape(wide, new[] { s[0], s[1], s[2], 2 * s[3], 2 * s[4] });
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
        => _conv.NamedTensors(Join(prefix, "resample.1")).Select(n => (n.Name.Replace(".conv.", "."), n.Tensor))
            .Concat(_timeConv?.NamedTensors(Join(prefix, "time_conv")) ?? Enumerable.Empty<(string, Tensor<T>)>());

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _conv;
        yield return _timeConv;
    }
}

/// <summary>
/// <c>Encoder3d</c> with Wan 2.1's temporal pattern: a causal 3x3x3 stem, residual stages joined by a 2D then 3D
/// downsamplings, a residual-attention-residual middle and an RMS norm, SiLU and causal head to twice the latent channels.
/// </summary>
internal sealed class WanEncoder3d<T> : WanVideoModule<T>
{
    private readonly WanCausalConv3d<T> _conv1, _head;
    private readonly List<WanVideoModule<T>> _downsamples = new();
    private readonly WanResidualBlock<T> _middle1, _middle2;
    private readonly WanAttentionBlock<T> _middleAttention;
    private readonly WanRmsNorm<T> _headNorm;

    public WanEncoder3d(int dim, int outChannels, IReadOnlyList<int> dimMult, int resBlocks)
    {
        var dims = new[] { dim }.Concat(dimMult.Select(u => dim * u)).ToArray();
        _conv1 = new WanCausalConv3d<T>(3, dims[0], 3, 3);
        for (int i = 0; i < dimMult.Count; i++)
        {
            int inDim = dims[i];
            for (int r = 0; r < resBlocks; r++)
            {
                _downsamples.Add(new WanResidualBlock<T>(inDim, dims[i + 1]));
                inDim = dims[i + 1];
            }
            if (i != dimMult.Count - 1)
                _downsamples.Add(new WanResample<T>(dims[i + 1], i == 0 ? WanResampleMode.Downsample2d : WanResampleMode.Downsample3d));
        }
        int last = dims[dims.Length - 1];
        _middle1 = new WanResidualBlock<T>(last, last);
        _middleAttention = new WanAttentionBlock<T>(last);
        _middle2 = new WanResidualBlock<T>(last, last);
        _headNorm = new WanRmsNorm<T>(last);
        _head = new WanCausalConv3d<T>(last, outChannels, 3, 3);
    }

    public Tensor<T> Forward(Tensor<T> x, WanFeatureCache<T>? cache)
    {
        x = Cached(_conv1, x, cache);
        foreach (var layer in _downsamples)
            x = layer is WanResidualBlock<T> block ? block.Forward(x, cache) : ((WanResample<T>)layer).Forward(x, cache);
        x = _middle2.Forward(_middleAttention.Forward(_middle1.Forward(x, cache)), cache);
        return Cached(_head, Engine.Swish(_headNorm.Forward(x)), cache);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        var all = _conv1.NamedTensors(Join(prefix, "conv1"));
        for (int i = 0; i < _downsamples.Count; i++) all = all.Concat(_downsamples[i].NamedTensors(Join(prefix, $"downsamples.{i}")));
        return all.Concat(_middle1.NamedTensors(Join(prefix, "middle.0"))).Concat(_middleAttention.NamedTensors(Join(prefix, "middle.1")))
            .Concat(_middle2.NamedTensors(Join(prefix, "middle.2"))).Concat(_headNorm.NamedTensors(Join(prefix, "head.0")))
            .Concat(_head.NamedTensors(Join(prefix, "head.2")));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _conv1;
        foreach (var layer in _downsamples) yield return layer;
        yield return _middle1;
        yield return _middleAttention;
        yield return _middle2;
        yield return _headNorm;
        yield return _head;
    }
}

/// <summary>
/// <c>Decoder3d</c>: the encoder mirrored, three residual blocks a stage, 3D then 2D upsamplings that halve the width the
/// next stage starts from, and an RMS norm, SiLU and causal head to RGB.
/// </summary>
internal sealed class WanDecoder3d<T> : WanVideoModule<T>
{
    private readonly WanCausalConv3d<T> _conv1, _head;
    private readonly List<WanVideoModule<T>> _upsamples = new();
    private readonly WanResidualBlock<T> _middle1, _middle2;
    private readonly WanAttentionBlock<T> _middleAttention;
    private readonly WanRmsNorm<T> _headNorm;

    public WanDecoder3d(int dim, int latentChannels, IReadOnlyList<int> dimMult, int resBlocks)
    {
        var dims = new[] { dimMult[dimMult.Count - 1] }.Concat(dimMult.Reverse()).Select(u => dim * u).ToArray();
        _conv1 = new WanCausalConv3d<T>(latentChannels, dims[0], 3, 3);
        _middle1 = new WanResidualBlock<T>(dims[0], dims[0]);
        _middleAttention = new WanAttentionBlock<T>(dims[0]);
        _middle2 = new WanResidualBlock<T>(dims[0], dims[0]);
        for (int i = 0; i < dimMult.Count; i++)
        {
            int inDim = i > 0 ? dims[i] / 2 : dims[i];
            for (int r = 0; r < resBlocks + 1; r++)
            {
                _upsamples.Add(new WanResidualBlock<T>(inDim, dims[i + 1]));
                inDim = dims[i + 1];
            }
            if (i != dimMult.Count - 1)
                _upsamples.Add(new WanResample<T>(dims[i + 1], i < dimMult.Count - 2 ? WanResampleMode.Upsample3d : WanResampleMode.Upsample2d));
        }
        int last = dims[dims.Length - 1];
        _headNorm = new WanRmsNorm<T>(last);
        _head = new WanCausalConv3d<T>(last, 3, 3, 3);
    }

    public Tensor<T> Forward(Tensor<T> x, WanFeatureCache<T>? cache)
    {
        x = Cached(_conv1, x, cache);
        x = _middle2.Forward(_middleAttention.Forward(_middle1.Forward(x, cache)), cache);
        foreach (var layer in _upsamples)
            x = layer is WanResidualBlock<T> block ? block.Forward(x, cache) : ((WanResample<T>)layer).Forward(x, cache);
        return Cached(_head, Engine.Swish(_headNorm.Forward(x)), cache);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
    {
        var all = _conv1.NamedTensors(Join(prefix, "conv1")).Concat(_middle1.NamedTensors(Join(prefix, "middle.0")))
            .Concat(_middleAttention.NamedTensors(Join(prefix, "middle.1"))).Concat(_middle2.NamedTensors(Join(prefix, "middle.2")));
        for (int i = 0; i < _upsamples.Count; i++) all = all.Concat(_upsamples[i].NamedTensors(Join(prefix, $"upsamples.{i}")));
        return all.Concat(_headNorm.NamedTensors(Join(prefix, "head.0"))).Concat(_head.NamedTensors(Join(prefix, "head.2")));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _conv1;
        yield return _middle1;
        yield return _middleAttention;
        yield return _middle2;
        foreach (var layer in _upsamples) yield return layer;
        yield return _headNorm;
        yield return _head;
    }
}

/// <summary>
/// Wan 2.1's video VAE (<c>WanVideoVAE</c>, z_dim 16): 8x spatial and 4x temporal compression with a causal layout, the
/// first frame alone in the first latent. It encodes the first frame and then four frames at a time, decodes one latent
/// at a time, carrying each convolution's cache, and normalizes the posterior mean with the released per-channel
/// statistics.
/// </summary>
internal sealed class WanVideoVae<T> : WanVideoModule<T>
{
    /// <summary>The latent channels; the released statistics are for 16.</summary>
    internal const int LatentChannels = 16;

    private static readonly double[] Mean =
    {
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    };

    private static readonly double[] Std =
    {
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    };

    private readonly WanEncoder3d<T> _encoder;
    private readonly WanCausalConv3d<T> _quant, _postQuant;
    private readonly WanDecoder3d<T> _decoder;

    public WanVideoVae(int dim, IReadOnlyList<int> dimMult, int resBlocks)
    {
        if (dimMult.Count < 3) throw new ArgumentOutOfRangeException(nameof(dimMult), "Wan's VAE needs two temporal stages after the first.");
        _encoder = new WanEncoder3d<T>(dim, 2 * LatentChannels, dimMult, resBlocks);
        _quant = new WanCausalConv3d<T>(2 * LatentChannels, 2 * LatentChannels, 1, 1);
        _postQuant = new WanCausalConv3d<T>(LatentChannels, LatentChannels, 1, 1);
        _decoder = new WanDecoder3d<T>(dim, LatentChannels, dimMult, resBlocks);
        SpatialFactor = 1 << (dimMult.Count - 1);
    }

    /// <summary>The spatial compression.</summary>
    internal int SpatialFactor { get; }

    private static Tensor<T> Statistic(double[] values, int rank)
    {
        var shape = new int[rank];
        for (int i = 0; i < rank; i++) shape[i] = 1;
        shape[1] = LatentChannels;
        return new Tensor<T>(shape, new Vector<T>(values.Select(v => NumOps.FromDouble(v)).ToArray()));
    }

    /// <summary>The normalized posterior mean of [B, 3, 1 + 4k, H, W] in [-1, 1]: [B, 16, 1 + k, H / f, W / f].</summary>
    public Tensor<T> Encode(Tensor<T> video)
    {
        var cache = new WanFeatureCache<T>();
        int t = video.Shape[2], chunks = 1 + (t - 1) / 4;
        var outputs = new Tensor<T>[chunks];
        for (int i = 0; i < chunks; i++)
        {
            cache.BeginChunk();
            outputs[i] = _encoder.Forward(i == 0 ? Frames(video, 0, 1) : Frames(video, 1 + 4 * (i - 1), 4), cache);
        }
        var moments = _quant.Forward(Engine.TensorConcatenate(outputs, 2), null);
        var mu = Engine.TensorNarrow(moments, 1, 0, LatentChannels);
        var shape = mu.Shape.ToArray();
        var centred = Engine.TensorSubtract(mu, Engine.TensorBroadcastTo(Statistic(Mean, 5), shape));
        return Engine.TensorDivide(centred, Engine.TensorBroadcastTo(Statistic(Std, 5), shape));
    }

    /// <summary>Normalized latents [B, 16, n, h, w] to video [B, 3, 4n − 3, f·h, f·w].</summary>
    public Tensor<T> Decode(Tensor<T> latents)
    {
        var shape = latents.Shape.ToArray();
        var z = Engine.TensorAdd(Engine.TensorMultiply(latents, Engine.TensorBroadcastTo(Statistic(Std, 5), shape)),
            Engine.TensorBroadcastTo(Statistic(Mean, 5), shape));
        z = _postQuant.Forward(z, null);
        var cache = new WanFeatureCache<T>();
        var outputs = new Tensor<T>[shape[2]];
        for (int i = 0; i < shape[2]; i++)
        {
            cache.BeginChunk();
            outputs[i] = _decoder.Forward(Frames(z, i, 1), cache);
        }
        return Engine.TensorConcatenate(outputs, 2);
    }

    public override IEnumerable<(string Name, Tensor<T> Tensor)> NamedTensors(string prefix)
        => _encoder.NamedTensors(Join(prefix, "encoder")).Concat(_quant.NamedTensors(Join(prefix, "conv1")))
            .Concat(_postQuant.NamedTensors(Join(prefix, "conv2"))).Concat(_decoder.NamedTensors(Join(prefix, "decoder")));

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _encoder;
        yield return _quant;
        yield return _postQuant;
        yield return _decoder;
    }
}
