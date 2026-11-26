using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>Shared pieces of RVRT (Liang et al. 2022; JingyunLiang/RVRT, models/network_rvrt.py).</summary>
internal static class RvrtBlocks
{
    /// <summary>nn.Linear with PyTorch's default initialization (RVRT never re-initializes its weights).</summary>
    internal static DenseLayer<T> Linear<T>(int outputs, IActivationFunction<T>? activation = null, bool bias = true,
        IInitializationStrategy<T>? init = null)
        => new(outputs, activation, init ?? new TorchDefaultInitializationStrategy<T>(), bias ? BiasMode.Always : BiasMode.Never);

    /// <summary>nn.LayerNorm(dim) at PyTorch's epsilon.</summary>
    internal static LayerNormalizationLayer<T> Norm<T>(int dim) => new(dim, 1e-5);

    /// <summary>[n, d, c, h, w] to [n, d, h, w, c].</summary>
    internal static Tensor<T> ChannelsLast<T>(IEngine engine, Tensor<T> x) => engine.TensorPermute(x, new[] { 0, 1, 3, 4, 2 });

    /// <summary>[n, d, h, w, c] to [n, d, c, h, w].</summary>
    internal static Tensor<T> ChannelsFirst<T>(IEngine engine, Tensor<T> x) => engine.TensorPermute(x, new[] { 0, 1, 4, 2, 3 });

    /// <summary>A per-frame 2-D layer over [n, d, c, h, w] (RVRT's Conv3d with a (1, k, k) kernel).</summary>
    internal static Tensor<T> PerFrame<T>(IEngine engine, IReadOnlyList<ILayer<T>> layers, int index, Tensor<T> x)
    {
        int n = x.Shape[0], d = x.Shape[1];
        var y = layers[index].Forward(engine.Reshape(x, new[] { n * d, x.Shape[2], x.Shape[3], x.Shape[4] }));
        return engine.Reshape(y, new[] { n, d, y.Shape[1], y.Shape[2], y.Shape[3] });
    }

    /// <summary>Zero padding at the end of <paramref name="axis"/>.</summary>
    internal static Tensor<T> PadEnd<T>(IEngine engine, Tensor<T> x, int axis, int extra)
    {
        if (extra == 0) return x;
        var shape = x.Shape.ToArray();
        shape[axis] = extra;
        return engine.TensorConcatenate(new[] { x, new Tensor<T>(shape) }, axis);
    }

    /// <summary>get_window_size: a dimension no larger than its window takes the whole dimension and no shift.</summary>
    internal static (int[] Window, int[] Shift) WindowFor(int[] size, int[] window, int[] shift)
    {
        var w = (int[])window.Clone();
        var s = (int[])shift.Clone();
        for (int i = 0; i < 3; i++)
            if (size[i] <= window[i])
            {
                w[i] = size[i];
                s[i] = 0;
            }
        return (w, s);
    }

    /// <summary>window_partition: [B, D, H, W, C] to [B·nW, Wd·Wh·Ww, C].</summary>
    internal static Tensor<T> Partition<T>(IEngine engine, Tensor<T> x, int[] w)
    {
        int b = x.Shape[0], d = x.Shape[1], h = x.Shape[2], wd = x.Shape[3], c = x.Shape[4];
        var v = engine.Reshape(x, new[] { b, d / w[0], w[0], h / w[1], w[1], wd / w[2], w[2], c });
        v = engine.TensorPermute(v, new[] { 0, 1, 3, 5, 2, 4, 6, 7 });
        return engine.Reshape(v, new[] { b * (d / w[0]) * (h / w[1]) * (wd / w[2]), w[0] * w[1] * w[2], c });
    }

    /// <summary>window_reverse: [B·nW, N, C] back to [B, D, H, W, C].</summary>
    internal static Tensor<T> Reverse<T>(IEngine engine, Tensor<T> windows, int[] w, int b, int d, int h, int wd)
    {
        int c = windows.Shape[2];
        var v = engine.Reshape(windows, new[] { b, d / w[0], h / w[1], wd / w[2], w[0], w[1], w[2], c });
        v = engine.TensorPermute(v, new[] { 0, 1, 4, 2, 5, 3, 6, 7 });
        return engine.Reshape(v, new[] { b, d, h, wd, c });
    }

    // Python's slice(start, stop) over a dimension of length n (negative counts from the end; -0 is 0).
    private static (int From, int To) Slice(int? start, int? stop, int n)
    {
        int Resolve(int v) => Math.Max(0, Math.Min(n, v < 0 ? v + n : v));
        return (start is int a ? Resolve(a) : 0, stop is int b ? Resolve(b) : n);
    }

    /// <summary>compute_mask: -100 between tokens of a shifted window that came from different regions, else 0.</summary>
    internal static double[] ShiftMask(int d, int h, int w, int[] window, int[] shift, out int windows)
    {
        var region = new int[d * h * w];
        int count = 0;
        var dims = new[] { d, h, w };
        var ranges = new (int From, int To)[3][];
        for (int a = 0; a < 3; a++)
            ranges[a] = new[]
            {
                Slice(null, -window[a], dims[a]),
                Slice(-window[a], -shift[a], dims[a]),
                Slice(-shift[a], null, dims[a]),
            };
        foreach (var rd in ranges[0])
            foreach (var rh in ranges[1])
                foreach (var rw in ranges[2])
                {
                    for (int z = rd.From; z < rd.To; z++)
                        for (int y = rh.From; y < rh.To; y++)
                            for (int x = rw.From; x < rw.To; x++) region[(z * h + y) * w + x] = count;
                    count++;
                }

        int n = window[0] * window[1] * window[2];
        windows = d / window[0] * (h / window[1]) * (w / window[2]);
        var mask = new double[windows * n * n];
        int index = 0;
        var tokens = new int[n];
        for (int bz = 0; bz < d / window[0]; bz++)
            for (int by = 0; by < h / window[1]; by++)
                for (int bx = 0; bx < w / window[2]; bx++, index++)
                {
                    int t = 0;
                    for (int z = 0; z < window[0]; z++)
                        for (int y = 0; y < window[1]; y++)
                            for (int x = 0; x < window[2]; x++)
                                tokens[t++] = region[((bz * window[0] + z) * h + by * window[1] + y) * w + bx * window[2] + x];
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            mask[(index * n + i) * n + j] = tokens[j] - tokens[i] != 0 ? -100.0 : 0.0;
                }
        return mask;
    }

    /// <summary>get_position_index: the relative-position table row of every token pair of a full window.</summary>
    internal static int[,] PositionIndex(int[] window)
    {
        int n = window[0] * window[1] * window[2];
        var coords = new int[n, 3];
        int t = 0;
        for (int z = 0; z < window[0]; z++)
            for (int y = 0; y < window[1]; y++)
                for (int x = 0; x < window[2]; x++, t++)
                {
                    coords[t, 0] = z;
                    coords[t, 1] = y;
                    coords[t, 2] = x;
                }
        var index = new int[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                int dz = coords[i, 0] - coords[j, 0] + window[0] - 1;
                int dy = coords[i, 1] - coords[j, 1] + window[1] - 1;
                int dx = coords[i, 2] - coords[j, 2] + window[2] - 1;
                index[i, j] = dz * (2 * window[1] - 1) * (2 * window[2] - 1) + dy * (2 * window[2] - 1) + dx;
            }
        return index;
    }
}

/// <summary>
/// RVRT's WindowAttention: multi-head self-attention inside a 3-D window with a learned relative-position bias (a zero-
/// initialized table read through the full window's position index, sliced to the actual token count as the release
/// does) and the shifted-window mask.
/// </summary>
internal sealed class RvrtWindowAttention<T>
{
    private readonly int _qkv, _proj, _table, _heads, _dim, _tableSize;
    private readonly int[,] _index;

    internal RvrtWindowAttention(List<ILayer<T>> layers, int dim, int[] window, int heads, bool qkvBias,
        IInitializationStrategy<T>? linearInit = null, IInitializationStrategy<T>? tableInit = null)
    {
        _dim = dim;
        _heads = heads;
        _tableSize = (2 * window[0] - 1) * (2 * window[1] - 1) * (2 * window[2] - 1);
        _index = RvrtBlocks.PositionIndex(window);
        _table = layers.Count;
        // The table is a bias-free [table, heads] matrix, read out through the identity and gathered in Forward.
        layers.Add(RvrtBlocks.Linear<T>(heads, bias: false, init: tableInit ?? new ZeroInitializationStrategy<T>()));
        _qkv = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(3 * dim, bias: qkvBias, init: linearInit));
        _proj = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim, init: linearInit));
    }

    private Tensor<T>? _identity;
    private Tensor<int>? _pairIndex;

    private Tensor<T> Identity(INumericOperations<T> ops)
    {
        if (_identity is not null) return _identity;
        var identity = new Tensor<T>(new[] { _tableSize, _tableSize });
        for (int i = 0; i < _tableSize; i++) identity[i * _tableSize + i] = ops.One;
        return _identity = identity;
    }

    // The table row of each of the first n tokens' pairs (the release slices the full window's index to n).
    private Tensor<int> PairIndex(int n)
    {
        if (_pairIndex is { Length: var length } && length == n * n) return _pairIndex;
        var index = new Tensor<int>(new[] { n * n });
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) index[i * n + j] = _index[i, j];
        return _pairIndex = index;
    }

    /// <param name="windows">[B·nW, N, C].</param>
    /// <param name="mask">[nW, N, N] or null.</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> windows, double[]? mask, int maskWindows)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = windows.Shape[0], n = windows.Shape[1], hd = _dim / _heads;
        var qkv = engine.TensorPermute(engine.Reshape(layers[_qkv].Forward(windows), new[] { b, n, 3, _heads, hd }), new[] { 2, 0, 3, 1, 4 });
        Tensor<T> Part(int i) => engine.Reshape(engine.TensorNarrow(qkv, 0, i, 1), new[] { b * _heads, n, hd });
        var q = engine.TensorMultiplyScalar(Part(0), ops.FromDouble(Math.Pow(hd, -0.5)));
        var k = Part(1);
        var v = Part(2);
        var attn = engine.Reshape(engine.TensorBatchMatMul(q, engine.TensorPermute(k, new[] { 0, 2, 1 })), new[] { b, _heads, n, n });

        // relative_position_bias_table[relative_position_index]: the [table, heads] matrix is read out once through
        // the identity and its rows gathered per token pair, so the tape scatters the gradient into the table rows.
        var table = layers[_table].Forward(Identity(ops));
        var bias = engine.TensorPermute(engine.Reshape(engine.TensorIndexSelectDiff(table, PairIndex(n), 0), new[] { n, n, _heads }),
            new[] { 2, 0, 1 });
        attn = engine.TensorAdd(attn, engine.Reshape(bias, new[] { 1, _heads, n, n }));

        if (mask is not null)
        {
            var m = new Tensor<T>(new[] { 1, maskWindows, 1, n, n });
            for (int i = 0; i < mask.Length; i++) m[i] = ops.FromDouble(mask[i]);
            attn = engine.Reshape(engine.TensorAdd(engine.Reshape(attn, new[] { b / maskWindows, maskWindows, _heads, n, n }), m),
                new[] { b, _heads, n, n });
        }

        attn = engine.TensorSoftmax(attn, 3);
        var x = engine.TensorBatchMatMul(engine.Reshape(attn, new[] { b * _heads, n, n }), v);
        x = engine.Reshape(engine.TensorPermute(engine.Reshape(x, new[] { b, _heads, n, hd }), new[] { 0, 2, 1, 3 }), new[] { b, n, _dim });
        return layers[_proj].Forward(x);
    }
}

/// <summary>
/// RVRT's STL (and PSRT's SwinTransformerBlock): pre-norm shifted-window attention and a GELU MLP, each residual.
/// RVRT shrinks a window to a smaller input and drops its shift; PSRT keeps the window and pads, and adds stochastic
/// depth (DropPath) to both branches in training.
/// </summary>
internal sealed class RvrtSwinLayer<T>
{
    private readonly int _norm1, _norm2, _fc1, _fc2;
    private readonly int[] _window, _shift;
    private readonly bool _clampWindow;
    private readonly double _dropPath;
    private readonly Random? _random;
    private readonly RvrtWindowAttention<T> _attention;

    internal RvrtSwinLayer(List<ILayer<T>> layers, int dim, int heads, int[] window, int[] shift, double mlpRatio, bool qkvBias,
        bool clampWindow = true, IInitializationStrategy<T>? linearInit = null, IInitializationStrategy<T>? tableInit = null,
        double dropPath = 0, Random? random = null)
    {
        _window = window;
        _shift = shift;
        _clampWindow = clampWindow;
        _dropPath = dropPath;
        _random = random;
        _norm1 = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        _attention = new RvrtWindowAttention<T>(layers, dim, window, heads, qkvBias, linearInit, tableInit);
        _norm2 = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        _fc1 = layers.Count;
        // nn.GELU is the erf form; the library's GELU is the tanh approximation (|difference| < 1e-3).
        layers.Add(RvrtBlocks.Linear<T>((int)(dim * mlpRatio), new GELUActivation<T>(), init: linearInit));
        _fc2 = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim, init: linearInit));
    }

    // timm's DropPath: each sample's branch is dropped with probability p and the survivors scaled by 1 / (1 - p).
    private Tensor<T> DropPath(IEngine engine, Tensor<T> branch, bool training)
    {
        if (!training || _dropPath <= 0 || _random is null) return branch;
        var ops = MathHelper.GetNumericOperations<T>();
        // The per-sample factors are a per-step draw, so a compiled training plan redraws them on every replay.
        var random = _random;
        double drop = _dropPath;
        var shape = new int[branch.Rank];
        shape[0] = branch.Shape[0];
        for (int i = 1; i < shape.Length; i++) shape[i] = 1;
        var keep = AiDotNet.Training.CompiledStepRandom<T>.Draw(shape, m =>
        {
            var span = m.AsWritableSpan();
            for (int i = 0; i < span.Length; i++) span[i] = random.NextDouble() < 1 - drop ? ops.FromDouble(1 / (1 - drop)) : ops.Zero;
        });
        return engine.TensorMultiply(branch, engine.TensorBroadcastTo(keep, branch.Shape.ToArray()));
    }

    /// <param name="x">[B, D, H, W, C].</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, bool training = false)
    {
        int b = x.Shape[0], d = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        var (window, shift) = _clampWindow
            ? RvrtBlocks.WindowFor(new[] { d, h, w }, _window, _shift)
            : ((int[])_window.Clone(), (int[])_shift.Clone());

        var y = layers[_norm1].Forward(x);
        int dp = (d + window[0] - 1) / window[0] * window[0];
        int hp = (h + window[1] - 1) / window[1] * window[1];
        int wp = (w + window[2] - 1) / window[2] * window[2];
        y = RvrtBlocks.PadEnd(engine, RvrtBlocks.PadEnd(engine, RvrtBlocks.PadEnd(engine, y, 1, dp - d), 2, hp - h), 3, wp - w);

        bool shifted = shift[0] > 0 || shift[1] > 0 || shift[2] > 0;
        double[]? mask = null;
        int maskWindows = 1;
        if (shifted)
        {
            y = engine.TensorRoll(y, new[] { -shift[0], -shift[1], -shift[2] }, new[] { 1, 2, 3 });
            mask = RvrtBlocks.ShiftMask(dp, hp, wp, window, shift, out maskWindows);
        }

        var attended = _attention.Forward(engine, layers, RvrtBlocks.Partition(engine, y, window), mask, maskWindows);
        y = RvrtBlocks.Reverse(engine, attended, window, b, dp, hp, wp);
        if (shifted) y = engine.TensorRoll(y, new[] { shift[0], shift[1], shift[2] }, new[] { 1, 2, 3 });
        if (dp != d || hp != h || wp != w)
            y = engine.TensorNarrow(engine.TensorNarrow(engine.TensorNarrow(y, 1, 0, d), 2, 0, h), 3, 0, w);

        x = engine.TensorAdd(x, DropPath(engine, y, training));
        return engine.TensorAdd(x, DropPath(engine, layers[_fc2].Forward(layers[_fc1].Forward(layers[_norm2].Forward(x))), training));
    }
}

/// <summary>
/// RVRT's RSTBWithInputConv: a (1, 3, 3) convolution to the embedding width, LayerNorm, residual Swin blocks (each a
/// group of alternately shifted layers and a residual linear), and LayerNorm.
/// </summary>
internal sealed class RvrtStage<T>
{
    private readonly int _conv, _normIn, _normOut;
    private readonly List<(RvrtSwinLayer<T>[] Layers, int Linear)> _blocks = new();

    internal RvrtStage(List<ILayer<T>> layers, int dim, int numBlocks, int depth, int heads, int[] window, double mlpRatio)
    {
        _conv = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(dim, 3, null));
        _normIn = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
        var shift = window.Select(v => v / 2).ToArray();
        for (int blk = 0; blk < numBlocks; blk++)
        {
            var group = new RvrtSwinLayer<T>[depth];
            for (int i = 0; i < depth; i++)
                group[i] = new RvrtSwinLayer<T>(layers, dim, heads, window, i % 2 == 0 ? new[] { 0, 0, 0 } : shift, mlpRatio, true);
            int linear = layers.Count;
            layers.Add(RvrtBlocks.Linear<T>(dim));
            _blocks.Add((group, linear));
        }
        _normOut = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(dim));
    }

    /// <param name="x">[n, d, c_in, h, w].</param>
    /// <returns>[n, d, dim, h, w].</returns>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x)
    {
        var y = layers[_normIn].Forward(RvrtBlocks.ChannelsLast(engine, RvrtBlocks.PerFrame(engine, layers, _conv, x)));
        foreach (var (group, linear) in _blocks)
        {
            var z = y;
            foreach (var layer in group) z = layer.Forward(engine, layers, z);
            y = engine.TensorAdd(y, layers[linear].Forward(z));
        }
        return RvrtBlocks.ChannelsFirst(engine, layers[_normOut].Forward(y));
    }
}
