using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>The release's operators (VFIformer_arch.py, transformer_layers.py).</summary>
internal static class VfiformerOps
{
    /// <summary>nn.LeakyReLU(0.2).</summary>
    internal static Tensor<T> LeakyRelu<T>(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorSubtract(engine.ReLU(x), engine.TensorMultiplyScalar(engine.ReLU(engine.TensorNegate(x)), ops.FromDouble(0.2)));
    }

    /// <summary>F.unfold(kernel_size=3, padding=1) viewed [B, C, 9, H, W]: each pixel's zero-padded 3x3 neighbourhood.</summary>
    internal static Tensor<T> Unfold3<T>(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        var padded = engine.PadNd(x, new[] { 1, 1, 1, 1 }, PadMode.Constant, ops.Zero);
        var taps = new Tensor<T>[9];
        for (int k = 0; k < 9; k++)
            taps[k] = engine.Reshape(engine.TensorSlice(padded, new[] { 0, 0, k / 3, k % 3 }, new[] { b, c, h, w }), new[] { b, c, 1, h, w });
        return engine.TensorConcatenate(taps, 2);
    }

    // The nine shifted-window regions of an H x W map, row band then column band, as calculate_mask labels them.
    private static double[] Regions(int h, int w, int window, int shift)
    {
        var ids = new double[h * w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int row = y < h - window ? 0 : y < h - shift ? 1 : 2;
                int col = x < w - window ? 0 : x < w - shift ? 1 : 2;
                ids[y * w + x] = row * 3.0 + col;
            }
        return ids;
    }

    /// <summary>TFL.calculate_mask: [nW, N, N], −100 between two pixels of a window from different regions.</summary>
    internal static double[] ShiftMask(int h, int w, int window, int shift)
    {
        var ids = Regions(h, w, window, shift);
        int nh = h / window, nw = w / window, n = window * window;
        var mask = new double[nh * nw * n * n];
        for (int win = 0; win < nh * nw; win++)
        {
            int wy = win / nw, wx = win % nw;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double a = ids[(wy * window + i / window) * w + wx * window + i % window];
                    double b = ids[(wy * window + j / window) * w + wx * window + j % window];
                    if (a != b) mask[(win * n + i) * n + j] = -100.0;
                }
        }
        return mask;
    }

    /// <summary>
    /// TFL.calculate_mask2: [nW, N, N], −100 where a window pixel's region differs from the (bilinearly halved,
    /// reflect-padded by window/4, unfolded at stride window/2) region value of the coarse window's pixel.
    /// </summary>
    internal static double[] CrossMask(int h, int w, int window, int shift)
    {
        var ids = Regions(h, w, window, shift);
        int hd = h / 2, wd = w / 2, pad = window / 4, hp = hd + 2 * pad, wp = wd + 2 * pad;
        var down = new double[hd * wd];
        for (int y = 0; y < hd; y++)
            for (int x = 0; x < wd; x++)
                down[y * wd + x] = (ids[2 * y * w + 2 * x] + ids[2 * y * w + 2 * x + 1] + ids[(2 * y + 1) * w + 2 * x] + ids[(2 * y + 1) * w + 2 * x + 1]) / 4;
        int Reflect(int i, int n) => i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i;
        int nh = h / window, nw = w / window, n = window * window, stride = window / 2;
        var mask = new double[nh * nw * n * n];
        for (int win = 0; win < nh * nw; win++)
        {
            int wy = win / nw, wx = win % nw;
            for (int i = 0; i < n; i++)
            {
                double a = ids[(wy * window + i / window) * w + wx * window + i % window];
                for (int j = 0; j < n; j++)
                {
                    int py = wy * stride + j / window - pad, px = wx * stride + j % window - pad;
                    double b = down[Reflect(py, hd) * wd + Reflect(px, wd)];
                    if (b - a != 0) mask[(win * n + i) * n + j] = -100.0;
                }
            }
        }
        return mask;
    }

    /// <summary>
    /// The coarse windows a cross-scale TFL attends to: the map [B, H, W, C] bilinearly halved, reflect-padded by
    /// window/4 and unfolded into window x window patches at stride window/2, one per window of the full map.
    /// </summary>
    internal static Tensor<T> CoarseWindows<T>(IEngine engine, Tensor<T> x, int window)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = x.Shape[0], h = x.Shape[1], w = x.Shape[2], c = x.Shape[3], half = window / 2, nh = h / window, nw = w / window;
        var channels = engine.TensorPermute(x, new[] { 0, 3, 1, 2 }).Contiguous();
        var down = AmtOps.Resize(engine, channels, h / 2, w / 2);
        var padded = engine.PadNd(down, new[] { window / 4, window / 4, window / 4, window / 4 }, PadMode.Reflect, ops.Zero);
        // Each patch is a 2x2 block of half-window tiles: tiles (wy, wx) to (wy + 1, wx + 1).
        var tiles = engine.Reshape(padded.Contiguous(), new[] { b, c, nh + 1, half, nw + 1, half });
        Tensor<T> Tiles(int dy, int dx) => engine.TensorSlice(tiles, new[] { 0, 0, dy, 0, dx, 0 }, new[] { b, c, nh, half, nw, half });
        var top = engine.TensorConcatenate(new[] { Tiles(0, 0), Tiles(0, 1) }, 5);
        var bottom = engine.TensorConcatenate(new[] { Tiles(1, 0), Tiles(1, 1) }, 5);
        var patches = engine.TensorConcatenate(new[] { top, bottom }, 3);
        return engine.Reshape(engine.TensorPermute(patches, new[] { 0, 2, 4, 3, 5, 1 }).Contiguous(), new[] { b * nh * nw, window * window, c });
    }

    /// <summary>The stochastic-depth scale of one branch: each sample kept with probability 1 − p and scaled by 1/(1 − p).</summary>
    internal static Tensor<T> DropPath<T>(IEngine engine, Tensor<T> branch, double p, Random? random)
    {
        if (p <= 0 || random is null) return branch;
        var ops = MathHelper.GetNumericOperations<T>();
        var samples = new Tensor<T>[branch.Shape[0]];
        for (int i = 0; i < samples.Length; i++)
            samples[i] = engine.TensorMultiplyScalar(engine.TensorNarrow(branch, 0, i, 1), random.NextDouble() < 1 - p ? ops.FromDouble(1 / (1 - p)) : ops.Zero);
        return samples.Length == 1 ? samples[0] : engine.TensorConcatenate(samples, 0);
    }
}

/// <summary>IFBlock: the input bilinearly reduced by the scale, two stride-2 convolution-PReLU pairs, eight residual
/// pairs and a 4x4 stride-2 transposed convolution to four flow channels, enlarged back by the scale (half the input).</summary>
internal sealed class VfifIfBlock<T> : LazyWeightModule<T>
{
    private readonly int _scale;
    private readonly AmtConvPRelu<T> _down1, _down2;
    private readonly AmtConvPRelu<T>[] _body;
    private readonly AmtDeconv<T> _last;

    public VfifIfBlock(int inChannels, int scale, int c)
    {
        _scale = scale;
        _down1 = new AmtConvPRelu<T>(inChannels, c / 2, 3, 2);
        _down2 = new AmtConvPRelu<T>(c / 2, c, 3, 2);
        _body = Enumerable.Range(0, 8).Select(_ => new AmtConvPRelu<T>(c, c)).ToArray();
        _last = new AmtDeconv<T>(c, 4);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        int h = x.Shape[2], w = x.Shape[3];
        if (_scale != 1) x = AmtOps.Resize(Engine, x, h / _scale, w / _scale);
        x = _down2.Forward(_down1.Forward(x));
        var y = x;
        foreach (var layer in _body) y = layer.Forward(y);
        var flow = _last.Forward(Engine.TensorAdd(y, x));
        return _scale == 1 ? flow : AmtOps.Resize(Engine, flow, flow.Shape[2] * _scale, flow.Shape[3] * _scale);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _down1, _down2 }.Concat(_body).Concat(new LazyWeightModule<T>[] { _last });
}

/// <summary>IFNet: three IFBlocks (scales 4, 2, 1) whose flows (at half size) accumulate, each later block reading the
/// frames warped along the 2x-enlarged sum.</summary>
internal sealed class VfifIfNet<T> : LazyWeightModule<T>
{
    private readonly VfifIfBlock<T>[] _blocks;

    public VfifIfNet(IReadOnlyList<int> widths)
        => _blocks = new[] { new VfifIfBlock<T>(6, 4, widths[0]), new VfifIfBlock<T>(10, 2, widths[1]), new VfifIfBlock<T>(10, 1, widths[2]) };

    /// <summary>The accumulated flows F1, F2, F3 [N, 4, H/2, W/2] from frames [N, 3, H, W].</summary>
    public Tensor<T>[] Forward(Tensor<T> img0, Tensor<T> img1)
    {
        var flows = new Tensor<T>[3];
        var sum = _blocks[0].Forward(Engine.TensorConcatenate(new[] { img0, img1 }, 1));
        flows[0] = sum;
        for (int i = 1; i < 3; i++)
        {
            var large = Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, sum, sum.Shape[2] * 2, sum.Shape[3] * 2), NumOps.FromDouble(2.0));
            var warped0 = AmtOps.Warp(Engine, img0, Engine.TensorNarrow(large, 1, 0, 2));
            var warped1 = AmtOps.Warp(Engine, img1, Engine.TensorNarrow(large, 1, 2, 2));
            sum = Engine.TensorAdd(sum, _blocks[i].Forward(Engine.TensorConcatenate(new[] { warped0, warped1, large }, 1)));
            flows[i] = sum;
        }
        return flows;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _blocks;
}

/// <summary>A convolution with torch's default initialization followed by a per-channel PReLU, or without one.</summary>
internal sealed class VfifConvAct<T> : LazyWeightModule<T>
{
    private readonly AmtConv2d<T> _conv;
    private readonly AmtPRelu<T>? _prelu;

    public VfifConvAct(int inChannels, int outChannels, int kernel, bool prelu = true)
    {
        _conv = new AmtConv2d<T>(inChannels, outChannels, kernel);
        if (prelu) _prelu = new AmtPRelu<T>(outChannels);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = _conv.Forward(x);
        return _prelu is null ? y : _prelu.Forward(y);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>?[] { _conv, _prelu };
}

/// <summary>
/// FlowRefineNetA: both maps' 3x3 neighbourhoods warped along their flows, an occlusion-weighted fusion of the centres,
/// the cosine correlation of the fusion with every neighbour, and a convolutional update of each flow.
/// </summary>
internal sealed class VfifRefineA<T> : LazyWeightModule<T>
{
    private readonly VfifConvAct<T>[] _occlusion, _correlation, _flow, _gru, _head;
    private readonly VfifConvAct<T> _motion;

    public VfifRefineA(int context, int c)
    {
        _occlusion = new[] { new VfifConvAct<T>(2 * context, c, 1), new VfifConvAct<T>(c, c, 1), new VfifConvAct<T>(c, 1, 1, false) };
        _correlation = new[] { new VfifConvAct<T>(9, c, 1), new VfifConvAct<T>(c, c, 1) };
        _flow = new[] { new VfifConvAct<T>(2, c, 3), new VfifConvAct<T>(c, c, 3) };
        _motion = new VfifConvAct<T>(2 * c, c, 3);
        _gru = new[] { new VfifConvAct<T>(c + 2 * context + 2, c, 3), new VfifConvAct<T>(c, c, 3) };
        _head = new[] { new VfifConvAct<T>(c, c, 3), new VfifConvAct<T>(c, 2, 3, false) };
    }

    private static Tensor<T> Run(IEnumerable<VfifConvAct<T>> layers, Tensor<T> x)
    {
        foreach (var layer in layers) x = layer.Forward(x);
        return x;
    }

    // x / √(Σ_c x² + 1e-12) over the channel axis.
    private Tensor<T> Normalize(Tensor<T> x)
    {
        var norm = Engine.TensorSqrt(Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorSquare(x), new[] { 1 }, true), NumOps.FromDouble(1e-12)));
        return Engine.TensorDivide(x, Engine.TensorBroadcastTo(norm, x.Shape.ToArray()));
    }

    public (Tensor<T> Flow0, Tensor<T> Flow1) Forward(Tensor<T> x0, Tensor<T> x1, Tensor<T> flow0, Tensor<T> flow1)
    {
        int b = x0.Shape[0], c = x0.Shape[1], h = x0.Shape[2], w = x0.Shape[3];
        Tensor<T> Contents(Tensor<T> x, Tensor<T> flow)
            => Engine.Reshape(AmtOps.Warp(Engine, Engine.Reshape(VfiformerOps.Unfold3(Engine, x), new[] { b, c * 9, h, w }), flow), new[] { b, c, 9, h, w });
        var contents0 = Contents(x0, flow0);
        var contents1 = Contents(x1, flow1);
        var fea0 = Engine.Reshape(Engine.TensorNarrow(contents0, 2, 4, 1).Contiguous(), new[] { b, c, h, w });
        var fea1 = Engine.Reshape(Engine.TensorNarrow(contents1, 2, 4, 1).Contiguous(), new[] { b, c, h, w });
        var occlusion = Engine.TensorBroadcastTo(Engine.Sigmoid(Run(_occlusion, Engine.TensorConcatenate(new[] { fea0, fea1 }, 1))), fea0.Shape.ToArray());
        var fea = Engine.TensorAdd(Engine.TensorMultiply(fea0, occlusion), Engine.TensorMultiply(fea1, Engine.ScalarMinusTensor(NumOps.One, occlusion)));
        var unit = Engine.TensorBroadcastTo(Engine.Reshape(Normalize(fea), new[] { b, c, 1, h, w }), new[] { b, c, 9, h, w });
        Tensor<T> Correlation(Tensor<T> contents) => Engine.ReduceSum(Engine.TensorMultiply(unit, Normalize(contents)), new[] { 1 }, false);
        var motion0 = _motion.Forward(Engine.TensorConcatenate(new[] { Run(_correlation, Correlation(contents0)), Run(_flow, flow0) }, 1));
        var motion1 = _motion.Forward(Engine.TensorConcatenate(new[] { Run(_correlation, Correlation(contents1)), Run(_flow, flow1) }, 1));
        var delta0 = Run(_head, Run(_gru, Engine.TensorConcatenate(new[] { fea, fea0, motion0, flow0 }, 1)));
        var delta1 = Run(_head, Run(_gru, Engine.TensorConcatenate(new[] { fea, fea1, motion1, flow1 }, 1)));
        return (Engine.TensorAdd(flow0, delta0), Engine.TensorAdd(flow1, delta1));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => _occlusion.Concat(_correlation).Concat(_flow).Concat(new[] { _motion }).Concat(_gru).Concat(_head);
}

/// <summary>Conv2: a (strided) and a plain 3x3 convolution, each with a per-channel PReLU.</summary>
internal sealed class VfifConv2<T> : LazyWeightModule<T>
{
    private readonly AmtConvPRelu<T> _conv1, _conv2;

    public VfifConv2(int inChannels, int outChannels, int stride)
    {
        _conv1 = new AmtConvPRelu<T>(inChannels, outChannels, 3, stride);
        _conv2 = new AmtConvPRelu<T>(outChannels, outChannels);
    }

    public Tensor<T> Forward(Tensor<T> x) => _conv2.Forward(_conv1.Forward(x));

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv1, _conv2 };
}

/// <summary>FlowRefineNet_Multis: a shared four-level feature pyramid, the flows refined coarse to fine from 1/8 size,
/// and both frames' pyramids warped along the refined flows (the transformer's context).</summary>
internal sealed class VfifRefine<T> : LazyWeightModule<T>
{
    private readonly VfifConv2<T>[] _pyramid;
    private readonly VfifRefineA<T>[] _refine;

    public VfifRefine(int c)
    {
        _pyramid = new[] { new VfifConv2<T>(3, c, 1), new VfifConv2<T>(c, 2 * c, 2), new VfifConv2<T>(2 * c, 4 * c, 2), new VfifConv2<T>(4 * c, 8 * c, 2) };
        _refine = new[] { new VfifRefineA<T>(c, c), new VfifRefineA<T>(2 * c, 2 * c), new VfifRefineA<T>(4 * c, 4 * c), new VfifRefineA<T>(8 * c, 8 * c) };
    }

    private Tensor<T>[] Pyramid(Tensor<T> x)
    {
        var levels = new Tensor<T>[4];
        for (int i = 0; i < 4; i++) levels[i] = x = _pyramid[i].Forward(x);
        return levels;
    }

    private Tensor<T> Scale(Tensor<T> flow, int h, int w, double factor)
        => Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flow, h, w), NumOps.FromDouble(factor));

    private Tensor<T>[] WarpLevels(Tensor<T>[] levels, Tensor<T> flow)
    {
        var outs = new Tensor<T>[levels.Length];
        for (int i = 0; i < levels.Length; i++)
        {
            outs[i] = AmtOps.Warp(Engine, levels[i], flow);
            flow = Scale(flow, flow.Shape[2] / 2, flow.Shape[3] / 2, 0.5);
        }
        return outs;
    }

    /// <summary>The refined flows [N, 4, H, W] from IFNet's (at half size) and both frames' warped pyramids.</summary>
    public (Tensor<T> Flow, Tensor<T>[] Context0, Tensor<T>[] Context1) Forward(Tensor<T> x0, Tensor<T> x1, Tensor<T> flow)
    {
        var s0 = Pyramid(x0);
        var s1 = Pyramid(x1);
        flow = Scale(flow, s0[3].Shape[2], s0[3].Shape[3], 0.25);
        for (int level = 3; level >= 0; level--)
        {
            if (level < 3) flow = Scale(flow, s0[level].Shape[2], s0[level].Shape[3], 2.0);
            var (f0, f1) = _refine[level].Forward(s0[level], s1[level], Engine.TensorNarrow(flow, 1, 0, 2), Engine.TensorNarrow(flow, 1, 2, 2));
            flow = Engine.TensorConcatenate(new[] { f0, f1 }, 1);
        }
        return (flow, WarpLevels(s0, Engine.TensorNarrow(flow, 1, 0, 2)), WarpLevels(s1, Engine.TensorNarrow(flow, 1, 2, 2)));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _pyramid.Cast<LazyWeightModule<T>>().Concat(_refine);
}

/// <summary>
/// WindowAttention and WindowCrossAttention: multi-head attention within each window with a learned relative position
/// bias; the cross form adds attention from the same queries to the coarse windows (with their own bias table) and merges
/// the two through a two-layer LeakyReLU projection with a residual.
/// </summary>
internal sealed class VfifAttention<T> : LazyWeightModule<T>
{
    private readonly int _heads, _window;
    private readonly bool _cross;
    private readonly EmaLinear<T> _qkv, _proj;
    private readonly EmaLinear<T>? _kv, _merge1, _merge2;
    private readonly LazyWeight<T> _tableX;
    private readonly LazyWeight<T>? _tableY;
    private readonly int[] _index;

    public VfifAttention(int dim, int window, int heads, bool cross)
    {
        _heads = heads;
        _window = window;
        _cross = cross;
        int span = 2 * window - 1;
        _tableX = EmaInit.Normal<T>(new[] { span * span, heads }, 0.02);
        _qkv = new EmaLinear<T>(dim, 3 * dim);
        if (cross)
        {
            _tableY = EmaInit.Normal<T>(new[] { span * span, heads }, 0.02);
            _kv = new EmaLinear<T>(dim, 2 * dim);
            _merge1 = new EmaLinear<T>(2 * dim, dim);
            _merge2 = new EmaLinear<T>(dim, dim);
        }
        _proj = new EmaLinear<T>(dim, dim);
        int n = window * window;
        _index = new int[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                _index[i * n + j] = (i / window - j / window + window - 1) * span + (i % window - j % window + window - 1);
    }

    // [B, N, heads·d] to [B·heads, N, d] and back.
    private Tensor<T> Heads(Tensor<T> x, int d)
    {
        int b = x.Shape[0], n = x.Shape[1];
        return Engine.Reshape(Engine.TensorPermute(Engine.Reshape(x, new[] { b, n, _heads, d }), new[] { 0, 2, 1, 3 }).Contiguous(), new[] { b * _heads, n, d });
    }

    private Tensor<T> Merge(Tensor<T> x, int b)
    {
        int n = x.Shape[1], d = x.Shape[2];
        return Engine.Reshape(Engine.TensorPermute(Engine.Reshape(x, new[] { b, _heads, n, d }), new[] { 0, 2, 1, 3 }).Contiguous(), new[] { b, n, _heads * d });
    }

    // softmax(q·kᵀ + bias + mask)·v for queries already scaled.
    private Tensor<T> Attend(Tensor<T> q, Tensor<T> k, Tensor<T> v, LazyWeight<T> table, Tensor<T>? mask, int b)
    {
        int n = _window * _window;
        var attn = Engine.BatchMatMul(q, Engine.TensorPermute(k, new[] { 0, 2, 1 }).Contiguous());
        var bias = Engine.TensorPermute(Engine.Reshape(Engine.TensorGather(table.Value, new Tensor<int>(_index, new[] { n * n }), axis: 0), new[] { n, n, _heads }), new[] { 2, 0, 1 }).Contiguous();
        var shaped = Engine.Reshape(attn, new[] { b, _heads, n, n });
        shaped = Engine.TensorAdd(shaped, Engine.TensorBroadcastTo(Engine.Reshape(bias, new[] { 1, _heads, n, n }), shaped.Shape.ToArray()));
        if (mask is not null)
        {
            int nW = mask.Shape[0];
            var windows = Engine.Reshape(shaped, new[] { b / nW, nW, _heads, n, n });
            windows = Engine.TensorAdd(windows, Engine.TensorBroadcastTo(Engine.Reshape(mask, new[] { 1, nW, 1, n, n }), windows.Shape.ToArray()));
            shaped = Engine.Reshape(windows, new[] { b, _heads, n, n });
        }
        var weights = Engine.Softmax(Engine.Reshape(shaped, new[] { b * _heads, n, n }), axis: -1);
        return Engine.BatchMatMul(weights, v);
    }

    /// <summary>Windows [B·nW, N, C] (and the coarse windows, the same shape) with masks [nW, N, N] or null.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? coarse, Tensor<T>? maskX, Tensor<T>? maskY)
    {
        int b = x.Shape[0], c = x.Shape[2], d = c / _heads;
        var qkv = _qkv.Forward(x);
        var q = Engine.TensorMultiplyScalar(Heads(Engine.TensorNarrow(qkv, 2, 0, c).Contiguous(), d), NumOps.FromDouble(1.0 / Math.Sqrt(d)));
        var k = Heads(Engine.TensorNarrow(qkv, 2, c, c).Contiguous(), d);
        var v = Heads(Engine.TensorNarrow(qkv, 2, 2 * c, c).Contiguous(), d);
        var y = Merge(Attend(q, k, v, _tableX, maskX, b), b);
        if (_cross && coarse is not null && _kv is not null && _merge1 is not null && _merge2 is not null && _tableY is not null)
        {
            var kv = _kv.Forward(coarse);
            var kc = Heads(Engine.TensorNarrow(kv, 2, 0, c).Contiguous(), d);
            var vc = Heads(Engine.TensorNarrow(kv, 2, c, c).Contiguous(), d);
            var yc = Merge(Attend(q, kc, vc, _tableY, maskY, b), b);
            y = Engine.TensorAdd(_merge2.Forward(VfiformerOps.LeakyRelu(Engine, _merge1.Forward(Engine.TensorConcatenate(new[] { y, yc }, 2)))), y);
        }
        return _proj.Forward(y);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _tableX;
        if (_tableY is not null) yield return _tableY;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>?[] { _qkv, _kv, _merge1, _merge2, _proj };
}

/// <summary>TFL: (shifted) window self- or cross-scale attention and a GELU MLP, each pre-normalized, residual and with
/// stochastic depth in training.</summary>
internal sealed class VfifTfl<T> : LazyWeightModule<T>
{
    private readonly int _window, _shift;
    private readonly bool _cross;
    private readonly double _dropPath;
    private readonly EmaLayerNorm<T> _norm1, _norm2;
    private readonly VfifAttention<T> _attention;
    private readonly EmaLinear<T> _fc1, _fc2;
    private readonly Dictionary<(int, int), (Tensor<T>? X, Tensor<T>? Y)> _masks = new();

    public VfifTfl(int dim, int heads, int window, int shift, double mlpRatio, bool cross, double dropPath)
    {
        _window = window;
        _shift = shift;
        _cross = cross;
        _dropPath = dropPath;
        _norm1 = new EmaLayerNorm<T>(dim, 1e-5);
        _attention = new VfifAttention<T>(dim, window, heads, cross);
        _norm2 = new EmaLayerNorm<T>(dim, 1e-5);
        _fc1 = new EmaLinear<T>(dim, (int)(dim * mlpRatio));
        _fc2 = new EmaLinear<T>((int)(dim * mlpRatio), dim);
    }

    private (Tensor<T>? X, Tensor<T>? Y) Masks(int h, int w)
    {
        if (_shift == 0) return (null, null);
        if (_masks.TryGetValue((h, w), out var cached)) return cached;
        int nW = h / _window * (w / _window), n = _window * _window;
        Tensor<T> ToTensor(double[] values)
        {
            var t = new Tensor<T>(new[] { nW, n, n });
            for (int i = 0; i < values.Length; i++) t[i] = NumOps.FromDouble(values[i]);
            return t;
        }
        var masks = (ToTensor(VfiformerOps.ShiftMask(h, w, _window, _shift)), _cross ? ToTensor(VfiformerOps.CrossMask(h, w, _window, _shift)) : (Tensor<T>?)null);
        _masks[(h, w)] = masks;
        return masks;
    }

    /// <summary>Tokens [B, H·W, C] of an H x W map.</summary>
    public Tensor<T> Forward(Tensor<T> x, int h, int w, Random? dropRandom)
    {
        int b = x.Shape[0], c = x.Shape[2];
        var normed = Engine.Reshape(_norm1.Forward(x), new[] { b, h, w, c });
        if (_shift > 0) normed = Engine.TensorRoll(normed, new[] { -_shift, -_shift }, new[] { 1, 2 });
        var windows = EmaOps.Partition(Engine, normed.Contiguous(), _window);
        var coarse = _cross ? VfiformerOps.CoarseWindows(Engine, normed, _window) : null;
        var (maskX, maskY) = Masks(h, w);
        var attended = EmaOps.Reverse(Engine, _attention.Forward(windows, coarse, maskX, maskY), _window, h, w);
        if (_shift > 0) attended = Engine.TensorRoll(attended, new[] { _shift, _shift }, new[] { 1, 2 });
        x = Engine.TensorAdd(x, VfiformerOps.DropPath(Engine, Engine.Reshape(attended.Contiguous(), new[] { b, h * w, c }), _dropPath, dropRandom));
        var mlp = _fc2.Forward(Engine.GELU(_fc1.Forward(_norm2.Forward(x))));
        return Engine.TensorAdd(x, VfiformerOps.DropPath(Engine, mlp, _dropPath, dropRandom));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _norm1, _attention, _norm2, _fc1, _fc2 };
}

/// <summary>RTFL: a group of TFLs (all cross-scale or none, as every release configuration lists them), a 3x3
/// convolution on the map and a residual over the group.</summary>
internal sealed class VfifRtfl<T> : LazyWeightModule<T>
{
    private readonly VfifTfl<T>[] _blocks;
    private readonly AmtConv2d<T> _conv;

    public VfifRtfl(int dim, int depth, int heads, int window, double mlpRatio, bool cross, IReadOnlyList<double> dropPaths)
    {
        _blocks = Enumerable.Range(0, depth).Select(i => new VfifTfl<T>(dim, heads, window, i % 2 == 0 ? 0 : window / 2, mlpRatio, cross, dropPaths[i])).ToArray();
        _conv = new AmtConv2d<T>(dim, dim, 3);
    }

    public Tensor<T> Forward(Tensor<T> x, int h, int w, Random? dropRandom)
    {
        int b = x.Shape[0], c = x.Shape[2];
        var y = x;
        foreach (var block in _blocks) y = block.Forward(y, h, w, dropRandom);
        var map = Engine.Reshape(Engine.TensorPermute(y, new[] { 0, 2, 1 }).Contiguous(), new[] { b, c, h, w });
        var conv = Engine.TensorPermute(Engine.Reshape(_conv.Forward(map), new[] { b, c, h * w }), new[] { 0, 2, 1 }).Contiguous();
        return Engine.TensorAdd(conv, x);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _blocks.Cast<LazyWeightModule<T>>().Concat(new LazyWeightModule<T>[] { _conv });
}

/// <summary>
/// TFModel: a four-scale encoder-decoder of RTFL groups (full, 1/2, 1/4 and 1/8 size), each later scale reached by a
/// strided convolution over the previous scale's features and both frames' warped context, decoded by transposed
/// convolutions with skips, ending in four channels (a residual and a blending mask).
/// </summary>
internal sealed class VfifTransformer<T> : LazyWeightModule<T>
{
    private readonly AmtConv2d<T> _first;
    private readonly EmaLayerNorm<T> _embedNorm, _norm;
    private readonly VfifRtfl<T>[][] _groups;
    private readonly AmtConv2d<T>[] _after;
    private readonly AmtConv2d<T>[] _afterSecond;
    private readonly AmtDeconv<T>[] _up;
    private readonly AmtConv2d<T> _last1a, _last1b, _last2a, _last2b;

    public VfifTransformer(int inChannels, int fuse, int dim, int window, int heads, double mlpRatio, IReadOnlyList<int> depths,
        IReadOnlyList<IReadOnlyList<bool>> cross, double dropPathRate)
    {
        _first = new AmtConv2d<T>(inChannels, dim, 3);
        _embedNorm = new EmaLayerNorm<T>(dim, 1e-5);
        _groups = new VfifRtfl<T>[4][];
        for (int g = 0; g < 4; g++)
        {
            // torch.linspace(0, rate, Σ depths) per scale; the release's last scale reads the third scale's rule.
            int rule = g == 3 ? 2 : g;
            int total = 2 * depths[rule];
            var rates = Enumerable.Range(0, total).Select(i => total > 1 ? dropPathRate * i / (total - 1) : 0.0).ToArray();
            int depth = depths[g];
            _groups[g] = Enumerable.Range(0, 2)
                .Select(r => new VfifRtfl<T>(dim, depth, heads, window, mlpRatio, cross[g][r], rates.Skip(r * depth).Take(depth).ToArray()))
                .ToArray();
        }
        _after = new[] { new AmtConv2d<T>(dim + 2 * fuse, dim, 3, 2), new AmtConv2d<T>(dim + 4 * fuse, dim, 3, 2), new AmtConv2d<T>(dim + 8 * fuse, dim, 3, 2) };
        _afterSecond = Enumerable.Range(0, 3).Select(_ => new AmtConv2d<T>(dim, dim, 3)).ToArray();
        _up = new[] { new AmtDeconv<T>(dim + 16 * fuse, dim), new AmtDeconv<T>(2 * dim, dim), new AmtDeconv<T>(2 * dim, dim) };
        _norm = new EmaLayerNorm<T>(dim, 1e-5);
        _last1a = new AmtConv2d<T>(2 * dim, dim, 3);
        _last1b = new AmtConv2d<T>(dim, dim, 3);
        _last2a = new AmtConv2d<T>(dim, dim / 2, 3);
        _last2b = new AmtConv2d<T>(dim / 2, 4, 3);
    }

    // forward_features: patch embedding (flatten and the shared LayerNorm), the group, the shared final LayerNorm.
    private Tensor<T> Features(Tensor<T> x, VfifRtfl<T>[] group, Random? dropRandom)
    {
        int b = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        var tokens = _embedNorm.Forward(Engine.TensorPermute(Engine.Reshape(x, new[] { b, c, h * w }), new[] { 0, 2, 1 }).Contiguous());
        foreach (var layer in group) tokens = layer.Forward(tokens, h, w, dropRandom);
        tokens = _norm.Forward(tokens);
        return Engine.Reshape(Engine.TensorPermute(tokens, new[] { 0, 2, 1 }).Contiguous(), new[] { b, c, h, w });
    }

    private Tensor<T> Cat(params Tensor<T>[] parts) => Engine.TensorConcatenate(parts, 1);

    private Tensor<T> Act(Tensor<T> x) => VfiformerOps.LeakyRelu(Engine, x);

    public Tensor<T> Forward(Tensor<T> x, Tensor<T>[] c0, Tensor<T>[] c1, Random? dropRandom)
    {
        var s0 = _first.Forward(x);
        var fea0 = Features(s0, _groups[0], dropRandom);
        var s1 = Act(_afterSecond[0].Forward(Act(_after[0].Forward(Cat(fea0, c0[0], c1[0])))));
        var fea1 = Features(s1, _groups[1], dropRandom);
        var s2 = Act(_afterSecond[1].Forward(Act(_after[1].Forward(Cat(fea1, c0[1], c1[1])))));
        var fea2 = Features(s2, _groups[2], dropRandom);
        var s3 = Act(_afterSecond[2].Forward(Act(_after[2].Forward(Cat(fea2, c0[2], c1[2])))));
        var fea3 = Features(s3, _groups[3], dropRandom);
        var up3 = Act(_up[0].Forward(Cat(fea3, c0[3], c1[3])));
        var up2 = Act(_up[1].Forward(Cat(up3, fea2)));
        var up1 = Act(_up[2].Forward(Cat(up2, fea1)));
        var output = Engine.TensorAdd(_last1b.Forward(Act(_last1a.Forward(Cat(up1, fea0)))), s0);
        return _last2b.Forward(Act(_last2a.Forward(output)));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _first, _embedNorm }.Concat(_groups.SelectMany(g => g)).Concat(_after).Concat(_afterSecond).Concat(_up)
            .Concat(new LazyWeightModule<T>[] { _norm, _last1a, _last1b, _last2a, _last2b });
}

/// <summary>
/// VFIformer: IFNet's flows refined by FlowRefineNet_Multis, both frames warped and fused, and the transformer's
/// sigmoid residual and blending mask: pred = clamp(warped0·mask + warped1·(1 − mask) + residual, 0, 1).
/// </summary>
internal sealed class VfiformerNetwork<T> : LazyWeightModule<T>
{
    private readonly VfifIfNet<T> _flownet;
    private readonly VfifRefine<T> _refine;
    private readonly AmtConv2d<T> _fuse1, _fuse2;
    private readonly VfifTransformer<T> _transformer;

    public VfiformerNetwork(IReadOnlyList<int> flowWidths, int c, int dim, int window, int heads, double mlpRatio, IReadOnlyList<int> depths,
        IReadOnlyList<IReadOnlyList<bool>> cross, double dropPathRate)
    {
        _flownet = new VfifIfNet<T>(flowWidths);
        _refine = new VfifRefine<T>(c);
        _fuse1 = new AmtConv2d<T>(12, 2 * c, 3);
        _fuse2 = new AmtConv2d<T>(2 * c, 2 * c, 3);
        _transformer = new VfifTransformer<T>(2 * c, c, dim, window, heads, mlpRatio, depths, cross, dropPathRate);
    }

    /// <summary>The midpoint frame and IFNet's flows from frames [N, 3, H, W] in [0, 1] with sides multiples of 8·window.</summary>
    public (Tensor<T> Prediction, Tensor<T>[] IfFlows, Tensor<T> Flow) Forward(Tensor<T> img0, Tensor<T> img1, Random? dropRandom)
    {
        var ifFlows = _flownet.Forward(img0, img1);
        var (flow, c0, c1) = _refine.Forward(img0, img1, ifFlows[2]);
        var warped0 = AmtOps.Warp(Engine, img0, Engine.TensorNarrow(flow, 1, 0, 2));
        var warped1 = AmtOps.Warp(Engine, img1, Engine.TensorNarrow(flow, 1, 2, 2));
        var x = VfiformerOps.LeakyRelu(Engine, _fuse2.Forward(VfiformerOps.LeakyRelu(Engine, _fuse1.Forward(Engine.TensorConcatenate(new[] { img0, img1, warped0, warped1 }, 1)))));
        var output = _transformer.Forward(x, c0, c1, dropRandom);
        var residual = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(Engine.Sigmoid(Engine.TensorNarrow(output, 1, 0, 3)), NumOps.FromDouble(2.0)), NumOps.FromDouble(-1.0));
        var mask = Engine.TensorBroadcastTo(Engine.Sigmoid(Engine.TensorNarrow(output, 1, 3, 1)), warped0.Shape.ToArray());
        var merged = Engine.TensorAdd(Engine.TensorMultiply(warped0, mask), Engine.TensorMultiply(warped1, Engine.ScalarMinusTensor(NumOps.One, mask)));
        return (Engine.TensorClamp(Engine.TensorAdd(merged, residual), NumOps.Zero, NumOps.One), ifFlows, flow);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _flownet, _refine, _fuse1, _fuse2, _transformer };
}
