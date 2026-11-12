using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>The release's operators (run.py, softsplat.py, correlation/correlation.py).</summary>
internal static class SoftSplatOps
{
    /// <summary>run.backwarp: bilinear grid_sample with zero padding and align_corners=True.</summary>
    internal static Tensor<T> Backwarp<T>(IEngine engine, Tensor<T> x, Tensor<T> flow)
        => BasicVsrBlocks.FlowWarp(engine, x, flow, GridSamplePadding.Zeros);

    /// <summary>
    /// correlation.FunctionCorrelation: for each of the 9x9 displacements (row-major, vertical outer) of the zero-padded
    /// second map, the channel mean of the product with the first.
    /// </summary>
    internal static Tensor<T> Correlation<T>(IEngine engine, Tensor<T> one, Tensor<T> two)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = one.Shape[0], c = one.Shape[1], h = one.Shape[2], w = one.Shape[3];
        var padded = engine.PadNd(two, new[] { 4, 4, 4, 4 }, PadMode.Constant, ops.Zero);
        var planes = new Tensor<T>[81];
        for (int dy = 0; dy < 9; dy++)
            for (int dx = 0; dx < 9; dx++)
            {
                var shifted = engine.TensorSlice(padded, new[] { 0, 0, dy, dx }, new[] { n, c, h, w });
                planes[dy * 9 + dx] = engine.ReduceMean(engine.TensorMultiply(one, shifted), new[] { 1 }, true);
            }
        return engine.TensorConcatenate(planes, 1);
    }

    /// <summary>
    /// softsplat_func: each source pixel's value added to the four pixels around its forward-displaced position with
    /// bilinear weights. The weights carry the gradient to the flow, as the release's softsplat_flow kernel does.
    /// </summary>
    internal static Tensor<T> Splat<T>(IEngine engine, Tensor<T> x, Tensor<T> flow)
        => SplatCorners(engine, x, flow, (fracX, fracY) =>
        {
            var ops = MathHelper.GetNumericOperations<T>();
            var restX = engine.ScalarMinusTensor(ops.One, fracX);
            var restY = engine.ScalarMinusTensor(ops.One, fracY);
            return new[]
            {
                engine.TensorMultiply(restX, restY), engine.TensorMultiply(fracX, restY),
                engine.TensorMultiply(restX, fracY), engine.TensorMultiply(fracX, fracY),
            };
        });

    /// <summary>
    /// Adds each source pixel's value, times a weight per corner, to the four pixels around its forward-displaced
    /// position (northwest, northeast, southwest, southeast); the weights are built from the fractional offsets
    /// [N, 1, H, W] so they carry the gradient to the flow. Positions outside the image, and non-finite flows, are dropped.
    /// </summary>
    internal static Tensor<T> SplatCorners<T>(IEngine engine, Tensor<T> x, Tensor<T> flow, Func<Tensor<T>, Tensor<T>, Tensor<T>[]> cornerWeights)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3], plane = h * w, total = n * plane;
        var flowValues = flow.ToArray();
        var floorX = new Tensor<T>(new[] { n, 1, h, w });
        var floorY = new Tensor<T>(new[] { n, 1, h, w });
        var baseX = new Tensor<T>(new[] { n, 1, h, w });
        var baseY = new Tensor<T>(new[] { n, 1, h, w });
        var indices = new Tensor<int>(new[] { 4 * total });
        for (int b = 0; b < n; b++)
            for (int y = 0; y < h; y++)
                for (int xx = 0; xx < w; xx++)
                {
                    int p = (b * h + y) * w + xx;
                    double fx = xx + ops.ToDouble(flowValues[(b * 2 * h + y) * w + xx]);
                    double fy = y + ops.ToDouble(flowValues[((b * 2 + 1) * h + y) * w + xx]);
                    baseX[p] = ops.FromDouble(xx);
                    baseY[p] = ops.FromDouble(y);
                    bool finite = !double.IsNaN(fx) && !double.IsInfinity(fx) && !double.IsNaN(fy) && !double.IsInfinity(fy);
                    int nx = finite ? (int)Math.Floor(fx) : 0, ny = finite ? (int)Math.Floor(fy) : 0;
                    floorX[p] = ops.FromDouble(nx);
                    floorY[p] = ops.FromDouble(ny);
                    for (int corner = 0; corner < 4; corner++)
                    {
                        int tx = nx + (corner & 1), ty = ny + (corner >> 1);
                        bool landed = finite && tx >= 0 && tx < w && ty >= 0 && ty < h;
                        indices[corner * total + p] = landed ? b * plane + ty * w + tx : total;
                    }
                }
        var fracX = engine.TensorSubtract(engine.TensorAdd(baseX, engine.TensorNarrow(flow, 1, 0, 1)), floorX);
        var fracY = engine.TensorSubtract(engine.TensorAdd(baseY, engine.TensorNarrow(flow, 1, 1, 1)), floorY);
        // Northwest, northeast, southwest, southeast, matching the corner order of the indices.
        var weights = cornerWeights(fracX, fracY);
        var shape = x.Shape.ToArray();
        var updates = weights.Select(weight =>
        {
            var weighted = engine.TensorMultiply(x, engine.TensorBroadcastTo(weight, shape));
            return engine.Reshape(engine.TensorPermute(weighted, new[] { 1, 0, 2, 3 }).Contiguous(), new[] { c, total });
        }).ToArray();
        // One extra column collects everything that lands outside, and is dropped.
        var sums = engine.TensorScatterAdd(new Tensor<T>(new[] { c, total + 1 }), indices, engine.TensorConcatenate(updates, 1), 1);
        var inside = engine.Reshape(engine.TensorNarrow(sums, 1, 0, total).Contiguous(), new[] { c, n, h, w });
        return engine.TensorPermute(inside, new[] { 1, 0, 2, 3 }).Contiguous();
    }

    /// <summary>softsplat(strMode='soft'): splat x·exp(Z) and exp(Z), and divide by the latter plus 1e-7.</summary>
    internal static Tensor<T> SoftmaxSplat<T>(IEngine engine, Tensor<T> x, Tensor<T> flow, Tensor<T> metric)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var weight = engine.TensorExp(metric);
        int c = x.Shape[1];
        var splatted = Splat(engine, engine.TensorConcatenate(new[] { engine.TensorMultiply(x, engine.TensorBroadcastTo(weight, x.Shape.ToArray())), weight }, 1), flow);
        var normalize = engine.TensorAddScalar(engine.TensorNarrow(splatted, 1, c, 1), ops.FromDouble(1e-7));
        var values = engine.TensorNarrow(splatted, 1, 0, c);
        return engine.TensorDivide(values, engine.TensorBroadcastTo(normalize, values.Shape.ToArray()));
    }

    /// <summary>F.interpolate(mode='bilinear', align_corners=False) of a flow, scaled by the width ratio as run.py does.</summary>
    internal static Tensor<T> ResizeFlow<T>(IEngine engine, Tensor<T> flow, int height, int width)
        => engine.TensorMultiplyScalar(AmtOps.Resize(engine, flow, height, width),
            MathHelper.GetNumericOperations<T>().FromDouble(width / (double)flow.Shape[3]));

    /// <summary>The release's pad of −1 on the bottom or right when an upsampled map is one larger than its row.</summary>
    internal static Tensor<T> CropTo<T>(IEngine engine, Tensor<T> x, Tensor<T> like)
        => x.Shape[2] == like.Shape[2] && x.Shape[3] == like.Shape[3] ? x
            : engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { x.Shape[0], x.Shape[1], like.Shape[2], like.Shape[3] });
}

/// <summary>nn.Conv2d with torch's default initialization and an optional bias.</summary>
internal sealed class SoftConv2d<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T>? _bias;
    private readonly int _stride, _padding;

    public SoftConv2d(int inChannels, int outChannels, int kernel = 3, int stride = 1, bool bias = false)
    {
        _stride = stride;
        _padding = kernel / 2;
        int fanIn = inChannels * kernel * kernel;
        _weight = LazyWeight<T>.Uniform(new[] { outChannels, inChannels, kernel, kernel }, fanIn);
        if (bias) _bias = LazyWeight<T>.Uniform(new[] { outChannels }, fanIn);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = Engine.Conv2D(x, _weight.Value, new[] { _stride, _stride }, new[] { _padding, _padding }, new[] { 1, 1 });
        return _bias is null ? y : MgldOps.ChannelBias(Engine, y, _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        if (_bias is not null) yield return _bias;
    }
}

/// <summary>The Flow Extractor: six levels of a biased stride-2 and a biased 3x3 convolution, each LeakyReLU(0.1).</summary>
internal sealed class SoftFlowExtractor<T> : LazyWeightModule<T>
{
    private readonly SoftConv2d<T>[] _convs;

    public SoftFlowExtractor(IReadOnlyList<int> channels)
    {
        _convs = new SoftConv2d<T>[12];
        int inChannels = 3;
        for (int i = 0; i < 6; i++)
        {
            _convs[2 * i] = new SoftConv2d<T>(inChannels, channels[i], 3, 2, true);
            _convs[2 * i + 1] = new SoftConv2d<T>(channels[i], channels[i], 3, 1, true);
            inChannels = channels[i];
        }
    }

    public Tensor<T>[] Forward(Tensor<T> x)
    {
        var slope = NumOps.FromDouble(0.1);
        var levels = new Tensor<T>[6];
        for (int i = 0; i < 6; i++)
            levels[i] = x = Engine.LeakyReLU(_convs[2 * i + 1].Forward(Engine.LeakyReLU(_convs[2 * i].Forward(x), slope)), slope);
        return levels;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _convs;
}

/// <summary>A Flow Decoder: the correlation with the (warped) second features, then six biased convolutions to a flow.</summary>
internal sealed class SoftFlowDecoder<T> : LazyWeightModule<T>
{
    private readonly SoftConv2d<T>[] _convs;

    public SoftFlowDecoder(int inChannels, IReadOnlyList<int> hidden)
    {
        _convs = new SoftConv2d<T>[6];
        for (int i = 0; i < 6; i++)
        {
            int outChannels = i < 5 ? hidden[i] : 2;
            _convs[i] = new SoftConv2d<T>(inChannels, outChannels, 3, 1, true);
            inChannels = outChannels;
        }
    }

    public Tensor<T> Forward(Tensor<T> one, Tensor<T> two, Tensor<T>? previous)
    {
        Tensor<T> main;
        if (previous is null)
        {
            main = Engine.TensorConcatenate(new[] { one, SoftSplatOps.Correlation(Engine, one, two) }, 1);
        }
        else
        {
            var flow = SoftSplatOps.ResizeFlow(Engine, previous, one.Shape[2], one.Shape[3]);
            var volume = SoftSplatOps.Correlation(Engine, one, SoftSplatOps.Backwarp(Engine, two, flow));
            main = Engine.TensorConcatenate(new[] { one, volume, flow }, 1);
        }
        var slope = NumOps.FromDouble(0.1);
        for (int i = 0; i < 6; i++)
        {
            main = _convs[i].Forward(main);
            if (i < 5) main = Engine.LeakyReLU(main, slope);
        }
        return main;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _convs;
}

/// <summary>run.Flow: PWC-Net's extractor and six decoders, coarse to fine, in both directions.</summary>
internal sealed class SoftFlowNetwork<T> : LazyWeightModule<T>
{
    private readonly SoftFlowExtractor<T> _extractor;
    private readonly SoftFlowDecoder<T>[] _decoders;

    public SoftFlowNetwork(IReadOnlyList<int> channels, IReadOnlyList<int> hidden)
    {
        _extractor = new SoftFlowExtractor<T>(channels);
        // Finest first, as run.py registers netFirst to netSixth; the coarsest reads no previous flow.
        _decoders = Enumerable.Range(0, 6).Select(i => new SoftFlowDecoder<T>(channels[i] + 81 + (i < 5 ? 2 : 0), hidden)).ToArray();
    }

    public (Tensor<T> Forward, Tensor<T> Backward) Forward(Tensor<T> one, Tensor<T> two)
    {
        int h = one.Shape[2], w = one.Shape[3];
        var a = _extractor.Forward(one);
        var b = _extractor.Forward(two);
        Tensor<T>? forward = null, backward = null;
        for (int level = 5; level >= 0; level--)
        {
            forward = _decoders[level].Forward(a[level], b[level], forward);
            backward = _decoders[level].Forward(b[level], a[level], backward);
        }
        if (forward is null || backward is null) throw new InvalidOperationException("The flow estimator has no level.");
        return (SoftSplatOps.ResizeFlow(Engine, forward, h, w), SoftSplatOps.ResizeFlow(Engine, backward, h, w));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _extractor }.Concat(_decoders);
}

/// <summary>Synthesis.Basic's two layouts.</summary>
internal enum SoftBasicKind
{
    /// <summary>'relu-conv-relu-conv'.</summary>
    ReluConvReluConv,

    /// <summary>'conv-relu-conv'.</summary>
    ConvReluConv,
}

/// <summary>Synthesis.Basic with its skip: identity when the widths match, else a 1x1 projection.</summary>
internal sealed class SoftBasic<T> : LazyWeightModule<T>
{
    private readonly AmtPRelu<T>? _first;
    private readonly SoftConv2d<T> _conv1, _conv2;
    private readonly AmtPRelu<T> _second;
    private readonly SoftConv2d<T>? _shortcut;

    public SoftBasic(SoftBasicKind kind, int inChannels, int mid, int outChannels)
    {
        if (kind == SoftBasicKind.ReluConvReluConv) _first = new AmtPRelu<T>(inChannels);
        _conv1 = new SoftConv2d<T>(inChannels, mid);
        _second = new AmtPRelu<T>(mid);
        _conv2 = new SoftConv2d<T>(mid, outChannels);
        if (inChannels != outChannels) _shortcut = new SoftConv2d<T>(inChannels, outChannels, 1);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = _conv2.Forward(_second.Forward(_conv1.Forward(_first is null ? x : _first.Forward(x))));
        return Engine.TensorAdd(y, _shortcut is null ? x : _shortcut.Forward(x));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>?[] { _first, _conv1, _second, _conv2, _shortcut };
}

/// <summary>Synthesis.Downsample and Upsample: (2x bilinear,) PReLU, convolution (stride 2), PReLU, convolution.</summary>
internal sealed class SoftResample<T> : LazyWeightModule<T>
{
    private readonly bool _up;
    private readonly AmtPRelu<T> _first, _second;
    private readonly SoftConv2d<T> _conv1, _conv2;

    public SoftResample(bool up, int inChannels, int mid, int outChannels)
    {
        _up = up;
        _first = new AmtPRelu<T>(inChannels);
        _conv1 = new SoftConv2d<T>(inChannels, mid, 3, up ? 1 : 2);
        _second = new AmtPRelu<T>(mid);
        _conv2 = new SoftConv2d<T>(mid, outChannels);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        if (_up) x = AmtOps.Resize(Engine, x, x.Shape[2] * 2, x.Shape[3] * 2);
        return _conv2.Forward(_second.Forward(_conv1.Forward(_first.Forward(x))));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _first, _conv1, _second, _conv2 };
}

/// <summary>Synthesis.Encode: three levels of convolution-PReLU pairs (the second and third strided); the first level
/// is returned concatenated with the input frame.</summary>
internal sealed class SoftEncode<T> : LazyWeightModule<T>
{
    private readonly SoftConv2d<T>[] _convs;
    private readonly AmtPRelu<T>[] _prelus;

    public SoftEncode(IReadOnlyList<int> channels)
    {
        _convs = new SoftConv2d<T>[6];
        _prelus = new AmtPRelu<T>[6];
        int inChannels = 3;
        for (int i = 0; i < 3; i++)
        {
            _convs[2 * i] = new SoftConv2d<T>(inChannels, channels[i], 3, i == 0 ? 1 : 2);
            _prelus[2 * i] = new AmtPRelu<T>(channels[i]);
            _convs[2 * i + 1] = new SoftConv2d<T>(channels[i], channels[i]);
            _prelus[2 * i + 1] = new AmtPRelu<T>(channels[i]);
            inChannels = channels[i];
        }
    }

    public Tensor<T>[] Forward(Tensor<T> image)
    {
        var levels = new Tensor<T>[3];
        var x = image;
        for (int i = 0; i < 3; i++)
            levels[i] = x = _prelus[2 * i + 1].Forward(_convs[2 * i + 1].Forward(_prelus[2 * i].Forward(_convs[2 * i].Forward(x))));
        levels[0] = Engine.TensorConcatenate(new[] { image, levels[0] }, 1);
        return levels;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => Enumerable.Range(0, 6).SelectMany(i => new LazyWeightModule<T>[] { _convs[i], _prelus[i] });
}

/// <summary>
/// Synthesis.Softmetric: the importance metric from the frame and the brightness-constancy error (the channel mean of
/// the absolute difference to the backward-warped other level-0 features), refined by a four-row U-Net to one channel.
/// </summary>
internal sealed class SoftMetric<T> : LazyWeightModule<T>
{
    private readonly SoftConv2d<T> _input, _error;
    private readonly SoftBasic<T>[] _rows;
    private readonly SoftResample<T>[] _down, _up;
    private readonly SoftBasic<T> _output;

    public SoftMetric(IReadOnlyList<int> m)
    {
        _input = new SoftConv2d<T>(3, m[0] - 4);
        _error = new SoftConv2d<T>(1, 4);
        _rows = Enumerable.Range(0, 4).Select(r => new SoftBasic<T>(SoftBasicKind.ReluConvReluConv, m[r], m[r], m[r])).ToArray();
        _down = Enumerable.Range(0, 3).Select(r => new SoftResample<T>(false, m[r], m[r + 1], m[r + 1])).ToArray();
        _up = Enumerable.Range(0, 3).Select(r => new SoftResample<T>(true, m[3 - r], m[2 - r], m[2 - r])).ToArray();
        _output = new SoftBasic<T>(SoftBasicKind.ConvReluConv, m[0], m[0], 1);
    }

    public Tensor<T> Forward(Tensor<T>[] encoded, Tensor<T>[] other, Tensor<T> flow)
    {
        var error = Engine.ReduceMean(Engine.TensorAbs(Engine.TensorSubtract(encoded[0], SoftSplatOps.Backwarp(Engine, other[0], flow))), new[] { 1 }, true);
        var column = new Tensor<T>[4];
        column[0] = Engine.TensorConcatenate(new[] { _input.Forward(Engine.TensorNarrow(encoded[0], 1, 0, 3)), _error.Forward(error) }, 1);
        for (int r = 1; r < 4; r++) column[r] = _down[r - 1].Forward(column[r - 1]);
        for (int r = 3; r >= 0; r--)
        {
            column[r] = _rows[r].Forward(column[r]);
            if (r != 3) column[r] = Engine.TensorAdd(column[r], SoftSplatOps.CropTo(Engine, _up[2 - r].Forward(column[r + 1]), column[r]));
        }
        return _output.Forward(column[0]);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _input, _error }.Concat(_rows).Concat(_down).Concat(_up).Concat(new[] { _output });
}

/// <summary>
/// Synthesis.Warp: at each pyramid level, both frames' features with their metric softmax-splatted to time t (Z the
/// negated metric clipped to ±20), concatenated and fused by a conv-relu-conv block.
/// </summary>
internal sealed class SoftWarp<T> : LazyWeightModule<T>
{
    private readonly SoftBasic<T>[] _fuse;

    public SoftWarp(IReadOnlyList<int> c)
        => _fuse = Enumerable.Range(0, 3).Select(l =>
        {
            int levelChannels = (l == 0 ? 3 : 0) + c[l] + 1;
            return new SoftBasic<T>(SoftBasicKind.ConvReluConv, 2 * levelChannels, c[l], c[l]);
        }).ToArray();

    public Tensor<T>[] Forward(Tensor<T>[] encOne, Tensor<T>[] encTwo, Tensor<T> metricOne, Tensor<T> metricTwo, Tensor<T> forward, Tensor<T> backward)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>[3];
        Tensor<T> Importance(Tensor<T> metric) => Engine.TensorClamp(Engine.TensorNegate(metric), ops.FromDouble(-20.0), ops.FromDouble(20.0));
        for (int level = 0; level < 3; level++)
        {
            if (level != 0)
            {
                int h = encOne[level].Shape[2], w = encOne[level].Shape[3];
                metricOne = AmtOps.Resize(Engine, metricOne, h, w);
                metricTwo = AmtOps.Resize(Engine, metricTwo, h, w);
                forward = SoftSplatOps.ResizeFlow(Engine, forward, h, w);
                backward = SoftSplatOps.ResizeFlow(Engine, backward, h, w);
            }
            output[level] = _fuse[level].Forward(Engine.TensorConcatenate(new[]
            {
                SoftSplatOps.SoftmaxSplat(Engine, Engine.TensorConcatenate(new[] { encOne[level], metricOne }, 1), forward, Importance(metricOne)),
                SoftSplatOps.SoftmaxSplat(Engine, Engine.TensorConcatenate(new[] { encTwo[level], metricTwo }, 1), backward, Importance(metricTwo)),
            }, 1));
        }
        return output;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _fuse;
}

/// <summary>
/// run.Synthesis: the feature pyramids, the importance metrics scaled by 2t and 2(1 − t), the flows scaled by t and
/// 1 − t, the splatted pyramids, and a three-row, six-column GridNet to the frame.
/// </summary>
internal sealed class SoftSynthesis<T> : LazyWeightModule<T>
{
    private readonly SoftEncode<T> _encode;
    private readonly SoftMetric<T> _metric;
    private readonly SoftWarp<T> _warp;
    private readonly SoftBasic<T>[,] _rows;
    private readonly SoftResample<T>[,] _down, _up;
    private readonly SoftBasic<T> _output;

    public SoftSynthesis(IReadOnlyList<int> c, IReadOnlyList<int> m)
    {
        _encode = new SoftEncode<T>(c);
        _metric = new SoftMetric<T>(m);
        _warp = new SoftWarp<T>(c);
        _rows = new SoftBasic<T>[3, 5];
        for (int r = 0; r < 3; r++)
            for (int k = 0; k < 5; k++) _rows[r, k] = new SoftBasic<T>(SoftBasicKind.ReluConvReluConv, c[r], c[r], c[r]);
        // _down[col, r] maps row r to r + 1 in columns 0-2; _up[col - 3, r] maps row r + 1 to r in columns 3-5.
        _down = new SoftResample<T>[3, 2];
        _up = new SoftResample<T>[3, 2];
        for (int col = 0; col < 3; col++)
            for (int r = 0; r < 2; r++)
            {
                _down[col, r] = new SoftResample<T>(false, c[r], c[r + 1], c[r + 1]);
                _up[col, r] = new SoftResample<T>(true, c[r + 1], c[r], c[r]);
            }
        _output = new SoftBasic<T>(SoftBasicKind.ConvReluConv, c[0], c[0], 3);
    }

    public Tensor<T> Forward(Tensor<T> one, Tensor<T> two, Tensor<T> forward, Tensor<T> backward, double t)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var encOne = _encode.Forward(one);
        var encTwo = _encode.Forward(two);
        var metricOne = Engine.TensorMultiplyScalar(_metric.Forward(encOne, encTwo, forward), ops.FromDouble(2.0 * t));
        var metricTwo = Engine.TensorMultiplyScalar(_metric.Forward(encTwo, encOne, backward), ops.FromDouble(2.0 * (1.0 - t)));
        var warped = _warp.Forward(encOne, encTwo, metricOne, metricTwo,
            Engine.TensorMultiplyScalar(forward, ops.FromDouble(t)), Engine.TensorMultiplyScalar(backward, ops.FromDouble(1.0 - t)));
        var column = new Tensor<T>[3];
        column[0] = warped[0];
        column[1] = Engine.TensorAdd(warped[1], _down[0, 0].Forward(column[0]));
        column[2] = Engine.TensorAdd(warped[2], _down[0, 1].Forward(column[1]));
        for (int col = 1; col < 3; col++)
            for (int r = 0; r < 3; r++)
            {
                column[r] = _rows[r, col - 1].Forward(column[r]);
                if (r != 0) column[r] = Engine.TensorAdd(column[r], _down[col, r - 1].Forward(column[r - 1]));
            }
        for (int col = 3; col < 6; col++)
            for (int r = 2; r >= 0; r--)
            {
                column[r] = _rows[r, col - 1].Forward(column[r]);
                if (r != 2) column[r] = Engine.TensorAdd(column[r], SoftSplatOps.CropTo(Engine, _up[col - 3, r].Forward(column[r + 1]), column[r]));
            }
        return _output.Forward(column[0]);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
    {
        yield return _encode;
        yield return _metric;
        yield return _warp;
        foreach (var block in _rows) yield return block;
        foreach (var block in _down) yield return block;
        foreach (var block in _up) yield return block;
        yield return _output;
    }
}

/// <summary>
/// run.Network: both frames standardized by their joint mean and standard deviation (statistics outside the graph), the
/// bidirectional flows, the synthesis at time t, and the statistics restored.
/// </summary>
internal sealed class SoftSplatNetwork<T> : LazyWeightModule<T>
{
    private readonly SoftFlowNetwork<T> _flow;
    private readonly SoftSynthesis<T> _synthesis;

    public SoftSplatNetwork(IReadOnlyList<int> flowChannels, IReadOnlyList<int> flowHidden, IReadOnlyList<int> metric, IReadOnlyList<int> synthesis)
    {
        _flow = new SoftFlowNetwork<T>(flowChannels, flowHidden);
        _synthesis = new SoftSynthesis<T>(synthesis, metric);
    }

    /// <summary>The joint per-sample mean and (population) standard deviation of both frames [N, 3, H, W].</summary>
    internal static (Tensor<T> Mean, Tensor<T> Std) Statistics(Tensor<T> one, Tensor<T> two)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = one.Shape[0], per = one.Length / n;
        var a = one.ToArray();
        var b = two.ToArray();
        var mean = new Tensor<T>(new[] { n, 1, 1, 1 });
        var std = new Tensor<T>(new[] { n, 1, 1, 1 });
        for (int s = 0; s < n; s++)
        {
            double meanA = 0, meanB = 0;
            for (int i = 0; i < per; i++) { meanA += ops.ToDouble(a[s * per + i]); meanB += ops.ToDouble(b[s * per + i]); }
            meanA /= per;
            meanB /= per;
            double joint = (meanA + meanB) / 2, varA = 0, varB = 0;
            for (int i = 0; i < per; i++)
            {
                double da = ops.ToDouble(a[s * per + i]) - meanA, db = ops.ToDouble(b[s * per + i]) - meanB;
                varA += da * da;
                varB += db * db;
            }
            varA /= per;
            varB /= per;
            mean[s] = ops.FromDouble(joint);
            std[s] = ops.FromDouble(Math.Sqrt((varA + (joint - meanA) * (joint - meanA) + varB + (joint - meanB) * (joint - meanB)) / 2));
        }
        return (mean, std);
    }

    // Both frames minus the joint mean over the standard deviation plus 1e-7.
    private (Tensor<T> One, Tensor<T> Two, Tensor<T> Mean, Tensor<T> Std) Standardize(Tensor<T> one, Tensor<T> two)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = one.Shape.ToArray();
        var (mean, std) = Statistics(one, two);
        var meanFull = Engine.TensorBroadcastTo(mean, shape);
        var scale = Engine.TensorBroadcastTo(Engine.TensorAddScalar(std, ops.FromDouble(1e-7)), shape);
        return (Engine.TensorDivide(Engine.TensorSubtract(one, meanFull), scale), Engine.TensorDivide(Engine.TensorSubtract(two, meanFull), scale),
            meanFull, Engine.TensorBroadcastTo(std, shape));
    }

    /// <summary>The forward and backward flows between the standardized frames [N, 3, H, W].</summary>
    public (Tensor<T> Forward, Tensor<T> Backward) Flows(Tensor<T> one, Tensor<T> two)
    {
        var (a, b, _, _) = Standardize(one, two);
        return _flow.Forward(a, b);
    }

    /// <summary>The frame at each time in <paramref name="times"/> from frames [N, 3, H, W] in [0, 1] with even sides.</summary>
    public Tensor<T>[] Forward(Tensor<T> one, Tensor<T> two, IReadOnlyList<double> times)
    {
        var (a, b, mean, std) = Standardize(one, two);
        var (forward, backward) = _flow.Forward(a, b);
        return times.Select(t => Engine.TensorAdd(Engine.TensorMultiply(_synthesis.Forward(a, b, forward, backward, t), std), mean)).ToArray();
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _flow, _synthesis };
}
