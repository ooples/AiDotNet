using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>How a release convolution's weights start.</summary>
internal enum EmaConvInit
{
    /// <summary>torch's default (kaiming-uniform with a = √5, bias uniform in ±1/√fan_in).</summary>
    Torch,

    /// <summary>The release's _init_weights: N(0, 2/fan_out) weights and zero bias.</summary>
    FanOutNormal,
}

internal static class EmaInit
{
    // Box-Muller normal draws scaled by std.
    internal static LazyWeight<T> Normal<T>(int[] shape, double std) => new(shape, (tensor, random) =>
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < tensor.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            tensor[i] = ops.FromDouble(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
    });
}

/// <summary>nn.Conv2d with stride, padding and dilation.</summary>
internal sealed class EmaConv2d<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight, _bias;
    private readonly int _stride, _padding, _dilation;

    public EmaConv2d(int inChannels, int outChannels, int kernel, int stride = 1, int padding = -1, int dilation = 1,
        EmaConvInit init = EmaConvInit.Torch)
    {
        _stride = stride;
        _padding = padding < 0 ? kernel / 2 : padding;
        _dilation = dilation;
        var shape = new[] { outChannels, inChannels, kernel, kernel };
        int fanIn = inChannels * kernel * kernel;
        if (init == EmaConvInit.FanOutNormal)
        {
            _weight = EmaInit.Normal<T>(shape, Math.Sqrt(2.0 / (kernel * kernel * outChannels)));
            _bias = LazyWeight<T>.Constant(new[] { outChannels }, 0.0);
        }
        else
        {
            _weight = LazyWeight<T>.Uniform(shape, fanIn);
            _bias = LazyWeight<T>.Uniform(new[] { outChannels }, fanIn);
        }
    }

    public Tensor<T> Forward(Tensor<T> x)
        => MgldOps.ChannelBias(Engine, Engine.Conv2D(x, _weight.Value, new[] { _stride, _stride }, new[] { _padding, _padding },
            new[] { _dilation, _dilation }), _bias.Value);

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>A convolution followed by a per-channel PReLU (the release's conv()).</summary>
internal sealed class EmaConvPRelu<T> : LazyWeightModule<T>
{
    private readonly EmaConv2d<T> _conv;
    private readonly AmtPRelu<T> _prelu;

    public EmaConvPRelu(int inChannels, int outChannels, int kernel = 3, int stride = 1, EmaConvInit init = EmaConvInit.Torch)
    {
        _conv = new EmaConv2d<T>(inChannels, outChannels, kernel, stride, init: init);
        _prelu = new AmtPRelu<T>(outChannels);
    }

    public Tensor<T> Forward(Tensor<T> x) => _prelu.Forward(_conv.Forward(x));

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv, _prelu };
}

/// <summary>nn.Linear over the last axis; the attention blocks' trunc_normal(0.02) weights and zero bias, or torch's
/// default.</summary>
internal sealed class EmaLinear<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight, _bias;

    public EmaLinear(int inFeatures, int outFeatures, bool truncatedNormal = true)
    {
        var shape = new[] { outFeatures, inFeatures };
        // timm's trunc_normal_(std=.02) truncates at ±2 in absolute terms, a hundred standard deviations out.
        _weight = truncatedNormal ? EmaInit.Normal<T>(shape, 0.02) : LazyWeight<T>.Uniform(shape, inFeatures);
        _bias = truncatedNormal ? LazyWeight<T>.Constant(new[] { outFeatures }, 0.0) : LazyWeight<T>.Uniform(new[] { outFeatures }, inFeatures);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int inFeatures = shape[shape.Length - 1], outFeatures = _weight.Shape[0];
        int rows = x.Length / inFeatures;
        var y = Engine.TensorMatMul(Engine.Reshape(x, new[] { rows, inFeatures }), Engine.TensorPermute(_weight.Value, new[] { 1, 0 }));
        y = Engine.TensorAdd(y, Engine.TensorBroadcastTo(Engine.Reshape(_bias.Value, new[] { 1, outFeatures }), new[] { rows, outFeatures }));
        shape[shape.Length - 1] = outFeatures;
        return Engine.Reshape(y, shape);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>nn.LayerNorm over the last axis.</summary>
internal sealed class EmaLayerNorm<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight, _bias;
    private readonly double _epsilon;

    public EmaLayerNorm(int dim, double epsilon)
    {
        _epsilon = epsilon;
        _weight = LazyWeight<T>.Constant(new[] { dim }, 1.0);
        _bias = LazyWeight<T>.Constant(new[] { dim }, 0.0);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = x.Shape.ToArray();
        int last = shape.Length - 1;
        var centred = Engine.TensorSubtract(x, Engine.TensorBroadcastTo(Engine.ReduceMean(x, new[] { last }, true), shape));
        var variance = Engine.ReduceMean(Engine.TensorSquare(centred), new[] { last }, true);
        var normalized = Engine.TensorDivide(centred, Engine.TensorBroadcastTo(Engine.TensorSqrt(Engine.TensorAddScalar(variance, ops.FromDouble(_epsilon))), shape));
        var affineShape = Enumerable.Repeat(1, shape.Length).ToArray();
        affineShape[last] = shape[last];
        return Engine.TensorAdd(Engine.TensorMultiply(normalized, Engine.TensorBroadcastTo(Engine.Reshape(_weight.Value, affineShape), shape)),
            Engine.TensorBroadcastTo(Engine.Reshape(_bias.Value, affineShape), shape));
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>DWConv: a 3x3 depthwise convolution (N(0, 2/9) weights, zero bias) on tokens [B, H·W, C].</summary>
internal sealed class EmaDepthwiseConv<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight, _bias;

    public EmaDepthwiseConv(int dim)
    {
        _weight = EmaInit.Normal<T>(new[] { dim, 1, 3, 3 }, Math.Sqrt(2.0 / 9));
        _bias = LazyWeight<T>.Constant(new[] { dim }, 0.0);
    }

    public Tensor<T> Forward(Tensor<T> tokens, int h, int w)
    {
        int b = tokens.Shape[0], c = tokens.Shape[2];
        // Channels-last throughout: the per-channel taps broadcast over [B, H, W, C].
        var padded = EmaOps.PadSpatial(Engine, Engine.Reshape(tokens, new[] { b, h, w, c }), 1, 1, 1, 1);
        var shape = new[] { b, h, w, c };
        Tensor<T>? sum = null;
        for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++)
            {
                var tap = Engine.Reshape(Engine.TensorNarrow(Engine.TensorNarrow(_weight.Value, 2, ky, 1), 3, kx, 1), new[] { 1, 1, 1, c });
                var term = Engine.TensorMultiply(Engine.TensorSlice(padded, new[] { 0, ky, kx, 0 }, shape), Engine.TensorBroadcastTo(tap, shape));
                sum = sum is null ? term : Engine.TensorAdd(sum, term);
            }
        var y = Engine.TensorAdd(sum ?? throw new InvalidOperationException("The 3x3 kernel is never empty."),
            Engine.TensorBroadcastTo(Engine.Reshape(_bias.Value, new[] { 1, 1, 1, c }), shape));
        return Engine.Reshape(y, new[] { b, h * w, c });
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>The release's tensor operations (warplayer.py, feature_extractor.py's window helpers, loss.py).</summary>
internal static class EmaOps
{
    /// <summary>warplayer.warp: bilinear backward warp, border padding, align_corners=True.</summary>
    internal static Tensor<T> Warp<T>(IEngine engine, Tensor<T> x, Tensor<T> flow) => AmtOps.Warp(engine, x, flow);

    /// <summary>F.interpolate(mode='bilinear', align_corners=False) to an explicit size.</summary>
    internal static Tensor<T> Resize<T>(IEngine engine, Tensor<T> x, int height, int width) => AmtOps.Resize(engine, x, height, width);

    /// <summary>Zero padding of a channels-last map [B, H, W, C] on its spatial axes.</summary>
    internal static Tensor<T> PadSpatial<T>(IEngine engine, Tensor<T> x, int top, int bottom, int left, int right)
    {
        int b = x.Shape[0], h = x.Shape[1], c = x.Shape[3];
        if (top + bottom > 0)
        {
            var parts = new List<Tensor<T>>();
            if (top > 0) parts.Add(new Tensor<T>(new[] { b, top, x.Shape[2], c }));
            parts.Add(x);
            if (bottom > 0) parts.Add(new Tensor<T>(new[] { b, bottom, x.Shape[2], c }));
            x = engine.TensorConcatenate(parts.ToArray(), 1);
            h = x.Shape[1];
        }
        if (left + right > 0)
        {
            var parts = new List<Tensor<T>>();
            if (left > 0) parts.Add(new Tensor<T>(new[] { b, h, left, c }));
            parts.Add(x);
            if (right > 0) parts.Add(new Tensor<T>(new[] { b, h, right, c }));
            x = engine.TensorConcatenate(parts.ToArray(), 2);
        }
        return x;
    }

    /// <summary>window_partition: [B, H, W, C] to windows [B·nW, w·w, C], windows row-major within each image.</summary>
    internal static Tensor<T> Partition<T>(IEngine engine, Tensor<T> x, int window)
    {
        int b = x.Shape[0], h = x.Shape[1], w = x.Shape[2], c = x.Shape[3];
        var split = engine.Reshape(x, new[] { b, h / window, window, w / window, window, c });
        return engine.Reshape(engine.TensorPermute(split, new[] { 0, 1, 3, 2, 4, 5 }), new[] { b * (h / window) * (w / window), window * window, c });
    }

    /// <summary>window_reverse: windows [B·nW, w·w, C] back to [B, H, W, C].</summary>
    internal static Tensor<T> Reverse<T>(IEngine engine, Tensor<T> windows, int window, int h, int w)
    {
        int c = windows.Shape[2], b = windows.Shape[0] / ((h / window) * (w / window));
        var split = engine.Reshape(windows, new[] { b, h / window, w / window, window, window, c });
        return engine.Reshape(engine.TensorPermute(split, new[] { 0, 1, 3, 2, 4, 5 }), new[] { b, h, w, c });
    }

    // The region id of each padded pixel for three row and three column bands, as the release labels them.
    private static int[] Regions(int h, int w, int[] rowBounds, int[] colBounds)
    {
        var ids = new int[h * w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int row = y < rowBounds[0] ? 0 : y < rowBounds[1] ? 1 : 2;
                int col = x < colBounds[0] ? 0 : x < colBounds[1] ? 1 : 2;
                ids[y * w + x] = row * 3 + col;
            }
        return ids;
    }

    // [nW, N, N]: −100 between two pixels of a window that belong to different regions.
    private static double[] RegionMask(int[] ids, int h, int w, int window)
    {
        int nh = h / window, nw = w / window, n = window * window;
        var mask = new double[nh * nw * n * n];
        for (int wy = 0; wy < nh; wy++)
            for (int wx = 0; wx < nw; wx++)
            {
                int win = wy * nw + wx;
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                    {
                        int yi = wy * window + i / window, xi = wx * window + i % window;
                        int yj = wy * window + j / window, xj = wx * window + j % window;
                        if (ids[yi * w + xi] != ids[yj * w + xj]) mask[(win * n + i) * n + j] = -100.0;
                    }
            }
        return mask;
    }

    /// <summary>The centre padding of pad_if_needed: (left, right, top, bottom) bringing h and w to window multiples.</summary>
    internal static (int Left, int Right, int Top, int Bottom) WindowPadding(int h, int w, int window)
    {
        int padH = (h + window - 1) / window * window - h, padW = (w + window - 1) / window * window - w;
        return (padW / 2, padW - padW / 2, padH / 2, padH - padH / 2);
    }

    /// <summary>
    /// MotionFormerBlock's attention mask [nW, N, N] for a feature h x w: pad_if_needed's mask between padding bands,
    /// and for a shifted block the shifted-window mask with the padding mask's nonzeros also set to −100 (both in the
    /// release's coordinates); null when neither applies.
    /// </summary>
    internal static double[]? AttentionMask(int h, int w, int window, int shift)
    {
        var (left, right, top, bottom) = WindowPadding(h, w, window);
        int hp = h + top + bottom, wp = w + left + right;
        double[]? padMask = left + right + top + bottom > 0
            ? RegionMask(Regions(hp, wp, new[] { top, top + h }, new[] { left, left + w }), hp, wp, window)
            : null;
        if (shift == 0) return padMask;
        var shiftMask = RegionMask(Regions(hp, wp, new[] { hp - window, hp - shift }, new[] { wp - window, wp - shift }), hp, wp, window);
        if (padMask is not null)
            for (int i = 0; i < shiftMask.Length; i++)
                if (padMask[i] != 0) shiftMask[i] = -100.0;
        return shiftMask;
    }

    /// <summary>MotionFormer.get_cor: [B, H, W, 2] holding each position's (x, y) on linspace(−1, 1).</summary>
    internal static Tensor<T> CoordinateGrid<T>(int batch, int h, int w)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var grid = new Tensor<T>(new[] { batch, h, w, 2 });
        double Step(int i, int n) => n > 1 ? -1.0 + 2.0 * i / (n - 1) : -1.0;
        for (int b = 0; b < batch; b++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int o = ((b * h + y) * w + x) * 2;
                    grid[o] = ops.FromDouble(Step(x, w));
                    grid[o + 1] = ops.FromDouble(Step(y, h));
                }
        return grid;
    }

    // conv_gauss on [M, 1, H, W]: reflect padding by 2, the 5x5 binomial kernel / 256 (times scale).
    private static Tensor<T> Gauss<T>(IEngine engine, Tensor<T> x, double scale)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var taps = new[] { 1.0, 4, 6, 4, 1 };
        var kernel = new Tensor<T>(new[] { 1, 1, 5, 5 });
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++) kernel[i * 5 + j] = ops.FromDouble(scale * taps[i] * taps[j] / 256.0);
        return engine.Conv2D(engine.PadNd(x, new[] { 2, 2, 2, 2 }, PadMode.Reflect, ops.Zero), kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
    }

    // loss.upsample: zeros between pixels, then the binomial kernel times 4.
    private static Tensor<T> Upsample<T>(IEngine engine, Tensor<T> x)
    {
        int m = x.Shape[0], h = x.Shape[2], w = x.Shape[3];
        var six = engine.Reshape(x, new[] { m, 1, h, 1, w, 1 });
        var cols = engine.TensorConcatenate(new[] { six, new Tensor<T>(six._shape) }, 5);
        var rows = engine.TensorConcatenate(new[] { cols, new Tensor<T>(cols._shape) }, 3);
        return Gauss(engine, engine.Reshape(rows, new[] { m, 1, 2 * h, 2 * w }), 4.0);
    }

    // laplacian_pyramid on [M, 1, H, W]: each level's detail current − upsample(downsample(gauss(current))).
    internal static List<Tensor<T>> LaplacianPyramid<T>(IEngine engine, Tensor<T> image, int levels)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var pick = new Tensor<T>(new[] { 1, 1, 1, 1 });
        pick[0] = ops.One;
        var pyramid = new List<Tensor<T>>();
        var current = image;
        for (int level = 0; level < levels; level++)
        {
            var down = engine.Conv2D(Gauss(engine, current, 1.0), pick, new[] { 2, 2 }, new[] { 0, 0 }, new[] { 1, 1 });
            pyramid.Add(engine.TensorSubtract(current, Upsample(engine, down)));
            current = down;
        }
        return pyramid;
    }

    /// <summary>LapLoss: the sum over the Laplacian pyramid's levels of the mean absolute difference.</summary>
    internal static Tensor<T> LaplacianLoss<T>(IEngine engine, Tensor<T> prediction, Tensor<T> target, int levels)
    {
        int n = prediction.Shape[0], c = prediction.Shape[1], h = prediction.Shape[2], w = prediction.Shape[3];
        var a = LaplacianPyramid(engine, engine.Reshape(prediction, new[] { n * c, 1, h, w }), levels);
        var b = LaplacianPyramid(engine, engine.Reshape(target, new[] { n * c, 1, h, w }), levels);
        Tensor<T>? total = null;
        for (int i = 0; i < levels; i++)
        {
            var diff = engine.TensorAbs(engine.TensorSubtract(a[i], b[i]));
            var mean = engine.ReduceMean(diff, new[] { 0, 1, 2, 3 }, false);
            total = total is null ? mean : engine.TensorAdd(total, mean);
        }
        return total ?? throw new InvalidOperationException("The Laplacian loss needs a level.");
    }
}
