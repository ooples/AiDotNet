using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>nn.Conv2d with torch's default initialization, or RAFT's feature-encoder kaiming_normal(fan_out, relu).</summary>
internal sealed class AmtConv2d<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight, _bias;
    private readonly int _stride, _padding;

    public AmtConv2d(int inChannels, int outChannels, int kernel, int stride = 1, int padding = -1, bool kaimingFanOut = false)
    {
        _stride = stride;
        _padding = padding < 0 ? kernel / 2 : padding;
        int fanIn = inChannels * kernel * kernel;
        var shape = new[] { outChannels, inChannels, kernel, kernel };
        if (kaimingFanOut)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            double std = Math.Sqrt(2.0 / (outChannels * kernel * kernel));
            _weight = new LazyWeight<T>(shape, (tensor, random) =>
            {
                for (int i = 0; i < tensor.Length; i++)
                {
                    double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                    tensor[i] = ops.FromDouble(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                }
            });
        }
        else
        {
            _weight = LazyWeight<T>.Uniform(shape, fanIn);
        }
        _bias = LazyWeight<T>.Uniform(new[] { outChannels }, fanIn);
    }

    public Tensor<T> Forward(Tensor<T> x)
        => MgldOps.ChannelBias(Engine, Engine.Conv2D(x, _weight.Value, new[] { _stride, _stride }, new[] { _padding, _padding }, new[] { 1, 1 }), _bias.Value);

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>nn.ConvTranspose2d(in, out, 4, 2, 1) with torch's default initialization (fan_in from the output channels).</summary>
internal sealed class AmtDeconv<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight, _bias;

    public AmtDeconv(int inChannels, int outChannels)
    {
        _weight = LazyWeight<T>.Uniform(new[] { inChannels, outChannels, 4, 4 }, outChannels * 16);
        _bias = LazyWeight<T>.Uniform(new[] { outChannels }, outChannels * 16);
    }

    public Tensor<T> Forward(Tensor<T> x)
        => MgldOps.ChannelBias(Engine, Engine.ConvTranspose2D(x, _weight.Value, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 0, 0 }), _bias.Value);

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>nn.PReLU(channels): a learned slope per channel, starting at 0.25.</summary>
internal sealed class AmtPRelu<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _slope;

    public AmtPRelu(int channels) => _slope = LazyWeight<T>.Constant(new[] { channels }, 0.25);

    public Tensor<T> Forward(Tensor<T> x)
    {
        var shape = Enumerable.Repeat(1, x.Rank).ToArray();
        shape[1] = _slope.Shape[0];
        var slope = Engine.TensorBroadcastTo(Engine.Reshape(_slope.Value, shape), x.Shape.ToArray());
        return Engine.TensorSubtract(Engine.ReLU(x), Engine.TensorMultiply(slope, Engine.ReLU(Engine.TensorNegate(x))));
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _slope;
    }
}

/// <summary>convrelu: a convolution followed by a per-channel PReLU.</summary>
internal sealed class AmtConvPRelu<T> : LazyWeightModule<T>
{
    private readonly AmtConv2d<T> _conv;
    private readonly AmtPRelu<T> _prelu;

    public AmtConvPRelu(int inChannels, int outChannels, int kernel = 3, int stride = 1)
    {
        _conv = new AmtConv2d<T>(inChannels, outChannels, kernel, stride);
        _prelu = new AmtPRelu<T>(outChannels);
    }

    public Tensor<T> Forward(Tensor<T> x) => _prelu.Forward(_conv.Forward(x));

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv, _prelu };
}

/// <summary>The release's tensor operations (raft.py, ifrnet.py, flow_utils.py, loss.py).</summary>
internal static class AmtOps
{
    /// <summary>nn.InstanceNorm2d (no affine, eps 1e-5): each sample's channels normalized over their pixels.</summary>
    internal static Tensor<T> InstanceNorm<T>(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int c = x.Shape[1];
        var gamma = new Tensor<T>(new[] { c });
        gamma.Fill(ops.One);
        return engine.GroupNorm(x, c, gamma, new Tensor<T>(new[] { c }), 1e-5, out _, out _);
    }

    /// <summary>F.interpolate(mode='bilinear', align_corners=False) to an explicit size.</summary>
    internal static Tensor<T> Resize<T>(IEngine engine, Tensor<T> x, int height, int width)
        => x.Shape[2] == height && x.Shape[3] == width ? x : engine.Interpolate(x, new[] { height, width }, InterpolateMode.Bilinear, false);

    /// <summary>flow_utils.warp: bilinear backward warp, border padding, align_corners=True.</summary>
    internal static Tensor<T> Warp<T>(IEngine engine, Tensor<T> x, Tensor<T> flow)
        => BasicVsrBlocks.FlowWarp(engine, x, flow, GridSamplePadding.Border);

    /// <summary>coords_grid: [B, 2, h, w] holding each pixel's (x, y).</summary>
    internal static Tensor<T> CoordsGrid<T>(int batch, int height, int width)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var grid = new Tensor<T>(new[] { batch, 2, height, width });
        for (int b = 0; b < batch; b++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    grid[b, 0, y, x] = ops.FromDouble(x);
                    grid[b, 1, y, x] = ops.FromDouble(y);
                }
        return grid;
    }

    /// <summary>
    /// BidirCorrBlock's pyramids: the all-pairs feature correlation f0ᵀf1 / √D as [B·h·w, 1, h, w] volumes, its
    /// transpose likewise, each average-pooled by 2 per further level.
    /// </summary>
    internal static (Tensor<T>[] Forward, Tensor<T>[] Transposed) CorrelationPyramids<T>(IEngine engine, Tensor<T> fmap0, Tensor<T> fmap1, int levels)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = fmap0.Shape[0], d = fmap0.Shape[1], h = fmap0.Shape[2], w = fmap0.Shape[3];
        var a = engine.TensorPermute(engine.Reshape(fmap0, new[] { b, d, h * w }), new[] { 0, 2, 1 });
        var corr = engine.TensorMultiplyScalar(engine.BatchMatMul(a, engine.Reshape(fmap1, new[] { b, d, h * w })), ops.FromDouble(1.0 / Math.Sqrt(d)));
        var corrT = engine.TensorPermute(corr, new[] { 0, 2, 1 });
        var forward = new Tensor<T>[levels];
        var transposed = new Tensor<T>[levels];
        forward[0] = engine.Reshape(corr, new[] { b * h * w, 1, h, w });
        transposed[0] = engine.Reshape(corrT, new[] { b * h * w, 1, h, w });
        for (int i = 1; i < levels; i++)
        {
            forward[i] = engine.AvgPool2D(forward[i - 1], 2, 2, 0, false);
            transposed[i] = engine.AvgPool2D(transposed[i - 1], 2, 2, 0, false);
        }
        return (forward, transposed);
    }

    /// <summary>
    /// BidirCorrBlock.__call__ for one pyramid: for every pixel, the (2r + 1)² window of its correlation volume around
    /// coords / 2^level at every level, concatenated over the levels. RAFT's window is stacked (dy, dx) and added to
    /// (x, y), so tap (i, j) samples (x + i − r, y + j − r); zero outside the volume.
    /// </summary>
    internal static Tensor<T> Lookup<T>(IEngine engine, Tensor<T>[] pyramid, Tensor<T> coords, int radius)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = coords.Shape[0], h = coords.Shape[2], w = coords.Shape[3], k = 2 * radius + 1;
        var points = engine.Reshape(engine.TensorPermute(coords, new[] { 0, 2, 3, 1 }), new[] { b * h * w, 1, 1, 2 });
        var delta = new Tensor<T>(new[] { 1, k, k, 2 });
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
            {
                delta[0, i, j, 0] = ops.FromDouble(i - radius);
                delta[0, i, j, 1] = ops.FromDouble(j - radius);
            }
        var windowShape = new[] { b * h * w, k, k, 2 };
        var levels = new Tensor<T>[pyramid.Length];
        for (int level = 0; level < pyramid.Length; level++)
        {
            var volume = pyramid[level];
            int lh = volume.Shape[2], lw = volume.Shape[3];
            var centroid = engine.TensorMultiplyScalar(points, ops.FromDouble(1.0 / (1 << level)));
            var window = engine.TensorAdd(engine.TensorBroadcastTo(centroid, windowShape), engine.TensorBroadcastTo(delta, windowShape));
            // bilinear_sampler: 2x / (W − 1) − 1 (a one-pixel level is guarded against the release's division by zero).
            var scale = new Tensor<T>(new[] { 1, 1, 1, 2 });
            scale[0] = ops.FromDouble(2.0 / Math.Max(1, lw - 1));
            scale[1] = ops.FromDouble(2.0 / Math.Max(1, lh - 1));
            var grid = engine.TensorAddScalar(engine.TensorMultiply(window, engine.TensorBroadcastTo(scale, windowShape)), ops.FromDouble(-1.0));
            var sampled = engine.GridSample(volume, grid, GridSampleMode.Bilinear, GridSamplePadding.Zeros, true);
            levels[level] = engine.TensorPermute(engine.Reshape(sampled, new[] { b, h, w, k * k }), new[] { 0, 3, 1, 2 });
        }
        return engine.TensorConcatenate(levels, 1);
    }

    /// <summary>The Charbonnier loss mean((x² + 1e-6)^½).</summary>
    internal static Tensor<T> Charbonnier<T>(IEngine engine, Tensor<T> prediction, Tensor<T> target)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var diff = engine.TensorSubtract(prediction, target);
        var rho = engine.TensorSqrt(engine.TensorAddScalar(engine.TensorSquare(diff), ops.FromDouble(1e-6)));
        return engine.ReduceMean(rho, Enumerable.Range(0, rho.Rank).ToArray(), false);
    }

    // TernaryLoss.transform: the channel mean's 7x7 neighbourhood differences d mapped to d / √(0.81 + d²).
    private static Tensor<T> Census<T>(IEngine engine, Tensor<T> image)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var gray = engine.ReduceMean(image, new[] { 1 }, true);
        var identity = new Tensor<T>(new[] { 49, 1, 7, 7 });
        for (int k = 0; k < 49; k++) identity[k * 49 + k] = ops.One;
        var difference = engine.TensorSubtract(engine.Conv2D(gray, identity, new[] { 1, 1 }, new[] { 3, 3 }, new[] { 1, 1 }),
            engine.TensorBroadcastTo(gray, new[] { gray.Shape[0], 49, gray.Shape[2], gray.Shape[3] }));
        return engine.TensorDivide(difference, engine.TensorSqrt(engine.TensorAddScalar(engine.TensorSquare(difference), ops.FromDouble(0.81))));
    }

    /// <summary>
    /// TernaryLoss: the mean over the 49 taps of δ²/(0.1 + δ²) between the census transforms (the target's detached),
    /// zeroed within 3 pixels of the border and averaged over every pixel.
    /// </summary>
    internal static Tensor<T> Ternary<T>(IEngine engine, Tensor<T> prediction, Tensor<T> target)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var diff = engine.TensorSubtract(Census(engine, prediction), engine.StopGradient(Census(engine, target)));
        var squared = engine.TensorSquare(diff);
        var dist = engine.ReduceMean(engine.TensorDivide(squared, engine.TensorAddScalar(squared, ops.FromDouble(0.1))), new[] { 1 }, true);
        int n = dist.Shape[0], h = dist.Shape[2], w = dist.Shape[3];
        var mask = new Tensor<T>(dist._shape);
        for (int b = 0; b < n; b++)
            for (int y = 3; y < h - 3; y++)
                for (int x = 3; x < w - 3; x++)
                    mask[(b * h + y) * w + x] = ops.One;
        return engine.ReduceMean(engine.TensorMultiply(dist, mask), new[] { 0, 1, 2, 3 }, false);
    }
}
