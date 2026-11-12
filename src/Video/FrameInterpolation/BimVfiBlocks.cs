using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>arch.LayerNorm(data_format='channels_first'): each pixel's channels normalized (eps 1e-6), then a learned
/// per-channel scale and shift.</summary>
internal sealed class BimLayerNorm<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight, _bias;

    public BimLayerNorm(int channels)
    {
        _weight = LazyWeight<T>.Constant(new[] { channels }, 1.0);
        _bias = LazyWeight<T>.Constant(new[] { channels }, 0.0);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var centered = Engine.TensorSubtract(x, Engine.TensorBroadcastTo(Engine.ReduceMean(x, new[] { 1 }, true), x.Shape.ToArray()));
        var variance = Engine.ReduceMean(Engine.TensorSquare(centered), new[] { 1 }, true);
        var normalized = Engine.TensorDivide(centered,
            Engine.TensorBroadcastTo(Engine.TensorSqrt(Engine.TensorAddScalar(variance, ops.FromDouble(1e-6))), x.Shape.ToArray()));
        var scale = Engine.TensorBroadcastTo(Engine.Reshape(_weight.Value, new[] { 1, _weight.Shape[0], 1, 1 }), x.Shape.ToArray());
        return MgldOps.ChannelBias(Engine, Engine.TensorMultiply(normalized, scale), _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>arch.ResBlock: x + conv(LeakyReLU(conv(x))), both convolutions width-preserving.</summary>
internal sealed class BimResBlock<T> : LazyWeightModule<T>
{
    private readonly AmtConv2d<T> _conv1, _conv2;

    public BimResBlock(int channels, int kernel)
    {
        _conv1 = new AmtConv2d<T>(channels, channels, kernel);
        _conv2 = new AmtConv2d<T>(channels, channels, kernel);
    }

    public Tensor<T> Forward(Tensor<T> x)
        => Engine.TensorAdd(x, _conv2.Forward(Engine.LeakyReLU(_conv1.Forward(x), MathHelper.GetNumericOperations<T>().FromDouble(0.01))));

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv1, _conv2 };
}

/// <summary>resnet_encoder.BasicBlock (stride 1): pre-activation instance norms around two 3x3 convolutions, LeakyReLU
/// after each, plus the identity.</summary>
internal sealed class BimBasicBlock<T> : LazyWeightModule<T>
{
    private readonly AmtConv2d<T> _conv1, _conv2;

    public BimBasicBlock(int channels)
    {
        _conv1 = new AmtConv2d<T>(channels, channels, 3);
        _conv2 = new AmtConv2d<T>(channels, channels, 3);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var slope = MathHelper.GetNumericOperations<T>().FromDouble(0.01);
        var y = Engine.LeakyReLU(_conv1.Forward(AmtOps.InstanceNorm(Engine, x)), slope);
        y = Engine.LeakyReLU(_conv2.Forward(AmtOps.InstanceNorm(Engine, y)), slope);
        return Engine.TensorAdd(y, x);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv1, _conv2 };
}

/// <summary>resnet_encoder.ResNetPyramid: features at full, 1/2 and 1/4 size with C, 2C and 4C channels.</summary>
internal sealed class BimResNetPyramid<T> : LazyWeightModule<T>
{
    private readonly AmtConv2d<T> _stem, _down1, _down2, _last;
    private readonly BimBasicBlock<T> _block0, _block1, _block2;

    public BimResNetPyramid(int channels)
    {
        _stem = new AmtConv2d<T>(3, channels, 3);
        _block0 = new BimBasicBlock<T>(channels);
        _down1 = new AmtConv2d<T>(channels, channels * 2, 2, 2, 0);
        _block1 = new BimBasicBlock<T>(channels * 2);
        _down2 = new AmtConv2d<T>(channels * 2, channels * 4, 2, 2, 0);
        _block2 = new BimBasicBlock<T>(channels * 4);
        _last = new AmtConv2d<T>(channels * 4, channels * 4, 3);
    }

    public Tensor<T>[] Forward(Tensor<T> image)
    {
        var c0 = _block0.Forward(_stem.Forward(image));
        var c1 = _block1.Forward(_down1.Forward(c0));
        var c2 = _last.Forward(_block2.Forward(_down2.Forward(c1)));
        return new[] { c0, c1, c2 };
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _stem, _block0, _down1, _block1, _down2, _block2, _last };
}

/// <summary>The release's tensor operations (backwarp.py, costvol.py, caun.py, loss.py, F.interpolate).</summary>
internal static class BimOps
{
    /// <summary>backwarp: bilinear backward warp, zero padding, align_corners=True.</summary>
    internal static Tensor<T> Warp<T>(IEngine engine, Tensor<T> x, Tensor<T> flow)
        => BasicVsrBlocks.FlowWarp(engine, x, flow, GridSamplePadding.Zeros);

    /// <summary>
    /// costvol_func: for each displacement of a k x k window (y-major), the channel sum of one(p)·two(p + d), zero where
    /// p + d leaves the image.
    /// </summary>
    internal static Tensor<T> CostVolume<T>(IEngine engine, Tensor<T> one, Tensor<T> two, int kernel)
    {
        int r = (kernel - 1) / 2;
        int n = two.Shape[0], c = two.Shape[1], h = two.Shape[2], w = two.Shape[3];
        var padded = engine.PadNd(two, new[] { r, r, r, r }, PadMode.Constant, MathHelper.GetNumericOperations<T>().Zero);
        var maps = new Tensor<T>[kernel * kernel];
        for (int dy = 0; dy < kernel; dy++)
            for (int dx = 0; dx < kernel; dx++)
            {
                var shifted = engine.TensorSlice(padded, new[] { 0, 0, dy, dx }, new[] { n, c, h, w });
                maps[dy * kernel + dx] = engine.ReduceSum(engine.TensorMultiply(one, shifted), new[] { 1 }, true);
            }
        return engine.TensorConcatenate(maps, 1);
    }

    /// <summary>F.interpolate(mode='bilinear', align_corners=False) to an explicit size.</summary>
    internal static Tensor<T> Resize<T>(IEngine engine, Tensor<T> x, int height, int width) => AmtOps.Resize(engine, x, height, width);

    /// <summary>
    /// F.interpolate(scale_factor=1/factor, mode='bilinear', align_corners=False, antialias=True): PIL's separable
    /// triangle filter widened to the reduction factor, truncated and renormalized at the borders.
    /// </summary>
    internal static Tensor<T> AntialiasDownsample<T>(IEngine engine, Tensor<T> x, int factor)
    {
        if (factor == 1) return x;
        int n = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        int oh = h / factor, ow = w / factor;
        var rows = engine.TensorBroadcastTo(Engine3(AntialiasWeights<T>(h, oh, factor)), new[] { n * c, oh, h });
        var colsT = engine.TensorBroadcastTo(Engine3(Transpose(AntialiasWeights<T>(w, ow, factor))), new[] { n * c, w, ow });
        var flat = engine.Reshape(x, new[] { n * c, h, w });
        var reduced = engine.BatchMatMul(rows, engine.BatchMatMul(flat, colsT));
        return engine.Reshape(reduced, new[] { n, c, oh, ow });

        Tensor<T> Engine3(Tensor<T> m) => engine.Reshape(m, new[] { 1, m.Shape[0], m.Shape[1] });
    }

    // aten's _upsample_bilinear2d_aa weights: output o's center scale·(o + 0.5), support = scale, triangle filter.
    internal static Tensor<T> AntialiasWeights<T>(int input, int output, int factor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var m = new Tensor<T>(new[] { output, input });
        double scale = (double)input / output, support = scale;
        for (int o = 0; o < output; o++)
        {
            double center = scale * (o + 0.5);
            int lo = Math.Max((int)(center - support + 0.5), 0);
            int hi = Math.Min((int)(center + support + 0.5), input);
            var weights = new double[hi - lo];
            double total = 0;
            for (int j = lo; j < hi; j++)
            {
                double v = Math.Max(0.0, 1.0 - Math.Abs((j - center + 0.5) / scale));
                weights[j - lo] = v;
                total += v;
            }
            for (int j = lo; j < hi; j++)
                if (total > 0) m[o, j] = ops.FromDouble(weights[j - lo] / total);
        }
        return m;
    }

    private static Tensor<T> Transpose<T>(Tensor<T> m)
    {
        var t = new Tensor<T>(new[] { m.Shape[1], m.Shape[0] });
        for (int i = 0; i < m.Shape[0]; i++)
            for (int j = 0; j < m.Shape[1]; j++)
                t[j, i] = m[i, j];
        return t;
    }

    /// <summary>
    /// CAUN.upsample_input: each fine pixel a softmax-weighted (over 9 logits) combination of its coarse pixel's 3x3
    /// edge-replicated neighbourhood. <paramref name="logits"/> is [N, 9, factor·H, factor·W].
    /// </summary>
    internal static Tensor<T> ConvexUpsample<T>(IEngine engine, Tensor<T> input, Tensor<T> logits, int factor)
    {
        int n = input.Shape[0], c = input.Shape[1], h = input.Shape[2], w = input.Shape[3];
        var weights = engine.Softmax(logits, axis: 1);
        var padded = engine.PadNd(input, new[] { 1, 1, 1, 1 }, PadMode.Replicate, MathHelper.GetNumericOperations<T>().Zero);
        var fine = new[] { n, c, h * factor, w * factor };
        Tensor<T>? sum = null;
        for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++)
            {
                var tap = engine.TensorSlice(padded, new[] { 0, 0, ky, kx }, new[] { n, c, h, w });
                var repeated = engine.Reshape(engine.TensorBroadcastTo(engine.Reshape(tap, new[] { n, c, h, 1, w, 1 }),
                    new[] { n, c, h, factor, w, factor }), fine);
                var weight = engine.TensorBroadcastTo(engine.TensorNarrow(weights, 1, ky * 3 + kx, 1), fine);
                var term = engine.TensorMultiply(weight, repeated);
                sum = sum is null ? term : engine.TensorAdd(sum, term);
            }
        return sum ?? throw new InvalidOperationException("The 3x3 neighbourhood is never empty.");
    }

    /// <summary>The Charbonnier loss mean((x² + 1e-6)^½).</summary>
    internal static Tensor<T> Charbonnier<T>(IEngine engine, Tensor<T> prediction, Tensor<T> target) => AmtOps.Charbonnier(engine, prediction, target);

    /// <summary>The flow losses' elementwise (x² + 1e-12)^½.</summary>
    internal static Tensor<T> FlowCharbonnier<T>(IEngine engine, Tensor<T> x)
        => engine.TensorSqrt(engine.TensorAddScalar(engine.TensorSquare(x), MathHelper.GetNumericOperations<T>().FromDouble(1e-12)));

    internal static Tensor<T> MeanAll<T>(IEngine engine, Tensor<T> x) => engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), false);

    // Ternary.transform on rgb2gray: 7x7 neighbourhood differences d mapped to d / √(0.81 + d²).
    private static Tensor<T> Census<T>(IEngine engine, Tensor<T> image)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var gray = engine.TensorAdd(engine.TensorAdd(
            engine.TensorMultiplyScalar(engine.TensorNarrow(image, 1, 0, 1), ops.FromDouble(0.2989)),
            engine.TensorMultiplyScalar(engine.TensorNarrow(image, 1, 1, 1), ops.FromDouble(0.5870))),
            engine.TensorMultiplyScalar(engine.TensorNarrow(image, 1, 2, 1), ops.FromDouble(0.1140)));
        var identity = new Tensor<T>(new[] { 49, 1, 7, 7 });
        for (int k = 0; k < 49; k++) identity[k * 49 + k] = ops.One;
        var difference = engine.TensorSubtract(engine.Conv2D(gray, identity, new[] { 1, 1 }, new[] { 3, 3 }, new[] { 1, 1 }),
            engine.TensorBroadcastTo(gray, new[] { gray.Shape[0], 49, gray.Shape[2], gray.Shape[3] }));
        return engine.TensorDivide(difference, engine.TensorSqrt(engine.TensorAddScalar(engine.TensorSquare(difference), ops.FromDouble(0.81))));
    }

    /// <summary>
    /// loss.Ternary: the mean over the 49 taps of δ²/(0.1 + δ²) between the census transforms, zeroed on the
    /// one-pixel border and averaged over every pixel.
    /// </summary>
    internal static Tensor<T> Ternary<T>(IEngine engine, Tensor<T> prediction, Tensor<T> target)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var squared = engine.TensorSquare(engine.TensorSubtract(Census(engine, target), Census(engine, prediction)));
        var dist = engine.ReduceMean(engine.TensorDivide(squared, engine.TensorAddScalar(squared, ops.FromDouble(0.1))), new[] { 1 }, true);
        int n = dist.Shape[0], h = dist.Shape[2], w = dist.Shape[3];
        var mask = new Tensor<T>(dist._shape);
        for (int b = 0; b < n; b++)
            for (int y = 1; y < h - 1; y++)
                for (int x = 1; x < w - 1; x++)
                    mask[(b * h + y) * w + x] = ops.One;
        return MeanAll(engine, engine.TensorMultiply(dist, mask));
    }
}
