using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>How an ABME weight starts.</summary>
internal enum AbmeInit
{
    /// <summary>The motion networks' re-initialization: <c>kaiming_normal(mode='fan_in')</c> with a zero bias.</summary>
    KaimingFanIn,

    /// <summary>torch's default, which the synthesis network keeps: uniform in ±1/√fan_in.</summary>
    TorchDefault,
}

/// <summary>nn.Conv2d with ABME's options (dilation, bias, initialization), optionally followed by LeakyReLU(0.1).</summary>
internal sealed class AbmeConv2d<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T>? _bias;
    private readonly int _stride, _padding, _dilation;
    private readonly bool _leaky;

    public AbmeConv2d(int inChannels, int outChannels, int kernel, int stride, int padding, int dilation, AbmeInit init,
        bool leaky, bool bias = true)
    {
        _stride = stride;
        _padding = padding;
        _dilation = dilation;
        _leaky = leaky;
        int fanIn = inChannels * kernel * kernel;
        _weight = AbmeOps.Weight<T>(new[] { outChannels, inChannels, kernel, kernel }, fanIn, init);
        if (bias) _bias = init == AbmeInit.TorchDefault ? LazyWeight<T>.Uniform(new[] { outChannels }, fanIn) : LazyWeight<T>.Constant(new[] { outChannels }, 0.0);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = Engine.Conv2D(x, _weight.Value, new[] { _stride, _stride }, new[] { _padding, _padding }, new[] { _dilation, _dilation });
        if (_bias is not null) y = MgldOps.ChannelBias(Engine, y, _bias.Value);
        return _leaky ? Engine.LeakyReLU(y, NumOps.FromDouble(0.1)) : y;
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        if (_bias is not null) yield return _bias;
    }
}

/// <summary>nn.ConvTranspose2d(in, out, 4, 2, 1), re-initialized as the motion networks do.</summary>
internal sealed class AbmeDeconv<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T> _bias;

    public AbmeDeconv(int inChannels, int outChannels)
    {
        // torch's fan_in for a transposed convolution reads weight.size(1), the output channels.
        _weight = AbmeOps.Weight<T>(new[] { inChannels, outChannels, 4, 4 }, outChannels * 16, AbmeInit.KaimingFanIn);
        _bias = LazyWeight<T>.Constant(new[] { outChannels }, 0.0);
    }

    public Tensor<T> Forward(Tensor<T> x)
        => MgldOps.ChannelBias(Engine, Engine.ConvTranspose2D(x, _weight.Value, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 0, 0 }), _bias.Value);

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>nn.PReLU(): one learned slope, starting at 0.25.</summary>
internal sealed class AbmePRelu<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _slope = LazyWeight<T>.Constant(new[] { 1 }, 0.25);

    public Tensor<T> Forward(Tensor<T> x)
    {
        var slope = Engine.TensorBroadcastTo(Engine.Reshape(_slope.Value, Enumerable.Repeat(1, x.Rank).ToArray()), x.Shape.ToArray());
        return Engine.TensorSubtract(Engine.ReLU(x), Engine.TensorMultiply(slope, Engine.ReLU(Engine.TensorNegate(x))));
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _slope;
    }
}

/// <summary>The release's tensor operations (utils.py, Bilateral_Correlation.py, Upsample.py, the correlation package).</summary>
internal static class AbmeOps
{
    internal static LazyWeight<T> Weight<T>(int[] shape, int fanIn, AbmeInit init)
    {
        if (init == AbmeInit.TorchDefault) return LazyWeight<T>.Uniform(shape, fanIn);
        var ops = MathHelper.GetNumericOperations<T>();
        return new LazyWeight<T>(shape, (tensor, random) =>
        {
            double std = Math.Sqrt(2.0 / fanIn);
            for (int i = 0; i < tensor.Length; i++)
            {
                double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                tensor[i] = ops.FromDouble(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
            }
        });
    }

    /// <summary>
    /// The warp's validity mask: grid_sample of ones (zero padding, align_corners=True) at the flowed positions, 1 where
    /// it reaches 0.999 and 0 elsewhere. Every entry is overwritten by a constant, so no gradient passes through it.
    /// </summary>
    internal static Tensor<T> ValidMask<T>(Tensor<T> flow)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = flow.Shape[0], h = flow.Shape[2], w = flow.Shape[3];
        var mask = new Tensor<T>(new[] { n, 1, h, w });
        for (int b = 0; b < n; b++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double px = w > 1 ? x + ops.ToDouble(flow[b, 0, y, x]) : 0.0;
                    double py = h > 1 ? y + ops.ToDouble(flow[b, 1, y, x]) : 0.0;
                    int x0 = (int)Math.Floor(px), y0 = (int)Math.Floor(py);
                    double wx0 = 1.0 - (px - x0), wy0 = 1.0 - (py - y0);
                    double total = 0;
                    for (int dy = 0; dy <= 1; dy++)
                        for (int dx = 0; dx <= 1; dx++)
                        {
                            int xi = x0 + dx, yi = y0 + dy;
                            if (xi < 0 || xi >= w || yi < 0 || yi >= h) continue;
                            total += (dx == 0 ? wx0 : 1 - wx0) * (dy == 0 ? wy0 : 1 - wy0);
                        }
                    mask[b, 0, y, x] = total < 0.999 ? ops.Zero : ops.One;
                }
        return mask;
    }

    private static Tensor<T> Masked<T>(IEngine engine, Tensor<T> warped, Tensor<T> flow)
        => engine.TensorMultiply(warped, engine.TensorBroadcastTo(ValidMask(flow), warped.Shape.ToArray()));

    /// <summary>utils.warp: bilinear backward warp (zero padding, align_corners=True) times the validity mask.</summary>
    internal static Tensor<T> Warp<T>(IEngine engine, Tensor<T> x, Tensor<T> flow)
        => Masked(engine, BasicVsrBlocks.FlowWarp(engine, x, flow, GridSamplePadding.Zeros), flow);

    /// <summary>utils.warp with its mask: the warped tensor and the [N, 1, H, W] mask.</summary>
    internal static (Tensor<T> Warped, Tensor<T> Mask) WarpWithMask<T>(IEngine engine, Tensor<T> x, Tensor<T> flow)
    {
        var mask = ValidMask(flow);
        var warped = engine.TensorMultiply(BasicVsrBlocks.FlowWarp(engine, x, flow, GridSamplePadding.Zeros),
            engine.TensorBroadcastTo(mask, x.Shape.ToArray()));
        return (warped, mask);
    }

    // BilateralCorrelation.warp: border padding for the features, the mask still from zero padding.
    private static Tensor<T> BorderWarp<T>(IEngine engine, Tensor<T> x, Tensor<T> flow)
        => Masked(engine, BasicVsrBlocks.FlowWarp(engine, x, flow, GridSamplePadding.Border), flow);

    /// <summary>x / √(Σ_c x² + 1e-6).</summary>
    internal static Tensor<T> L2Normalize<T>(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var norm = engine.TensorSqrt(engine.TensorAddScalar(engine.ReduceSum(engine.TensorSquare(x), new[] { 1 }, true), ops.FromDouble(1e-6)));
        return engine.TensorDivide(x, engine.TensorBroadcastTo(norm, x.Shape.ToArray()));
    }

    /// <summary>
    /// BilateralCorrelation at time 0.5: the channel-normalized features of both frames warped from the intermediate
    /// time along −(V + d) and +(V + d) for every displacement d in [−md, md]² (x fastest), their channel inner products.
    /// </summary>
    internal static Tensor<T> BilateralCorrelation<T>(IEngine engine, Tensor<T> feature1, Tensor<T> feature2, Tensor<T> sbm, int md)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var f1 = L2Normalize(engine, feature1);
        var f2 = L2Normalize(engine, feature2);
        int n = sbm.Shape[0], h = sbm.Shape[2], w = sbm.Shape[3];
        var maps = new List<Tensor<T>>();
        for (int v = -md; v <= md; v++)
            for (int u = -md; u <= md; u++)
            {
                var offset = new Tensor<T>(new[] { n, 2, h, w });
                for (int b = 0; b < n; b++)
                    for (int i = 0; i < h * w; i++)
                    {
                        offset[(b * 2) * h * w + i] = ops.FromDouble(u);
                        offset[(b * 2 + 1) * h * w + i] = ops.FromDouble(v);
                    }
                var displaced = engine.TensorAdd(sbm, offset);
                var backward = BorderWarp(engine, f1, engine.TensorNegate(displaced));
                var forward = BorderWarp(engine, f2, displaced);
                maps.Add(engine.ReduceSum(engine.TensorMultiply(forward, backward), new[] { 1 }, true));
            }
        return engine.TensorConcatenate(maps.ToArray(), 1);
    }

    /// <summary>
    /// FlowNet's correlation (kernel 1, stride 1, padding and displacement <paramref name="md"/>): for every displacement
    /// (y-major), the channel mean of f1(x)·f2(x + d), f2 zero outside the image.
    /// </summary>
    internal static Tensor<T> Correlation<T>(IEngine engine, Tensor<T> f1, Tensor<T> f2, int md)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = f2.Shape[0], c = f2.Shape[1], h = f2.Shape[2], w = f2.Shape[3];
        var rows = new Tensor<T>(new[] { n, c, md, w });
        var padded = engine.TensorConcatenate(new[] { rows, f2, rows }, 2);
        var cols = new Tensor<T>(new[] { n, c, h + 2 * md, md });
        padded = engine.TensorConcatenate(new[] { cols, padded, cols }, 3);
        var maps = new List<Tensor<T>>();
        for (int dy = -md; dy <= md; dy++)
            for (int dx = -md; dx <= md; dx++)
            {
                var shifted = engine.TensorSlice(padded, new[] { 0, 0, md + dy, md + dx }, new[] { n, c, h, w });
                maps.Add(engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(f1, shifted), new[] { 1 }, true),
                    ops.FromDouble(1.0 / c)));
            }
        return engine.TensorConcatenate(maps.ToArray(), 1);
    }

    /// <summary>
    /// Upsample.Upsample(x, 2): a transposed convolution with the bilinear kernel [½, 1, ½]² after replicate padding,
    /// i.e. even samples copy the input and odd samples average a neighbour pair, the last edge repeated.
    /// </summary>
    internal static Tensor<T> Upsample2<T>(IEngine engine, Tensor<T> x)
    {
        Tensor<T> Along(Tensor<T> t, int axis)
        {
            int size = t.Shape[axis];
            var next = size > 1
                ? engine.TensorConcatenate(new[] { engine.TensorNarrow(t, axis, 1, size - 1), engine.TensorNarrow(t, axis, size - 1, 1) }, axis)
                : t;
            var odd = engine.TensorMultiplyScalar(engine.TensorAdd(t, next), MathHelper.GetNumericOperations<T>().FromDouble(0.5));
            var shape = t.Shape.ToArray();
            var expanded = shape.Take(axis + 1).Concat(new[] { 1 }).Concat(shape.Skip(axis + 1)).ToArray();
            var interleaved = engine.TensorConcatenate(new[] { engine.Reshape(t, expanded), engine.Reshape(odd, expanded) }, axis + 1);
            shape[axis] = 2 * size;
            return engine.Reshape(interleaved, shape);
        }
        return Along(Along(x, 2), 3);
    }

    /// <summary>F.interpolate(mode='bilinear', align_corners=False) to an explicit size.</summary>
    internal static Tensor<T> Resize<T>(IEngine engine, Tensor<T> x, int height, int width, InterpolateMode mode = InterpolateMode.Bilinear)
        => x.Shape[2] == height && x.Shape[3] == width ? x : engine.Interpolate(x, new[] { height, width }, mode, false);

    /// <summary>Scales the x and y flow channels separately.</summary>
    internal static Tensor<T> ScaleFlow<T>(IEngine engine, Tensor<T> flow, double sx, double sy)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (sx == sy) return engine.TensorMultiplyScalar(flow, ops.FromDouble(sx));
        return engine.TensorConcatenate(new[]
        {
            engine.TensorMultiplyScalar(engine.TensorNarrow(flow, 1, 0, 1), ops.FromDouble(sx)),
            engine.TensorMultiplyScalar(engine.TensorNarrow(flow, 1, 1, 1), ops.FromDouble(sy)),
        }, 1);
    }

    /// <summary>The Charbonnier penalty ρ(x) = (x² + ε²)^½, averaged.</summary>
    internal static Tensor<T> Charbonnier<T>(IEngine engine, Tensor<T> prediction, Tensor<T> target, double epsilon)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var diff = engine.TensorSubtract(prediction, target);
        var rho = engine.TensorSqrt(engine.TensorAddScalar(engine.TensorSquare(diff), ops.FromDouble(epsilon * epsilon)));
        return engine.ReduceMean(rho, Enumerable.Range(0, rho.Rank).ToArray(), false);
    }

    // The 7x7 census transform of the 0-255 luma: each neighbour's difference d mapped to d / √(0.81 + d²).
    private static Tensor<T> Census<T>(IEngine engine, Tensor<T> image)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> Channel(int c, double weight) => engine.TensorMultiplyScalar(engine.TensorNarrow(image, 1, c, 1), ops.FromDouble(255.0 * weight));
        var gray = engine.TensorAdd(engine.TensorAdd(Channel(0, 0.299), Channel(1, 0.587)), Channel(2, 0.114));
        var identity = new Tensor<T>(new[] { 49, 1, 7, 7 });
        for (int k = 0; k < 49; k++) identity[k * 49 + k] = ops.One;
        var patches = engine.Conv2D(gray, identity, new[] { 1, 1 }, new[] { 3, 3 }, new[] { 1, 1 });
        var difference = engine.TensorSubtract(patches, engine.TensorBroadcastTo(gray, patches.Shape.ToArray()));
        var scale = engine.TensorSqrt(engine.TensorAddScalar(engine.TensorSquare(difference), ops.FromDouble(0.81)));
        return engine.TensorDivide(difference, scale);
    }

    /// <summary>
    /// The census loss: the soft Hamming distance Σ_k δ²/(0.1 + δ²) between the two images' 7x7 census transforms,
    /// averaged over the pixels whose patch lies inside the image (UnFlow's ternary loss, Meister et al. 2018).
    /// </summary>
    internal static Tensor<T> CensusLoss<T>(IEngine engine, Tensor<T> prediction, Tensor<T> target)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var delta = engine.TensorSquare(engine.TensorSubtract(Census(engine, prediction), Census(engine, target)));
        var soft = engine.TensorDivide(delta, engine.TensorAddScalar(delta, ops.FromDouble(0.1)));
        var distance = engine.ReduceSum(soft, new[] { 1 }, true);
        int n = distance.Shape[0], h = distance.Shape[2], w = distance.Shape[3];
        var valid = new Tensor<T>(distance._shape);
        int count = 0;
        for (int b = 0; b < n; b++)
            for (int y = 3; y < h - 3; y++)
                for (int x = 3; x < w - 3; x++)
                {
                    valid[(b * h + y) * w + x] = ops.One;
                    count++;
                }
        var sum = engine.ReduceSum(engine.TensorMultiply(distance, valid), new[] { 0, 1, 2, 3 }, false);
        return engine.TensorMultiplyScalar(sum, ops.FromDouble(1.0 / Math.Max(1, count)));
    }
}
