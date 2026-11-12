using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>The release's operators (m2m.py, pwcnet.py, costvol.py, loss.py).</summary>
internal static class M2mOps
{
    /// <summary>
    /// costvol_func: for each of the 9x9 displacements (row-major, vertical outer), the channel mean of |one − two| at the
    /// displaced position, |one| where it falls outside, which is the zero-padded second map.
    /// </summary>
    internal static Tensor<T> CostVolume<T>(IEngine engine, Tensor<T> one, Tensor<T> two)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = one.Shape[0], c = one.Shape[1], h = one.Shape[2], w = one.Shape[3];
        var padded = engine.PadNd(two, new[] { 4, 4, 4, 4 }, PadMode.Constant, ops.Zero);
        var planes = new Tensor<T>[81];
        for (int dy = 0; dy < 9; dy++)
            for (int dx = 0; dx < 9; dx++)
            {
                var shifted = engine.TensorSlice(padded, new[] { 0, 0, dy, dx }, new[] { n, c, h, w });
                planes[dy * 9 + dx] = engine.ReduceMean(engine.TensorAbs(engine.TensorSubtract(one, shifted)), new[] { 1 }, true);
            }
        return engine.TensorConcatenate(planes, 1);
    }

    /// <summary>F.avg_pool2d(kernel_size=2, stride=2) on even sides.</summary>
    internal static Tensor<T> Pool2<T>(IEngine engine, Tensor<T> x)
    {
        int n = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        var blocks = engine.Reshape(x.Contiguous(), new[] { n, c, h / 2, 2, w / 2, 2 });
        return engine.ReduceMean(blocks, new[] { 3, 5 }, false);
    }

    /// <summary>
    /// forwarp_mframe_mask: every branch's frames splatted (summation) along their flows with the weight td·exp(Z),
    /// Z clipped to [−20, 20], summed over the branches and both directions and divided by the summed weights, each
    /// direction's plus 1e-7. Pixels whose summed weight stays below 1e-5 are reported as holes.
    /// </summary>
    internal static (Tensor<T> Output, Tensor<T> Holes) ForwardWarpFrames<T>(IEngine engine, IReadOnlyList<Tensor<T>> images0,
        IReadOnlyList<Tensor<T>> flows0, double time0, IReadOnlyList<Tensor<T>> images1, IReadOnlyList<Tensor<T>> flows1, double time1,
        IReadOnlyList<Tensor<T>> metrics0, IReadOnlyList<Tensor<T>> metrics1)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T>? output = null, normalize = null;
        void OneDirection(Tensor<T> image, Tensor<T> flow, double td, Tensor<T> metric)
        {
            var weight = engine.TensorMultiplyScalar(engine.TensorExp(engine.TensorClamp(metric, ops.FromDouble(-20.0), ops.FromDouble(20.0))), ops.FromDouble(td));
            int c = image.Shape[1];
            var splatted = SoftSplatOps.Splat(engine, engine.TensorConcatenate(new[] { engine.TensorMultiply(image, engine.TensorBroadcastTo(weight, image.Shape.ToArray())), weight }, 1), flow);
            var values = engine.TensorNarrow(splatted, 1, 0, c);
            var norm = engine.TensorAddScalar(engine.TensorNarrow(splatted, 1, c, 1), ops.FromDouble(1e-7));
            output = output is null ? values : engine.TensorAdd(output, values);
            normalize = normalize is null ? norm : engine.TensorAdd(normalize, norm);
        }
        for (int b = 0; b < flows0.Count; b++)
        {
            OneDirection(images0[b], flows0[b], time0, metrics0[b]);
            OneDirection(images1[b], flows1[b], time1, metrics1[b]);
        }
        if (output is null || normalize is null) throw new InvalidOperationException("Many-to-many splatting needs at least one branch.");
        var sums = normalize.ToArray();
        var holes = new Tensor<T>(normalize._shape);
        for (int i = 0; i < sums.Length; i++) holes[i] = ops.ToDouble(sums[i]) < 1e-5 ? ops.One : ops.Zero;
        return (engine.TensorDivide(output, engine.TensorBroadcastTo(normalize, output.Shape.ToArray())), holes);
    }

    // Ternary.transform on rgb2gray (0.2989, 0.5870, 0.1140): the 7x7 neighbourhood differences d, zero padded, mapped to
    // d / √(0.81 + d²).
    private static Tensor<T> Census<T>(IEngine engine, Tensor<T> image)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var weights = new Tensor<T>(new[] { 1, 3, 1, 1 });
        weights[0] = ops.FromDouble(0.2989);
        weights[1] = ops.FromDouble(0.5870);
        weights[2] = ops.FromDouble(0.1140);
        var gray = engine.Conv2D(image, weights, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var identity = new Tensor<T>(new[] { 49, 1, 7, 7 });
        for (int k = 0; k < 49; k++) identity[k * 49 + k] = ops.One;
        var difference = engine.TensorSubtract(engine.Conv2D(gray, identity, new[] { 1, 1 }, new[] { 3, 3 }, new[] { 1, 1 }),
            engine.TensorBroadcastTo(gray, new[] { gray.Shape[0], 49, gray.Shape[2], gray.Shape[3] }));
        return engine.TensorDivide(difference, engine.TensorSqrt(engine.TensorAddScalar(engine.TensorSquare(difference), ops.FromDouble(0.81))));
    }

    /// <summary>
    /// loss.Ternary: the mean over the 49 taps of δ²/(0.1 + δ²) between the census transforms, zeroed on the one-pixel
    /// border and averaged over every pixel.
    /// </summary>
    internal static Tensor<T> Ternary<T>(IEngine engine, Tensor<T> prediction, Tensor<T> target)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var squared = engine.TensorSquare(engine.TensorSubtract(Census(engine, prediction), Census(engine, target)));
        var dist = engine.ReduceMean(engine.TensorDivide(squared, engine.TensorAddScalar(squared, ops.FromDouble(0.1))), new[] { 1 }, true);
        int n = dist.Shape[0], h = dist.Shape[2], w = dist.Shape[3];
        var mask = new Tensor<T>(dist._shape);
        for (int b = 0; b < n; b++)
            for (int y = 1; y < h - 1; y++)
                for (int x = 1; x < w - 1; x++)
                    mask[(b * h + y) * w + x] = ops.One;
        return engine.ReduceMean(engine.TensorMultiply(dist, mask), new[] { 0, 1, 2, 3 }, false);
    }
}

/// <summary>pwcnet.Basic's convolutions: biased, torch's default initialization, 'sconv(2)' a 2x2 stride-2 convolution
/// without padding and 'conv(3,replpad)' a 3x3 convolution on the replicate-padded input.</summary>
internal sealed class M2mPwcConv<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight, _bias;
    private readonly bool _strided;

    public M2mPwcConv(int inChannels, int outChannels, bool strided)
    {
        _strided = strided;
        int kernel = strided ? 2 : 3, fanIn = inChannels * kernel * kernel;
        _weight = LazyWeight<T>.Uniform(new[] { outChannels, inChannels, kernel, kernel }, fanIn);
        _bias = LazyWeight<T>.Uniform(new[] { outChannels }, fanIn);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = _strided
            ? Engine.Conv2D(x, _weight.Value, new[] { 2, 2 }, new[] { 0, 0 }, new[] { 1, 1 })
            : Engine.Conv2D(Engine.PadNd(x, new[] { 1, 1, 1, 1 }, PadMode.Replicate, NumOps.Zero), _weight.Value, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        return MgldOps.ChannelBias(Engine, y, _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>pwcnet.Network.Extractor: three levels of sconv(2), conv, conv each followed by a single-slope PReLU, then
/// two 2x2 average pools.</summary>
internal sealed class M2mPwcExtractor<T> : LazyWeightModule<T>
{
    private readonly M2mPwcConv<T>[] _convs;
    private readonly AmtPRelu<T>[] _slopes;

    public M2mPwcExtractor(int channels)
    {
        _convs = Enumerable.Range(0, 9).Select(i => new M2mPwcConv<T>(i == 0 ? 3 : channels, channels, i % 3 == 0)).ToArray();
        _slopes = Enumerable.Range(0, 9).Select(_ => new AmtPRelu<T>(1)).ToArray();
    }

    public Tensor<T>[] Forward(Tensor<T> x)
    {
        var levels = new Tensor<T>[5];
        for (int level = 0; level < 3; level++)
        {
            for (int i = 3 * level; i < 3 * level + 3; i++) x = _slopes[i].Forward(_convs[i].Forward(x));
            levels[level] = x;
        }
        levels[3] = M2mOps.Pool2(Engine, levels[2]);
        levels[4] = M2mOps.Pool2(Engine, levels[3]);
        return levels;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _convs.Cast<LazyWeightModule<T>>().Concat(_slopes);
}

/// <summary>pwcnet.Network.Decoder: the PReLU'd cost volume against the warped second map, the first map and the 2x
/// upsampled flow, through five convolution-PReLU pairs and a convolution to the flow residual.</summary>
internal sealed class M2mPwcDecoder<T> : LazyWeightModule<T>
{
    private readonly AmtPRelu<T> _costActivation;
    private readonly M2mPwcConv<T>[] _convs;
    private readonly AmtPRelu<T>[] _slopes;

    public M2mPwcDecoder(int inChannels, IReadOnlyList<int> hidden)
    {
        _costActivation = new AmtPRelu<T>(1);
        var widths = new[] { inChannels }.Concat(hidden).Concat(new[] { 2 }).ToArray();
        _convs = Enumerable.Range(0, widths.Length - 1).Select(i => new M2mPwcConv<T>(widths[i], widths[i + 1], false)).ToArray();
        _slopes = Enumerable.Range(0, hidden.Count).Select(_ => new AmtPRelu<T>(1)).ToArray();
    }

    public Tensor<T> Forward(Tensor<T> one, Tensor<T> two, Tensor<T>? flow)
    {
        Tensor<T> x;
        if (flow is null)
        {
            x = Engine.TensorConcatenate(new[] { one, _costActivation.Forward(M2mOps.CostVolume(Engine, one, two)) }, 1);
        }
        else
        {
            flow = Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flow, flow.Shape[2] * 2, flow.Shape[3] * 2), NumOps.FromDouble(2.0));
            var warped = SoftSplatOps.Backwarp(Engine, two, Engine.StopGradient(flow));
            x = Engine.TensorConcatenate(new[] { one, _costActivation.Forward(M2mOps.CostVolume(Engine, one, warped)), flow }, 1);
        }
        for (int i = 0; i < _slopes.Length; i++) x = _slopes[i].Forward(_convs[i].Forward(x));
        var residual = _convs[_convs.Length - 1].Forward(x);
        return flow is null ? residual : Engine.TensorAdd(flow, residual);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _costActivation }.Concat(_convs).Concat(_slopes);
}

/// <summary>pwcnet.Network.bidir: the flows from the first frame to the second and back, at half the input's size.</summary>
internal sealed class M2mPwc<T> : LazyWeightModule<T>
{
    private readonly M2mPwcExtractor<T> _extractor;
    private readonly M2mPwcDecoder<T>[] _decoders;

    public M2mPwc(int channels, IReadOnlyList<int> hidden)
    {
        _extractor = new M2mPwcExtractor<T>(channels);
        // Finest first (netOne to netFiv); the coarsest reads no previous flow.
        _decoders = Enumerable.Range(0, 5).Select(i => new M2mPwcDecoder<T>(channels + 81 + (i < 4 ? 2 : 0), hidden)).ToArray();
    }

    public (Tensor<T> Forward, Tensor<T> Backward) Forward(Tensor<T> one, Tensor<T> two)
    {
        var a = _extractor.Forward(one);
        var b = _extractor.Forward(two);
        Tensor<T>? forward = null, backward = null;
        for (int level = 4; level >= 0; level--)
        {
            forward = _decoders[level].Forward(a[level], b[level], forward);
            backward = _decoders[level].Forward(b[level], a[level], backward);
        }
        if (forward is null || backward is null) throw new InvalidOperationException("The flow estimator has no level.");
        return (forward, backward);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _extractor }.Concat(_decoders);
}

/// <summary>m2m.Conv2: a strided and a plain 3x3 convolution, each with a per-channel PReLU.</summary>
internal sealed class M2mConv2<T> : LazyWeightModule<T>
{
    private readonly AmtConvPRelu<T> _conv1, _conv2;

    public M2mConv2(int inChannels, int outChannels)
    {
        _conv1 = new AmtConvPRelu<T>(inChannels, outChannels, 3, 2);
        _conv2 = new AmtConvPRelu<T>(outChannels, outChannels);
    }

    public Tensor<T> Forward(Tensor<T> x) => _conv2.Forward(_conv1.Forward(x));

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv1, _conv2 };
}

/// <summary>m2m.deconv: a 4x4 stride-2 transposed convolution and a per-channel PReLU.</summary>
internal sealed class M2mDeconv<T> : LazyWeightModule<T>
{
    private readonly AmtDeconv<T> _deconv;
    private readonly AmtPRelu<T> _prelu;

    public M2mDeconv(int inChannels, int outChannels)
    {
        _deconv = new AmtDeconv<T>(inChannels, outChannels);
        _prelu = new AmtPRelu<T>(outChannels);
    }

    public Tensor<T> Forward(Tensor<T> x) => _prelu.Forward(_deconv.Forward(x));

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _deconv, _prelu };
}

/// <summary>
/// The Motion Refinement Network (m2m.MotionRefineNet: ImgPyramid and EncDec): both frames' feature pyramids and a
/// shared encoder that warps each frame's features to the other along the coarse flows, a rank-R channel-height-width
/// attention cube at 1/16 size, and a decoder to <c>branch</c> flow residuals per direction and a photometric weight.
/// </summary>
internal sealed class M2mRefine<T> : LazyWeightModule<T>
{
    private readonly int _branch, _rank, _deepest;
    private readonly M2mConv2<T>[] _pyramid, _down;
    private readonly M2mDeconv<T>[] _up;
    private readonly AmtConv2d<T> _flow, _weight, _cubeChannel, _cubeHeight, _cubeWidth;

    public M2mRefine(int c, int branch, int rank)
    {
        _branch = branch;
        _rank = rank;
        _deepest = 16 * c;
        _pyramid = new[] { new M2mConv2<T>(3, c), new M2mConv2<T>(c, 2 * c), new M2mConv2<T>(2 * c, 4 * c), new M2mConv2<T>(4 * c, 8 * c) };
        _down = new[] { new M2mConv2<T>(8, 2 * c), new M2mConv2<T>(6 * c, 4 * c), new M2mConv2<T>(12 * c, 8 * c), new M2mConv2<T>(24 * c, 16 * c) };
        _up = new[] { new M2mDeconv<T>(48 * c, 8 * c), new M2mDeconv<T>(16 * c, 4 * c), new M2mDeconv<T>(8 * c, 2 * c), new M2mDeconv<T>(4 * c, c) };
        _flow = new AmtConv2d<T>(c, 2 * branch, 3);
        _weight = new AmtConv2d<T>(c, 1, 3);
        _cubeChannel = new AmtConv2d<T>(_deepest, rank * _deepest, 1);
        _cubeHeight = new AmtConv2d<T>(_deepest, rank, 1);
        _cubeWidth = new AmtConv2d<T>(_deepest, rank, 1);
    }

    private Tensor<T>[] Pyramid(Tensor<T> x)
    {
        var levels = new Tensor<T>[4];
        for (int i = 0; i < 4; i++) levels[i] = x = _pyramid[i].Forward(x);
        return levels;
    }

    private Tensor<T> HalveFlow(Tensor<T> flow)
        => Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flow, flow.Shape[2] / 2, flow.Shape[3] / 2), NumOps.FromDouble(0.5));

    // The mean over the rank of the outer products of sigmoid channel, height and width attentions, times the feature.
    private Tensor<T> Cube(Tensor<T> s)
    {
        int n = s.Shape[0], h = s.Shape[2], w = s.Shape[3];
        var full = new[] { n, _rank, _deepest, h, w };
        var channel = Engine.Sigmoid(_cubeChannel.Forward(Engine.ReduceMean(s, new[] { 2, 3 }, true)));
        var height = Engine.Sigmoid(_cubeHeight.Forward(Engine.ReduceMean(s, new[] { 3 }, true)));
        var width = Engine.Sigmoid(_cubeWidth.Forward(Engine.ReduceMean(s, new[] { 2 }, true)));
        var product = Engine.TensorMultiply(Engine.TensorMultiply(
            Engine.TensorBroadcastTo(Engine.Reshape(channel, new[] { n, _rank, _deepest, 1, 1 }), full),
            Engine.TensorBroadcastTo(Engine.Reshape(height, new[] { n, _rank, 1, h, 1 }), full)),
            Engine.TensorBroadcastTo(Engine.Reshape(width, new[] { n, _rank, 1, 1, w }), full));
        return Engine.TensorMultiply(s, Engine.ReduceMean(product, new[] { 1 }, false));
    }

    private Tensor<T> Cat(params Tensor<T>[] parts) => Engine.TensorConcatenate(parts, 1);

    /// <summary>
    /// The refined flows [N, 2·branch, H, W] each way and the photometric weights [N, 1, H, W] in [0.1, 0.9], from the
    /// coarse flows (at 1/ratio size) and the standardized frames.
    /// </summary>
    public (Tensor<T> Forward, Tensor<T> Backward, Tensor<T> Weight0, Tensor<T> Weight1) Forward(Tensor<T> flow0, Tensor<T> flow1,
        Tensor<T> im0, Tensor<T> im1, int ratio)
    {
        int h = im0.Shape[2], w = im0.Shape[3];
        flow0 = Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flow0, h, w), NumOps.FromDouble(ratio));
        flow1 = Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flow1, h, w), NumOps.FromDouble(ratio));
        var c0 = Pyramid(im0);
        var c1 = Pyramid(im1);
        var f0 = flow0;
        var f1 = flow1;
        var s0 = new Tensor<T>[4];
        var s1 = new Tensor<T>[4];
        s0[0] = _down[0].Forward(Cat(f0, im0, SoftSplatOps.Backwarp(Engine, im1, f0)));
        s1[0] = _down[0].Forward(Cat(f1, im1, SoftSplatOps.Backwarp(Engine, im0, f1)));
        for (int level = 1; level < 4; level++)
        {
            f0 = HalveFlow(f0);
            f1 = HalveFlow(f1);
            var warped0 = SoftSplatOps.Backwarp(Engine, Cat(s0[level - 1], c0[level - 1]), f1);
            var warped1 = SoftSplatOps.Backwarp(Engine, Cat(s1[level - 1], c1[level - 1]), f0);
            s0[level] = _down[level].Forward(Cat(s0[level - 1], c0[level - 1], warped1));
            s1[level] = _down[level].Forward(Cat(s1[level - 1], c1[level - 1], warped0));
        }
        s0[3] = Cube(s0[3]);
        s1[3] = Cube(s1[3]);
        f0 = HalveFlow(f0);
        f1 = HalveFlow(f1);
        var deep0 = SoftSplatOps.Backwarp(Engine, Cat(s0[3], c0[3]), f1);
        var deep1 = SoftSplatOps.Backwarp(Engine, Cat(s1[3], c1[3]), f0);
        var x0 = _up[0].Forward(Cat(s0[3], c0[3], deep1));
        var x1 = _up[0].Forward(Cat(s1[3], c1[3], deep0));
        for (int level = 1; level < 4; level++)
        {
            x0 = _up[level].Forward(Cat(s0[3 - level], x0));
            x1 = _up[level].Forward(Cat(s1[3 - level], x1));
        }
        Tensor<T> PhotoWeight(Tensor<T> x)
            => Engine.TensorAddScalar(Engine.TensorMultiplyScalar(Engine.Sigmoid(_weight.Forward(x)), NumOps.FromDouble(0.8)), NumOps.FromDouble(0.1));
        var forward = Engine.TensorAdd(Engine.TensorTile(flow0, new[] { 1, _branch, 1, 1 }), _flow.Forward(x0));
        var backward = Engine.TensorAdd(Engine.TensorTile(flow1, new[] { 1, _branch, 1, 1 }), _flow.Forward(x1));
        return (forward, backward, PhotoWeight(x0), PhotoWeight(x1));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => _pyramid.Cast<LazyWeightModule<T>>().Concat(_down).Concat(_up)
            .Concat(new LazyWeightModule<T>[] { _flow, _weight, _cubeChannel, _cubeHeight, _cubeWidth });
}

/// <summary>m2m.M2M_PWC: the PWC-Net flows at 1/ratio size, the Motion Refinement Network and many-to-many splatting.</summary>
internal sealed class M2mNetwork<T> : LazyWeightModule<T>
{
    private readonly int _branch;
    private readonly M2mPwc<T> _flow;
    private readonly M2mRefine<T> _refine;
    private readonly LazyWeight<T> _alpha;

    public M2mNetwork(int c, int branch, int rank, int flowChannels, IReadOnlyList<int> flowHidden)
    {
        _branch = branch;
        _flow = new M2mPwc<T>(flowChannels, flowHidden);
        _refine = new M2mRefine<T>(c, branch, rank);
        _alpha = LazyWeight<T>.Constant(new[] { 1, 1, 1, 1 }, 10.0);
    }

    /// <summary>The flow estimator's weights, which the release trains at a learning rate of zero.</summary>
    internal IEnumerable<Tensor<T>> FlowWeights => _flow.GetParameterStateChunks().Select(chunk => chunk.Tensor);

    /// <summary>The refined flows and photometric weights the splatting reads, from frames [N, 3, H, W] in [0, 1].</summary>
    internal (Tensor<T> Forward, Tensor<T> Backward, Tensor<T> Weight0, Tensor<T> Weight1) Motion(Tensor<T> im0, Tensor<T> im1, int ratio)
    {
        var (a, b, _, _) = Standardize(im0, im1);
        return MotionOf(a, b, ratio);
    }

    private (Tensor<T> Forward, Tensor<T> Backward, Tensor<T> Weight0, Tensor<T> Weight1) MotionOf(Tensor<T> a, Tensor<T> b, int ratio)
    {
        int h = a.Shape[2], w = a.Shape[3];
        var (forward, backward) = _flow.Forward(AmtOps.Resize(Engine, a, 2 * h / ratio, 2 * w / ratio), AmtOps.Resize(Engine, b, 2 * h / ratio, 2 * w / ratio));
        return _refine.Forward(forward, backward, a, b, ratio);
    }

    // Both frames minus the joint mean over the standard deviation plus 1e-7, computed without a gradient.
    private (Tensor<T> One, Tensor<T> Two, Tensor<T> Mean, Tensor<T> Scale) Standardize(Tensor<T> one, Tensor<T> two)
    {
        var shape = one.Shape.ToArray();
        var (mean, std) = SoftSplatNetwork<T>.Statistics(one, two);
        var meanFull = Engine.TensorBroadcastTo(mean, shape);
        var scale = Engine.TensorBroadcastTo(Engine.TensorAddScalar(std, NumOps.FromDouble(1e-7)), shape);
        return (Engine.TensorDivide(Engine.TensorSubtract(one, meanFull), scale), Engine.TensorDivide(Engine.TensorSubtract(two, meanFull), scale), meanFull, scale);
    }

    /// <summary>The frame at each time in <paramref name="times"/> from frames [N, 3, H, W] in [0, 1] whose sides are
    /// multiples of 16·ratio.</summary>
    public Tensor<T>[] Forward(Tensor<T> im0, Tensor<T> im1, IReadOnlyList<double> times, int ratio)
    {
        var (a, b, mean, scale) = Standardize(im0, im1);
        var (forward, backward, weight0, weight1) = MotionOf(a, b, ratio);
        var flows0 = Enumerable.Range(0, _branch).Select(k => Engine.TensorNarrow(forward, 1, 2 * k, 2).Contiguous()).ToArray();
        var flows1 = Enumerable.Range(0, _branch).Select(k => Engine.TensorNarrow(backward, 1, 2 * k, 2).Contiguous()).ToArray();
        var alpha = _alpha.Value;
        // (1 − weight·mean_c|frame − the other frame warped back (detached)|), clipped below at 0.001 and squared, times α.
        Tensor<T> Metric(Tensor<T> frame, Tensor<T> other, Tensor<T> flow, Tensor<T> weight)
        {
            var error = Engine.ReduceMean(Engine.TensorAbs(Engine.TensorSubtract(frame, Engine.StopGradient(SoftSplatOps.Backwarp(Engine, other, flow)))), new[] { 1 }, true);
            var photo = Engine.TensorClampMin(Engine.ScalarMinusTensor(NumOps.One, Engine.TensorMultiply(weight, error)), NumOps.FromDouble(0.001));
            return Engine.TensorMultiply(Engine.TensorBroadcastTo(alpha, photo.Shape.ToArray()), Engine.TensorSquare(photo));
        }
        var metrics0 = flows0.Select(f => Metric(a, b, f, weight0)).ToArray();
        var metrics1 = flows1.Select(f => Metric(b, a, f, weight1)).ToArray();
        var images0 = Enumerable.Repeat(a, _branch).ToArray();
        var images1 = Enumerable.Repeat(b, _branch).ToArray();
        return times.Select(t =>
        {
            var scaled0 = flows0.Select(f => Engine.TensorMultiplyScalar(f, NumOps.FromDouble(t))).ToArray();
            var scaled1 = flows1.Select(f => Engine.TensorMultiplyScalar(f, NumOps.FromDouble(1.0 - t))).ToArray();
            var (output, holes) = M2mOps.ForwardWarpFrames(Engine, images0, scaled0, 1.0 - t, images1, scaled1, t, metrics0, metrics1);
            // Holes no pixel reached take the time-weighted blend of the two frames.
            var blend = Engine.TensorAdd(Engine.TensorMultiplyScalar(a, NumOps.FromDouble(1.0 - t)), Engine.TensorMultiplyScalar(b, NumOps.FromDouble(t)));
            var filled = Engine.TensorAdd(output, Engine.TensorMultiply(Engine.TensorBroadcastTo(holes, blend.Shape.ToArray()), blend));
            return Engine.TensorAdd(Engine.TensorMultiply(filled, scale), mean);
        }).ToArray();
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _alpha;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _flow, _refine };
}
