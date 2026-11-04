using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>The release's operators (XVFInet.py, utils.py).</summary>
internal static class XvfiOps
{
    /// <summary>
    /// VFInet.bwarp: bilinear grid_sample (zero padding, align_corners=True), times the release's validity mask, which
    /// keeps a pixel only where at least 0.999 of its bilinear footprint lies inside the image and carries no gradient.
    /// </summary>
    internal static Tensor<T> Backwarp<T>(IEngine engine, Tensor<T> x, Tensor<T> flow)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0], h = x.Shape[2], w = x.Shape[3];
        var flowValues = flow.ToArray();
        var mask = new Tensor<T>(new[] { n, 1, h, w });
        for (int b = 0; b < n; b++)
            for (int y = 0; y < h; y++)
                for (int xx = 0; xx < w; xx++)
                {
                    double sx = xx + ops.ToDouble(flowValues[(b * 2 * h + y) * w + xx]);
                    double sy = y + ops.ToDouble(flowValues[((b * 2 + 1) * h + y) * w + xx]);
                    mask[(b * h + y) * w + xx] = Footprint(sx, w) * Footprint(sy, h) >= 0.999 ? ops.One : ops.Zero;
                }
        return engine.TensorMultiply(BasicVsrBlocks.FlowWarp(engine, x, flow, GridSamplePadding.Zeros), engine.TensorBroadcastTo(mask, x.Shape.ToArray()));
    }

    // The share of a 1-D linear footprint at s that lands on 0..extent-1.
    private static double Footprint(double s, int extent)
    {
        if (double.IsNaN(s) || double.IsInfinity(s)) return 0.0;
        int lo = (int)Math.Floor(s);
        double frac = s - lo;
        return (lo >= 0 && lo < extent ? 1.0 - frac : 0.0) + (lo + 1 >= 0 && lo + 1 < extent ? frac : 0.0);
    }

    /// <summary>
    /// VFInet.z_fwarp: each pixel of <paramref name="image"/> splatted to the four pixels around its position plus
    /// <paramref name="flow"/>, weighted by (z + 1e-5)·exp(−squared distance to the pixel); returns the splatted image and
    /// the splatted weights.
    /// </summary>
    internal static (Tensor<T> Image, Tensor<T> Weight) ZForwardWarp<T>(IEngine engine, Tensor<T> image, Tensor<T> flow, Tensor<T> z)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int c = image.Shape[1];
        var ones = new Tensor<T>(new[] { image.Shape[0], 1, image.Shape[2], image.Shape[3] });
        ones.Fill(ops.One);
        var weight = engine.TensorAddScalar(z, ops.FromDouble(1e-5));
        var splatted = SoftSplatOps.SplatCorners(engine, engine.TensorConcatenate(new[] { image, ones }, 1), flow, (fracX, fracY) =>
        {
            Tensor<T> Gaussian(int dx, int dy)
            {
                var ex = dx == 0 ? fracX : engine.TensorAddScalar(fracX, ops.FromDouble(-1.0));
                var ey = dy == 0 ? fracY : engine.TensorAddScalar(fracY, ops.FromDouble(-1.0));
                var squared = engine.TensorAdd(engine.TensorSquare(ex), engine.TensorSquare(ey));
                return engine.TensorMultiply(weight, engine.TensorExp(engine.TensorNegate(squared)));
            }
            return new[] { Gaussian(0, 0), Gaussian(1, 0), Gaussian(0, 1), Gaussian(1, 1) };
        });
        return (engine.TensorNarrow(splatted, 1, 0, c), engine.TensorNarrow(splatted, 1, c, 1));
    }

    /// <summary>F.interpolate(mode='bicubic', align_corners=False) to 1/<paramref name="factor"/> of the size.</summary>
    internal static Tensor<T> Downscale<T>(IEngine engine, Tensor<T> x, int factor)
        => factor == 1 ? x : engine.Interpolate(x, new[] { x.Shape[2] / factor, x.Shape[3] / factor }, InterpolateMode.Bicubic, false);

    /// <summary>nn.UpsamplingNearest2d(scale_factor=2): every pixel repeated over a 2x2 block.</summary>
    internal static Tensor<T> Nearest2x<T>(IEngine engine, Tensor<T> x)
    {
        int n = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        var spread = engine.TensorBroadcastTo(engine.Reshape(x, new[] { n, c, h, 1, w, 1 }), new[] { n, c, h, 2, w, 2 });
        return engine.Reshape(spread.Contiguous(), new[] { n, c, 2 * h, 2 * w });
    }

    /// <summary>set_smoothness_loss: the first-order flow differences weighted by exp(−λ²·mean squared image difference).</summary>
    internal static Tensor<T> Smoothness<T>(IEngine engine, Tensor<T> flow, Tensor<T> image, double edge)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> Diff(Tensor<T> x, int axis) => engine.TensorSubtract(engine.TensorNarrow(x, axis, 1, x.Shape[axis] - 1), engine.TensorNarrow(x, axis, 0, x.Shape[axis] - 1));
        Tensor<T> Term(int axis)
        {
            var weight = engine.TensorExp(engine.TensorMultiplyScalar(engine.ReduceMean(engine.TensorSquare(Diff(image, axis)), new[] { 1 }, true), ops.FromDouble(-edge * edge)));
            var gradient = engine.TensorAbs(Diff(flow, axis));
            return engine.ReduceMean(engine.TensorMultiply(engine.TensorBroadcastTo(weight, gradient.Shape.ToArray()), gradient), new[] { 0, 1, 2, 3 }, false);
        }
        return engine.TensorMultiplyScalar(engine.TensorAdd(Term(2), Term(3)), ops.FromDouble(0.5));
    }
}

/// <summary>weights_init: xavier_normal_ weights and zero bias, for nn.Conv2d and nn.Conv3d alike.</summary>
internal sealed class XvfiConv<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight, _bias;
    private readonly int[] _stride, _padding;
    private readonly bool _volumetric;

    public XvfiConv(int inChannels, int outChannels, int kernel, int stride, int padding, bool volumetric = false)
    {
        _volumetric = volumetric;
        var shape = volumetric ? new[] { outChannels, inChannels, 1, kernel, kernel } : new[] { outChannels, inChannels, kernel, kernel };
        _stride = volumetric ? new[] { 1, stride, stride } : new[] { stride, stride };
        _padding = volumetric ? new[] { 0, padding, padding } : new[] { padding, padding };
        _weight = EmaInit.Normal<T>(shape, Math.Sqrt(2.0 / ((inChannels + outChannels) * kernel * kernel)));
        _bias = LazyWeight<T>.Constant(new[] { outChannels }, 0.0);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = _volumetric
            ? Engine.Conv3D(x, _weight.Value, _stride, _padding, new[] { 1, 1, 1 })
            : Engine.Conv2D(x, _weight.Value, _stride, _padding, new[] { 1, 1 });
        return MgldOps.ChannelBias(Engine, y, _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>ResBlock2D_3D: two [1, 3, 3] convolutions with a ReLU between and a skip.</summary>
internal sealed class XvfiResBlock<T> : LazyWeightModule<T>
{
    private readonly XvfiConv<T> _conv1, _conv2;

    public XvfiResBlock(int channels)
    {
        _conv1 = new XvfiConv<T>(channels, channels, 3, 1, 1, true);
        _conv2 = new XvfiConv<T>(channels, channels, 3, 1, 1, true);
    }

    public Tensor<T> Forward(Tensor<T> x) => Engine.TensorAdd(x, _conv2.Forward(Engine.ReLU(_conv1.Forward(x))));

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv1, _conv2 };
}

/// <summary>
/// The flow stacks of VFInet (conv_flow_bottom, conv_flow2, conv_flow3): an optional 1x1 head, two 4x4 stride-2
/// convolutions, two nearest 2x upsamplings each followed by a 3x3 convolution, and a 3x3 output convolution.
/// </summary>
internal sealed class XvfiFlowStack<T> : LazyWeightModule<T>
{
    private readonly XvfiConv<T>? _head;
    private readonly XvfiConv<T> _down1, _down2, _up1, _up2, _out;

    public XvfiFlowStack(int inChannels, int nf, int outChannels, bool head)
    {
        if (head)
        {
            _head = new XvfiConv<T>(inChannels, nf, 1, 1, 0);
            inChannels = nf;
        }
        _down1 = new XvfiConv<T>(inChannels, 2 * nf, 4, 2, 1);
        _down2 = new XvfiConv<T>(2 * nf, 4 * nf, 4, 2, 1);
        _up1 = new XvfiConv<T>(4 * nf, 2 * nf, 3, 1, 1);
        _up2 = new XvfiConv<T>(2 * nf, nf, 3, 1, 1);
        _out = new XvfiConv<T>(nf, outChannels, 3, 1, 1);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        if (_head is not null) x = Engine.ReLU(_head.Forward(x));
        x = Engine.ReLU(_down2.Forward(Engine.ReLU(_down1.Forward(x))));
        x = Engine.ReLU(_up1.Forward(XvfiOps.Nearest2x(Engine, x)));
        x = Engine.ReLU(_up2.Forward(XvfiOps.Nearest2x(Engine, x)));
        return _out.Forward(x);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>?[] { _head, _down1, _down2, _up1, _up2, _out };
}

/// <summary>
/// RefineUNet: three 4x4 stride-2 encoders, a 3x3 bottleneck and three nearest-upsampled decoders with skips, to the
/// occlusion logit and the residual frame. (The release also registers conv1 and conv2, which forward never uses; they
/// are omitted.)
/// </summary>
internal sealed class XvfiRefineUNet<T> : LazyWeightModule<T>
{
    private readonly XvfiConv<T> _enc1, _enc2, _enc3, _dec0, _dec1, _dec2, _dec3;

    public XvfiRefineUNet(int inChannels, int nf)
    {
        _enc1 = new XvfiConv<T>(inChannels, nf, 4, 2, 1);
        _enc2 = new XvfiConv<T>(nf, 2 * nf, 4, 2, 1);
        _enc3 = new XvfiConv<T>(2 * nf, 4 * nf, 4, 2, 1);
        _dec0 = new XvfiConv<T>(4 * nf, 4 * nf, 3, 1, 1);
        _dec1 = new XvfiConv<T>(6 * nf, 2 * nf, 3, 1, 1);
        _dec2 = new XvfiConv<T>(3 * nf, nf, 3, 1, 1);
        _dec3 = new XvfiConv<T>(nf, 4, 3, 1, 1);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var enc1 = Engine.ReLU(_enc1.Forward(x));
        var enc2 = Engine.ReLU(_enc2.Forward(enc1));
        var y = Engine.ReLU(_dec0.Forward(Engine.ReLU(_enc3.Forward(enc2))));
        y = Engine.ReLU(_dec1.Forward(Engine.TensorConcatenate(new[] { XvfiOps.Nearest2x(Engine, y), enc2 }, 1)));
        y = Engine.ReLU(_dec2.Forward(Engine.TensorConcatenate(new[] { XvfiOps.Nearest2x(Engine, y), enc1 }, 1)));
        return _dec3.Forward(XvfiOps.Nearest2x(Engine, y));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _enc1, _enc2, _enc3, _dec0, _dec1, _dec2, _dec3 };
}

/// <summary>One scale level's result: the flows (0→1, 1→0), and when synthesized, the frame and the refined flows.</summary>
internal sealed class XvfiLevel<T>
{
    public Tensor<T> Flow { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T>? Output { get; set; }
    public Tensor<T>? RefinedFlow { get; set; }
}

/// <summary>
/// VFInet, shared by every scale level: BiOF-I estimates the bidirectional flows (from the features at the coarsest
/// level, else refining the 2x-upsampled flows of the level below), complementary flow reversal turns them into flows
/// from time t by z-weighted forward warping, a refinement stack corrects those, and BiOF-T synthesizes the frame from
/// the warped frames and a refinement U-Net's occlusion map and residual.
/// </summary>
internal sealed class XvfiVfiNet<T> : LazyWeightModule<T>
{
    private readonly int _scale;
    private readonly XvfiFlowStack<T> _bottom, _flow2, _flow3;
    private readonly XvfiConv<T> _flow1;
    private readonly XvfiRefineUNet<T> _refine;

    public XvfiVfiNet(int nf, int scale)
    {
        _scale = scale;
        _bottom = new XvfiFlowStack<T>(2 * nf, nf, 6, false);
        _flow1 = new XvfiConv<T>(2 * nf, nf, 3, 1, 1);
        _flow2 = new XvfiFlowStack<T>(2 * nf + 4, nf, 6, false);
        _flow3 = new XvfiFlowStack<T>(4 * nf + 4, nf, 4, true);
        _refine = new XvfiRefineUNet<T>(4 * nf / (scale * scale) + 4 * 3 + 4, nf);
    }

    /// <summary>
    /// VFInet.forward at one level: frames [N, 3, H, W] already at the level's size, the frames' features
    /// [N, nf, H/scale, W/scale], the level below's flows (null at the coarsest), time t, and whether to synthesize.
    /// </summary>
    public XvfiLevel<T> Forward(Tensor<T> image0, Tensor<T> image1, Tensor<T> feat0, Tensor<T> feat1, Tensor<T>? previous, double t, bool synthesize)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> Scale(Tensor<T> x, double s) => Engine.TensorMultiplyScalar(x, ops.FromDouble(s));
        Tensor<T> Channels(Tensor<T> x, int start, int count) => Engine.TensorNarrow(x, 1, start, count);
        Tensor<T> raw, flow;
        if (previous is null)
        {
            raw = _bottom.Forward(Engine.TensorConcatenate(new[] { feat0, feat1 }, 1));
            flow = Channels(raw, 0, 4);
        }
        else
        {
            var up = Scale(AmtOps.Resize(Engine, Engine.StopGradient(previous), 2 * previous.Shape[2], 2 * previous.Shape[3]), 2.0);
            var warped1 = XvfiOps.Backwarp(Engine, feat1, Channels(up, 0, 2));
            var warped0 = XvfiOps.Backwarp(Engine, feat0, Channels(up, 2, 2));
            raw = _flow2.Forward(Engine.TensorConcatenate(new[]
            {
                _flow1.Forward(Engine.TensorConcatenate(new[] { feat0, warped1 }, 1)),
                _flow1.Forward(Engine.TensorConcatenate(new[] { feat1, warped0 }, 1)),
                up,
            }, 1));
            flow = Engine.TensorAdd(Channels(raw, 0, 4), up);
        }
        var level = new XvfiLevel<T> { Flow = flow };
        if (!synthesize) return level;

        var flow01 = Channels(flow, 0, 2);
        var flow10 = Channels(flow, 2, 2);
        var (forward, norm0) = XvfiOps.ZForwardWarp(Engine, flow01, Scale(flow01, t), Engine.Sigmoid(Channels(raw, 4, 1)));
        var (backward, norm1) = XvfiOps.ZForwardWarp(Engine, flow10, Scale(flow10, 1 - t), Engine.Sigmoid(Channels(raw, 5, 1)));
        // Complementary flow reversal, Eqs. (1) and (2): divided by the splatted weights where any landed.
        var flowT0 = Engine.TensorAdd(Scale(forward, -(1 - t) * t), Scale(backward, t * t));
        var flowT1 = Engine.TensorAdd(Scale(forward, (1 - t) * (1 - t)), Scale(backward, -t * (1 - t)));
        var norm = Engine.TensorAdd(Scale(norm0, 1 - t), Scale(norm1, t));
        var normValues = norm.ToArray();
        var landed = new Tensor<T>(norm._shape);
        for (int i = 0; i < landed.Length; i++) landed[i] = ops.ToDouble(normValues[i]) > 0 ? ops.One : ops.Zero;
        var divisor = Engine.TensorBroadcastTo(Engine.TensorAdd(norm, Engine.ScalarMinusTensor(ops.One, landed)), flowT0.Shape.ToArray());
        var landedFull = Engine.TensorBroadcastTo(landed, flowT0.Shape.ToArray());
        Tensor<T> Normalize(Tensor<T> f)
            => Engine.TensorAdd(Engine.TensorMultiply(Engine.ScalarMinusTensor(ops.One, landedFull), f), Engine.TensorMultiply(landedFull, Engine.TensorDivide(f, divisor)));
        flowT0 = Normalize(flowT0);
        flowT1 = Normalize(flowT1);

        var refined = Engine.TensorAdd(_flow3.Forward(Engine.TensorConcatenate(new[]
        {
            feat0, XvfiOps.Backwarp(Engine, feat0, flowT0), XvfiOps.Backwarp(Engine, feat1, flowT1), feat1, flowT0, flowT1,
        }, 1)), Engine.TensorConcatenate(new[] { flowT0, flowT1 }, 1));
        flowT0 = Channels(refined, 0, 2);
        flowT1 = Channels(refined, 2, 2);
        var warpedFeat0 = XvfiOps.Backwarp(Engine, feat0, flowT0);
        var warpedFeat1 = XvfiOps.Backwarp(Engine, feat1, flowT1);
        int h = image0.Shape[2], w = image0.Shape[3];
        flowT0 = Scale(AmtOps.Resize(Engine, flowT0, h, w), _scale);
        flowT1 = Scale(AmtOps.Resize(Engine, flowT1, h, w), _scale);
        var warpedImage0 = XvfiOps.Backwarp(Engine, image0, flowT0);
        var warpedImage1 = XvfiOps.Backwarp(Engine, image1, flowT1);
        var shuffled = Engine.PixelShuffle(Engine.TensorConcatenate(new[] { feat0, feat1, warpedFeat0, warpedFeat1 }, 1), _scale);
        var refine = _refine.Forward(Engine.TensorConcatenate(new[] { shuffled, image0, image1, warpedImage0, warpedImage1, flowT0, flowT1 }, 1));
        var occlusion0 = Engine.Sigmoid(Channels(refine, 0, 1));
        var occlusion1 = Engine.ScalarMinusTensor(ops.One, occlusion0);
        var shape = image0.Shape.ToArray();
        var weight0 = Engine.TensorBroadcastTo(Scale(occlusion0, 1 - t), shape);
        var weight1 = Engine.TensorBroadcastTo(Scale(occlusion1, t), shape);
        var blended = Engine.TensorDivide(Engine.TensorAdd(Engine.TensorMultiply(weight0, warpedImage0), Engine.TensorMultiply(weight1, warpedImage1)),
            Engine.TensorAdd(weight0, weight1));
        level.Output = Engine.TensorAdd(blended, Channels(refine, 1, 3));
        level.RefinedFlow = refined;
        return level;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _bottom, _flow1, _flow2, _flow3, _refine };
}

/// <summary>A forward pass: the synthesized frame at every level (finest first) and level 0's refined flows.</summary>
internal sealed class XvfiOutput<T>
{
    public Tensor<T>[] Outputs { get; set; } = Array.Empty<Tensor<T>>();
    public Tensor<T>? RefinedFlow { get; set; }
}

/// <summary>
/// XVFInet: the shared feature extractor (a [1, 3, 3] convolution, one shared stride-2 convolution applied log2(scale)
/// times, a convolution and a residual group), a shared stride-2 convolution down to every scale level, and VFInet from
/// the lowest level up, each frame bicubically downscaled to the level's size.
/// </summary>
internal sealed class XvfiNetwork<T> : LazyWeightModule<T>
{
    private readonly int _scale;
    private readonly XvfiConv<T> _converter, _extractDown, _extractOut, _contextDown;
    private readonly XvfiResBlock<T> _res1, _res2;
    private readonly XvfiVfiNet<T> _vfinet;

    public XvfiNetwork(int nf, int scale)
    {
        _scale = scale;
        _converter = new XvfiConv<T>(3, nf, 3, 1, 1, true);
        _extractDown = new XvfiConv<T>(nf, nf, 3, 2, 1, true);
        _extractOut = new XvfiConv<T>(nf, nf, 3, 1, 1, true);
        _res1 = new XvfiResBlock<T>(nf);
        _res2 = new XvfiResBlock<T>(nf);
        _contextDown = new XvfiConv<T>(nf, nf, 3, 2, 1, true);
        _vfinet = new XvfiVfiNet<T>(nf, scale);
    }

    /// <summary>
    /// XVFInet.forward on frames [N, 3, H, W] in [−1, 1]: with <paramref name="training"/>, the frame at every level from
    /// <paramref name="depth"/> down to 0 and level 0's refined flows; otherwise only the flows below level 0 and the
    /// final frame.
    /// </summary>
    public XvfiOutput<T> Forward(Tensor<T> image0, Tensor<T> image1, double t, int depth, bool training)
    {
        int n = image0.Shape[0], h = image0.Shape[2], w = image0.Shape[3];
        var clip = Engine.TensorConcatenate(new[] { Engine.Reshape(image0, new[] { n, 3, 1, h, w }), Engine.Reshape(image1, new[] { n, 3, 1, h, w }) }, 2);
        var feat = Engine.ReLU(_converter.Forward(clip));
        for (int s = _scale; s > 1; s /= 2) feat = Engine.ReLU(_extractDown.Forward(feat));
        feat = _extractOut.Forward(feat);
        feat = Engine.TensorAdd(_res2.Forward(_res1.Forward(feat)), feat);
        var features = new List<Tensor<T>> { feat };
        for (int level = 1; level <= depth; level++) features.Add(feat = _contextDown.Forward(feat));

        var outputs = new List<Tensor<T>>();
        Tensor<T>? flow = null;
        Tensor<T>? refined = null;
        for (int level = depth; level >= 0; level--)
        {
            int factor = 1 << level;
            var f = features[level];
            var result = _vfinet.Forward(XvfiOps.Downscale(Engine, image0, factor), XvfiOps.Downscale(Engine, image1, factor),
                Frame(f, 0), Frame(f, 1), flow, t, training || level == 0);
            flow = result.Flow;
            if (result.Output is not null) outputs.Add(result.Output);
            if (level == 0) refined = result.RefinedFlow;
        }
        outputs.Reverse();
        return new XvfiOutput<T> { Outputs = outputs.ToArray(), RefinedFlow = refined };
    }

    private Tensor<T> Frame(Tensor<T> clip, int index)
    {
        var s = clip.Shape;
        return Engine.Reshape(Engine.TensorNarrow(clip, 2, index, 1).Contiguous(), new[] { s[0], s[1], s[3], s[4] });
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _converter, _extractDown, _extractOut, _res1, _res2, _contextDown, _vfinet };
}
