using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>An nn.Sequential of convolutions, PReLUs, LeakyReLUs and pixel shuffles.</summary>
internal sealed class BimStack<T> : LazyWeightModule<T>
{
    private readonly List<LazyWeightModule<T>> _modules = new();
    private readonly List<Func<Tensor<T>, Tensor<T>>> _steps = new();

    public BimStack<T> Conv(int inChannels, int outChannels, int kernel, int stride = 1, int padding = -1)
    {
        var conv = new AmtConv2d<T>(inChannels, outChannels, kernel, stride, padding);
        _modules.Add(conv);
        _steps.Add(conv.Forward);
        return this;
    }

    public BimStack<T> PRelu(int channels)
    {
        var prelu = new AmtPRelu<T>(channels);
        _modules.Add(prelu);
        _steps.Add(prelu.Forward);
        return this;
    }

    public BimStack<T> LeakyRelu(double slope)
    {
        _steps.Add(x => Engine.LeakyReLU(x, MathHelper.GetNumericOperations<T>().FromDouble(slope)));
        return this;
    }

    public BimStack<T> PixelShuffle(int factor)
    {
        _steps.Add(x => Engine.PixelShuffle(x, factor));
        return this;
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        foreach (var step in _steps) x = step(x);
        return x;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _modules;
}

/// <summary>bimfn.BiMMConv: layer-normed 1x1 residual blocks on the motion feature, ratio and angle embeddings, fused
/// as FV + FV·r·φ.</summary>
internal sealed class BimMConv<T> : LazyWeightModule<T>
{
    private readonly BimLayerNorm<T> _normFv, _normR, _normPhi;
    private readonly BimResBlock<T> _blockFv, _blockR, _blockPhi;

    public BimMConv(int channels)
    {
        _normFv = new BimLayerNorm<T>(channels);
        _normR = new BimLayerNorm<T>(channels);
        _normPhi = new BimLayerNorm<T>(channels);
        _blockFv = new BimResBlock<T>(channels, 1);
        _blockR = new BimResBlock<T>(channels, 1);
        _blockPhi = new BimResBlock<T>(channels, 1);
    }

    public Tensor<T> Forward(Tensor<T> fv, Tensor<T> r, Tensor<T> phi)
    {
        var fvOut = _blockFv.Forward(_normFv.Forward(fv));
        var rOut = _blockR.Forward(_normR.Forward(r));
        var phiOut = _blockPhi.Forward(_normPhi.Forward(phi));
        return Engine.TensorAdd(fvOut, Engine.TensorMultiply(Engine.TensorMultiply(fvOut, rOut), phiOut));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _normFv, _blockFv, _normR, _blockR, _normPhi, _blockPhi };
}

/// <summary>
/// bimfn.BiMFN: from both features warped by the previous flows, their 9x9 cost volumes, the flows and occlusion, a
/// motion feature modulated by the bidirectional motion field (ratio r, angle φ) predicts the residual of both flows.
/// </summary>
internal sealed class BimFlowNetwork<T> : LazyWeightModule<T>
{
    internal const int CostVolumeKernel = 9;

    private readonly BimStack<T> _convFlow, _convOcc, _convCorr, _conv0, _conv1, _conv2, _conv3, _dem, _aem, _out;
    private readonly BimMConv<T> _mconv;

    public BimFlowNetwork(int c)
    {
        _convFlow = new BimStack<T>().Conv(2, c * 2, 7).PRelu(c * 2).Conv(c * 2, c * 2, 3).PRelu(c * 2);
        _convOcc = new BimStack<T>().Conv(1, c * 2, 7).PRelu(c * 2).Conv(c * 2, c * 2, 3).PRelu(c * 2);
        _convCorr = new BimStack<T>().LeakyRelu(0.1).Conv(CostVolumeKernel * CostVolumeKernel, c * 2, 1).PRelu(c * 2).Conv(c * 2, c * 2, 3).PRelu(c * 2);
        _conv0 = new BimStack<T>().Conv(c * 18, c * 12, 1).PRelu(c * 12).Conv(c * 12, c * 12, 3);
        _conv1 = new BimStack<T>().Conv(c * 12, c * 8, 1).PRelu(c * 8).Conv(c * 8, c * 8, 3);
        _conv2 = new BimStack<T>().Conv(c * 8, c * 6, 1).PRelu(c * 6).Conv(c * 6, c * 6, 3);
        _conv3 = new BimStack<T>().Conv(c * 6, c * 6, 1).PRelu(c * 6).Conv(c * 6, c * 6, 3);
        _dem = new BimStack<T>().Conv(1, c * 4, 1).PRelu(c * 4).Conv(c * 4, c * 6, 1);
        _aem = new BimStack<T>().Conv(2, c * 4, 1).PRelu(c * 4).Conv(c * 4, c * 6, 1);
        _mconv = new BimMConv<T>(c * 6);
        _out = new BimStack<T>().Conv(c * 6, c * 4, 1).PRelu(c * 4).Conv(c * 4, 4, 1);
    }

    /// <summary>The refined flows [N, 4, h, w] (t→0 then t→1) and their residual over <paramref name="lastFlow"/>.</summary>
    public (Tensor<T> Flow, Tensor<T> Residual) Forward(Tensor<T> feat0, Tensor<T> feat1, Tensor<T> r, Tensor<T> phi,
        Tensor<T> lastFlow, Tensor<T> lastOcc)
    {
        var flow0 = Engine.TensorNarrow(lastFlow, 1, 0, 2);
        var flow1 = Engine.TensorNarrow(lastFlow, 1, 2, 2);
        var warped0 = BimOps.Warp(Engine, feat0, flow0);
        var warped1 = BimOps.Warp(Engine, feat1, flow1);
        var corr0 = _convCorr.Forward(BimOps.CostVolume(Engine, warped0, warped1, CostVolumeKernel));
        var corr1 = _convCorr.Forward(BimOps.CostVolume(Engine, warped1, warped0, CostVolumeKernel));
        var input = Engine.TensorConcatenate(new[]
        {
            corr0, corr1, warped0, warped1, _convFlow.Forward(flow0), _convFlow.Forward(flow1), _convOcc.Forward(lastOcc),
        }, 1);
        var fv = _conv3.Forward(_conv2.Forward(_conv1.Forward(_conv0.Forward(input))));
        var residual = _out.Forward(_mconv.Forward(fv, _dem.Forward(r), _aem.Forward(phi)));
        return (Engine.TensorAdd(residual, lastFlow), residual);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _convFlow, _convOcc, _convCorr, _conv0, _conv1, _conv2, _conv3, _dem, _aem, _mconv, _out };
}

/// <summary>
/// caun.CAUN: the context-aware upsampling network. Context features warped by the low-resolution flows predict
/// convex-combination weights that upsample the flows by 2 and 4.
/// </summary>
internal sealed class BimUpsampler<T> : LazyWeightModule<T>
{
    private readonly BimStack<T> _enc0, _enc1, _enc2, _kernelX2, _kernelX4;

    public BimUpsampler(int c)
    {
        _enc0 = new BimStack<T>().Conv(c * 8, c * 4, 3).PRelu(c * 4);
        _enc1 = new BimStack<T>().Conv(c * 5, c * 4, 3).PRelu(c * 4);
        _enc2 = new BimStack<T>().Conv(c * 3, c, 3).PRelu(c);
        _kernelX2 = new BimStack<T>().Conv(c * 4, c * 2, 3).PRelu(c * 2).Conv(c * 2, 18, 3);
        _kernelX4 = new BimStack<T>().Conv(c, c, 3).PRelu(c).Conv(c, 18, 3);
    }

    /// <summary>The flows at 4x, 2x and 1x the input size, and the occlusion logits upsampled by 4.</summary>
    public (Tensor<T>[] Flows, Tensor<T> Occlusion) Forward(Tensor<T> flow, Tensor<T>[] feat0, Tensor<T>[] feat1, Tensor<T> lastOcc)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int h = flow.Shape[2], w = flow.Shape[3];
        var warped = new Tensor<T>[3][];
        for (int i = 0; i < 3; i++)
        {
            int s = 1 << i;
            var flowBi = BimOps.Resize(Engine, Engine.TensorMultiplyScalar(flow, ops.FromDouble(s)), h * s, w * s);
            warped[i] = new[]
            {
                BimOps.Warp(Engine, feat0[2 - i], Engine.TensorNarrow(flowBi, 1, 0, 2)),
                BimOps.Warp(Engine, feat1[2 - i], Engine.TensorNarrow(flowBi, 1, 2, 2)),
            };
        }
        var feature0 = _enc0.Forward(Engine.TensorConcatenate(warped[0], 1));
        var feature1 = _enc1.Forward(Engine.TensorConcatenate(new[] { Engine.PixelShuffle(feature0, 2), warped[1][0], warped[1][1] }, 1));
        var feature2 = _enc2.Forward(Engine.TensorConcatenate(new[] { Engine.PixelShuffle(feature1, 2), warped[2][0], warped[2][1] }, 1));
        var maskX2 = _kernelX2.Forward(feature1);
        var maskX4 = _kernelX4.Forward(feature2);
        Tensor<T> Up(Tensor<T> mask, int part, int factor)
            => BimOps.ConvexUpsample(Engine, Engine.TensorMultiplyScalar(Engine.TensorNarrow(flow, 1, part * 2, 2), ops.FromDouble(factor)),
                Engine.TensorNarrow(mask, 1, part * 9, 9), factor);
        var upX2 = Engine.TensorConcatenate(new[] { Up(maskX2, 0, 2), Up(maskX2, 1, 2) }, 1);
        var upX4 = Engine.TensorConcatenate(new[] { Up(maskX4, 0, 4), Up(maskX4, 1, 4) }, 1);
        return (new[] { upX4, upX2, flow }, BimOps.Resize(Engine, lastOcc, lastOcc.Shape[2] * 4, lastOcc.Shape[3] * 4));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _enc0, _enc1, _enc2, _kernelX2, _kernelX4 };
}

/// <summary>
/// sn.SynthesisNetwork: a U-Net over both warped frames, the occlusion and the warped context pyramid predicts a
/// residual and the occlusion update; the frames are blended by sigmoid(occlusion).
/// </summary>
internal sealed class BimSynthesisNetwork<T> : LazyWeightModule<T>
{
    private readonly BimStack<T> _down1, _down2, _down3, _up1, _up2, _up3, _out;

    public BimSynthesisNetwork(int c)
    {
        _down1 = new BimStack<T>().Conv(7, c, 7).PRelu(c).Conv(c, c * 2, 3).PRelu(c * 2);
        _down2 = new BimStack<T>().Conv(c * 4, c * 2, 2, 2, 0).PRelu(c * 2).Conv(c * 2, c * 2, 3).PRelu(c * 2).Conv(c * 2, c * 2, 3).PRelu(c * 2);
        _down3 = new BimStack<T>().Conv(c * 6, c * 4, 2, 2, 0).PRelu(c * 4).Conv(c * 4, c * 4, 3).PRelu(c * 4).Conv(c * 4, c * 4, 3).PRelu(c * 4);
        _up1 = new BimStack<T>().Conv(c * 12, c * 8, 3).PixelShuffle(2).PRelu(c * 2).Conv(c * 2, c * 2, 3).PRelu(c * 2);
        _up2 = new BimStack<T>().Conv(c * 4, c * 4, 3).PixelShuffle(2).PRelu(c).Conv(c, c, 3).PRelu(c);
        _up3 = new BimStack<T>().Conv(c * 3, c * 2, 3).PRelu(c * 2).Conv(c * 2, c * 2, 3).PRelu(c * 2);
        _out = new BimStack<T>().Conv(c * 2, 4, 3);
    }

    private (Tensor<T>, Tensor<T>) WarpPair(Tensor<T> biFlow, Tensor<T> a, Tensor<T> b)
        => (BimOps.Warp(Engine, a, Engine.TensorNarrow(biFlow, 1, 0, 2)), BimOps.Warp(Engine, b, Engine.TensorNarrow(biFlow, 1, 2, 2)));

    /// <summary>The synthesized frame and the occlusion logits.</summary>
    public (Tensor<T> Image, Tensor<T> Occlusion) Forward(Tensor<T> image0, Tensor<T> image1, Tensor<T>[] context0, Tensor<T>[] context1,
        Tensor<T>[] flows, Tensor<T> occlusion)
    {
        var (warpedImage0, warpedImage1) = WarpPair(flows[0], image0, image1);
        var (c0, c1) = WarpPair(flows[0], context0[0], context1[0]);
        var s0 = _down1.Forward(Engine.TensorConcatenate(new[] { warpedImage0, warpedImage1, occlusion }, 1));
        var s1 = _down2.Forward(Engine.TensorConcatenate(new[] { s0, c0, c1 }, 1));
        (c0, c1) = WarpPair(flows[1], context0[1], context1[1]);
        var s2 = _down3.Forward(Engine.TensorConcatenate(new[] { s1, c0, c1 }, 1));
        (c0, c1) = WarpPair(flows[2], context0[2], context1[2]);
        var x = _up1.Forward(Engine.TensorConcatenate(new[] { s2, c0, c1 }, 1));
        x = _up2.Forward(Engine.TensorConcatenate(new[] { x, s1 }, 1));
        x = _up3.Forward(Engine.TensorConcatenate(new[] { x, s0 }, 1));
        var refine = _out.Forward(x);
        var occ = Engine.TensorAdd(occlusion, Engine.TensorNarrow(refine, 1, 3, 1));
        var mask = Engine.Sigmoid(occ);
        var shape = warpedImage0.Shape.ToArray();
        var maskB = Engine.TensorBroadcastTo(mask, shape);
        var inverse = Engine.TensorBroadcastTo(Engine.ScalarMinusTensor(MathHelper.GetNumericOperations<T>().One, mask), shape);
        var merged = Engine.TensorAdd(Engine.TensorMultiply(warpedImage0, maskB), Engine.TensorMultiply(warpedImage1, inverse));
        return (Engine.TensorAdd(merged, Engine.TensorNarrow(refine, 1, 0, 3)), occ);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _down1, _down2, _down3, _up1, _up2, _up3, _out };
}

/// <summary>One pyramid level's outputs (images on the normalized scale until <see cref="BimVfiNetwork{T}.Run"/>
/// restores them).</summary>
internal sealed class BimLevelOutput<T>
{
    public Tensor<T> Flow { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> FlowResidual { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> Image { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> Occlusion { get; set; } = new Tensor<T>(new[] { 0 });

    /// <summary>The teacher pass (KDVCF), present only when a ground-truth frame is given.</summary>
    public BimTeacherOutput<T>? Teacher { get; set; }
}

/// <summary>The teacher's flows, residuals and frame at one level.</summary>
internal sealed class BimTeacherOutput<T>
{
    public Tensor<T> Flow { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> ResidualT0 { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> ResidualT1 { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> Image { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> ZeroSideFlow0 { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> ZeroSideFlow1 { get; set; } = new Tensor<T>(new[] { 0 });
}

/// <summary>
/// bim_vfi.BiMVFI: motion and context ResNet pyramids, BiMFN, CAUN and the synthesis network run coarse-to-fine over
/// an image pyramid of mean/std-normalized, zero-padded frames.
/// </summary>
internal sealed class BimVfiNetwork<T> : LazyWeightModule<T>
{
    private readonly BimResNetPyramid<T> _motionEncoder, _contextEncoder;
    private readonly BimFlowNetwork<T> _flowNetwork;
    private readonly BimSynthesisNetwork<T> _synthesis;
    private readonly BimUpsampler<T> _upsampler;

    public BimVfiNetwork(int channels)
    {
        _motionEncoder = new BimResNetPyramid<T>(channels);
        _contextEncoder = new BimResNetPyramid<T>(channels);
        _flowNetwork = new BimFlowNetwork<T>(channels);
        _synthesis = new BimSynthesisNetwork<T>(channels);
        _upsampler = new BimUpsampler<T>(channels);
    }

    /// <summary>InputPadder(divisor = 2^(levels + 1)): zeros on the right and bottom.</summary>
    internal static (int Right, int Bottom) Padding(int height, int width, int levels)
    {
        int divisor = 1 << (levels + 1);
        return (((width / divisor + 1) * divisor - width) % divisor, ((height / divisor + 1) * divisor - height) % divisor);
    }

    private Tensor<T> Constant(int[] shape, double value)
    {
        var t = new Tensor<T>(shape);
        t.Fill(MathHelper.GetNumericOperations<T>().FromDouble(value));
        return t;
    }

    // torch.rand angles as (cos, sin) [N, 2, h, w].
    private Tensor<T> RandomAngle(int n, int h, int w, Random random)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var t = new Tensor<T>(new[] { n, 2, h, w });
        for (int b = 0; b < n; b++)
            for (int i = 0; i < h * w; i++)
            {
                double angle = random.NextDouble() * Math.PI * 2;
                t[(b * 2) * h * w + i] = ops.FromDouble(Math.Cos(angle));
                t[(b * 2 + 1) * h * w + i] = ops.FromDouble(Math.Sin(angle));
            }
        return t;
    }

    private Tensor<T> Halve(Tensor<T> x, double scale)
    {
        var resized = BimOps.Resize(Engine, Engine.StopGradient(x), x.Shape[2] / 2, x.Shape[3] / 2);
        return scale == 1.0 ? resized : Engine.TensorMultiplyScalar(resized, MathHelper.GetNumericOperations<T>().FromDouble(scale));
    }

    private BimLevelOutput<T> ForwardLevel(Tensor<T> img0, Tensor<T> img1, Tensor<T>? imgt, Tensor<T> lastFlow, Tensor<T> lastOcc,
        double t, Random? random)
    {
        var feat0 = _motionEncoder.Forward(img0);
        var feat1 = _motionEncoder.Forward(img1);
        var context0 = _contextEncoder.Forward(img0);
        var context1 = _contextEncoder.Forward(img1);
        int n = feat0[2].Shape[0], h = feat0[2].Shape[2], w = feat0[2].Shape[3];
        Tensor<T> r, phi;
        BimTeacherOutput<T>? teacher = null;
        if (imgt is not null)
        {
            var rng = random ?? throw new ArgumentNullException(nameof(random), "The teacher pass samples random angles.");
            var featT = _motionEncoder.Forward(imgt);
            var flowTea = Halve(lastFlow, 0.5);
            var occTea = Halve(lastOcc, 1.0);
            var zeros = new Tensor<T>(new[] { n, 2, h, w });
            var flow0t = Engine.TensorConcatenate(new[] { Engine.TensorNarrow(flowTea, 1, 0, 2), zeros }, 1);
            var flowT1 = Engine.TensorConcatenate(new[] { zeros, Engine.TensorNarrow(flowTea, 1, 2, 2) }, 1);
            var (low0t, res0t) = _flowNetwork.Forward(feat0[2], featT[2], Constant(new[] { n, 1, h, w }, 1.0), RandomAngle(n, h, w, rng), flow0t, occTea);
            var (lowT1, res1t) = _flowNetwork.Forward(featT[2], feat1[2], new Tensor<T>(new[] { n, 1, h, w }), RandomAngle(n, h, w, rng), flowT1, occTea);
            var tea0 = Engine.StopGradient(Engine.TensorNarrow(res0t, 1, 0, 2));
            var tea1 = Engine.StopGradient(Engine.TensorNarrow(res1t, 1, 2, 2));
            Tensor<T> Norm(Tensor<T> f) => Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(f), new[] { 1 }, true));
            var r0 = Norm(tea0);
            var r1 = Norm(tea1);
            Tensor<T> X(Tensor<T> f) => Engine.TensorNarrow(f, 1, 0, 1);
            Tensor<T> Y(Tensor<T> f) => Engine.TensorNarrow(f, 1, 1, 1);
            var product = Engine.TensorMultiply(r0, r1);
            var sin = Engine.TensorDivide(Engine.TensorSubtract(Engine.TensorMultiply(X(tea0), Y(tea1)), Engine.TensorMultiply(Y(tea0), X(tea1))), product);
            var cos = Engine.TensorDivide(Engine.TensorAdd(Engine.TensorMultiply(X(tea0), X(tea1)), Engine.TensorMultiply(Y(tea0), Y(tea1))), product);
            var lowTea = Engine.TensorConcatenate(new[] { Engine.TensorNarrow(low0t, 1, 0, 2), Engine.TensorNarrow(lowT1, 1, 2, 2) }, 1);
            var (flowsTea, upOccTea) = _upsampler.Forward(lowTea, context0, context1, occTea);
            var (imageTea, _) = _synthesis.Forward(img0, img1, context0, context1, flowsTea, upOccTea);
            teacher = new BimTeacherOutput<T>
            {
                Flow = flowsTea[0],
                ResidualT0 = tea0,
                ResidualT1 = tea1,
                Image = imageTea,
                ZeroSideFlow0 = Engine.TensorNarrow(res0t, 1, 2, 2),
                ZeroSideFlow1 = Engine.TensorNarrow(res1t, 1, 0, 2),
            };
            r = Engine.StopGradient(Engine.TensorDivide(r0, Engine.TensorAdd(r1, r0)));
            phi = Engine.StopGradient(Engine.TensorConcatenate(new[] { cos, sin }, 1));
        }
        else
        {
            // The uniform motion field: ratio t, angle π (opposite directions).
            r = Constant(new[] { n, 1, h, w }, t);
            phi = Engine.TensorConcatenate(new[] { Constant(new[] { n, 1, h, w }, Math.Cos(Math.PI)), Constant(new[] { n, 1, h, w }, Math.Sin(Math.PI)) }, 1);
        }

        var flowIn = Halve(lastFlow, 0.5);
        var occIn = Halve(lastOcc, 1.0);
        var (flowLow, flowRes) = _flowNetwork.Forward(feat0[2], feat1[2], r, phi, flowIn, occIn);
        var (flows, upOcc) = _upsampler.Forward(flowLow, context0, context1, occIn);
        var (image, occ) = _synthesis.Forward(img0, img1, context0, context1, flows, upOcc);
        return new BimLevelOutput<T> { Flow = flows[0], FlowResidual = flowRes, Image = image, Occlusion = occ, Teacher = teacher };
    }

    /// <summary>
    /// The levels' outputs, coarsest first, for frames [N, 3, H, W] in [0, 1]; images restored to that scale, at the
    /// padded size. With <paramref name="target"/> the target joins the normalization statistics and the teacher runs.
    /// </summary>
    public List<BimLevelOutput<T>> Run(Tensor<T> frame0, Tensor<T> frame1, double t, int levels, Tensor<T>? target = null, Random? random = null)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var frames = target is null ? new[] { frame0, frame1 } : new[] { frame0, frame1, target };
        var axes = new[] { 1, 2, 3 };
        var means = frames.Select(f => Engine.StopGradient(Engine.ReduceMean(f, axes, true))).ToArray();
        var mean = Engine.TensorMultiplyScalar(means.Aggregate(Engine.TensorAdd), ops.FromDouble(1.0 / frames.Length));
        var spread = frames.Select((f, i) =>
        {
            var centered = Engine.TensorSubtract(f, Engine.TensorBroadcastTo(means[i], f.Shape.ToArray()));
            return Engine.TensorAdd(Engine.ReduceMean(Engine.TensorSquare(centered), axes, true), Engine.TensorSquare(Engine.TensorSubtract(mean, means[i])));
        }).Aggregate(Engine.TensorAdd);
        var std = Engine.StopGradient(Engine.TensorAddScalar(Engine.TensorSqrt(Engine.TensorMultiplyScalar(spread, ops.FromDouble(1.0 / frames.Length))), ops.FromDouble(1e-7)));
        mean = Engine.StopGradient(mean);
        Tensor<T> Normalize(Tensor<T> f)
        {
            var shape = f.Shape.ToArray();
            var x = Engine.StopGradient(Engine.TensorDivide(Engine.TensorSubtract(f, Engine.TensorBroadcastTo(mean, shape)), Engine.TensorBroadcastTo(std, shape)));
            var (right, bottom) = Padding(f.Shape[2], f.Shape[3], levels);
            return right + bottom == 0 ? x : Engine.PadNd(x, new[] { 0, right, 0, bottom }, PadMode.Constant, ops.Zero);
        }
        var img0 = Normalize(frame0);
        var img1 = Normalize(frame1);
        var imgt = target is null ? null : Normalize(target);
        int n = img0.Shape[0], height = img0.Shape[2], width = img0.Shape[3];

        var outputs = new List<BimLevelOutput<T>>();
        Tensor<T> lastFlow = new Tensor<T>(new[] { n, 4, height >> levels, width >> levels });
        Tensor<T> lastOcc = new Tensor<T>(new[] { n, 1, height >> levels, width >> levels });
        for (int level = levels - 1; level >= 0; level--)
        {
            int factor = 1 << level;
            var output = ForwardLevel(BimOps.AntialiasDownsample(Engine, img0, factor), BimOps.AntialiasDownsample(Engine, img1, factor),
                imgt is null ? null : BimOps.AntialiasDownsample(Engine, imgt, factor), lastFlow, lastOcc, t, random);
            lastFlow = output.Flow;
            lastOcc = output.Occlusion;
            Tensor<T> Restore(Tensor<T> image)
            {
                var shape = image.Shape.ToArray();
                return Engine.TensorAdd(Engine.TensorMultiply(image, Engine.TensorBroadcastTo(std, shape)), Engine.TensorBroadcastTo(mean, shape));
            }
            output.Image = Restore(output.Image);
            if (output.Teacher is not null) output.Teacher.Image = Restore(output.Teacher.Image);
            outputs.Add(output);
        }
        return outputs;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _motionEncoder, _contextEncoder, _flowNetwork, _synthesis, _upsampler };
}
