using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>RAFT's bottleneck (AMT-S) or residual (AMT-L) block with instance normalization (feat_enc.py).</summary>
internal sealed class AmtEncoderBlock<T> : LazyWeightModule<T>
{
    private readonly AmtConv2d<T>[] _convs;
    private readonly AmtConv2d<T>? _downsample;

    public AmtEncoderBlock(int inPlanes, int planes, int stride, bool bottleneck)
    {
        _convs = bottleneck
            ? new[]
            {
                new AmtConv2d<T>(inPlanes, planes / 4, 1, 1, 0, kaimingFanOut: true),
                new AmtConv2d<T>(planes / 4, planes / 4, 3, stride, 1, kaimingFanOut: true),
                new AmtConv2d<T>(planes / 4, planes, 1, 1, 0, kaimingFanOut: true),
            }
            : new[]
            {
                new AmtConv2d<T>(inPlanes, planes, 3, stride, 1, kaimingFanOut: true),
                new AmtConv2d<T>(planes, planes, 3, 1, 1, kaimingFanOut: true),
            };
        if (stride != 1) _downsample = new AmtConv2d<T>(inPlanes, planes, 1, stride, 0, kaimingFanOut: true);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = x;
        foreach (var conv in _convs) y = Engine.ReLU(AmtOps.InstanceNorm(Engine, conv.Forward(y)));
        var skip = _downsample is null ? x : AmtOps.InstanceNorm(Engine, _downsample.Forward(x));
        return Engine.ReLU(Engine.TensorAdd(skip, y));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _convs.Concat(new[] { _downsample });
}

/// <summary>
/// The correlation feature encoder at 1/8 size: SmallEncoder (AMT-S: widths 32, 64, 96, bottleneck blocks) or
/// BasicEncoder (AMT-L: 64, 72, 128, residual blocks), with instance normalization.
/// </summary>
internal sealed class AmtFeatureEncoder<T> : LazyWeightModule<T>
{
    private readonly AmtConv2d<T> _stem, _head;
    private readonly AmtEncoderBlock<T>[] _blocks;

    public AmtFeatureEncoder(IReadOnlyList<int> widths, int outputDim, bool bottleneck)
    {
        _stem = new AmtConv2d<T>(3, widths[0], 7, 2, 3, kaimingFanOut: true);
        var blocks = new List<AmtEncoderBlock<T>>();
        int inPlanes = widths[0];
        for (int i = 0; i < 3; i++)
        {
            blocks.Add(new AmtEncoderBlock<T>(inPlanes, widths[i], i == 0 ? 1 : 2, bottleneck));
            blocks.Add(new AmtEncoderBlock<T>(widths[i], widths[i], 1, bottleneck));
            inPlanes = widths[i];
        }
        _blocks = blocks.ToArray();
        _head = new AmtConv2d<T>(inPlanes, outputDim, 1, 1, 0, kaimingFanOut: true);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        x = Engine.ReLU(AmtOps.InstanceNorm(Engine, _stem.Forward(x)));
        foreach (var block in _blocks) x = block.Forward(x);
        return _head.Forward(x);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _stem }.Concat(_blocks).Concat(new[] { _head });
}

/// <summary>IFRNet's four-level context encoder: per level a stride-2 convrelu (7x7 first in AMT-L) and a 3x3 one.</summary>
internal sealed class AmtContextEncoder<T> : LazyWeightModule<T>
{
    private readonly AmtConvPRelu<T>[] _convs;

    public AmtContextEncoder(IReadOnlyList<int> channels, bool large)
    {
        var convs = new List<AmtConvPRelu<T>>();
        int previous = 3;
        for (int i = 0; i < channels.Count; i++)
        {
            convs.Add(new AmtConvPRelu<T>(previous, channels[i], large && i == 0 ? 7 : 3, 2));
            convs.Add(new AmtConvPRelu<T>(channels[i], channels[i]));
            previous = channels[i];
        }
        _convs = convs.ToArray();
    }

    public Tensor<T>[] Forward(Tensor<T> x)
    {
        var levels = new Tensor<T>[_convs.Length / 2];
        for (int i = 0; i < levels.Length; i++) levels[i] = x = _convs[2 * i + 1].Forward(_convs[2 * i].Forward(x));
        return levels;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _convs;
}

/// <summary>IFRNet's ResBlock: the last <c>side</c> channels get an extra convolution twice, then a residual PReLU.</summary>
internal sealed class AmtResBlock<T> : LazyWeightModule<T>
{
    private readonly AmtConvPRelu<T> _conv1, _conv2, _conv3, _conv4;
    private readonly AmtConv2d<T> _conv5;
    private readonly AmtPRelu<T> _prelu;
    private readonly int _side;

    public AmtResBlock(int channels, int side)
    {
        _side = side;
        _conv1 = new AmtConvPRelu<T>(channels, channels);
        _conv2 = new AmtConvPRelu<T>(side, side);
        _conv3 = new AmtConvPRelu<T>(channels, channels);
        _conv4 = new AmtConvPRelu<T>(side, side);
        _conv5 = new AmtConv2d<T>(channels, channels, 3);
        _prelu = new AmtPRelu<T>(channels);
    }

    private Tensor<T> Side(Tensor<T> x, AmtConvPRelu<T> conv)
    {
        int c = x.Shape[1];
        return Engine.TensorConcatenate(new[] { Engine.TensorNarrow(x, 1, 0, c - _side), conv.Forward(Engine.TensorNarrow(x, 1, c - _side, _side)) }, 1);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = Side(_conv1.Forward(x), _conv2);
        y = _conv5.Forward(Side(_conv3.Forward(y), _conv4));
        return _prelu.Forward(Engine.TensorAdd(x, y));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv1, _conv2, _conv3, _conv4, _conv5, _prelu };
}

/// <summary>A decoder's convblock: convrelu, ResBlock, then a 2x transposed convolution.</summary>
internal sealed class AmtDecoderBlock<T> : LazyWeightModule<T>
{
    private readonly AmtConvPRelu<T> _conv;
    private readonly AmtResBlock<T> _res;
    private readonly AmtDeconv<T> _up;

    public AmtDecoderBlock(int inChannels, int width, int skip, int outChannels)
    {
        _conv = new AmtConvPRelu<T>(inChannels, width);
        _res = new AmtResBlock<T>(width, skip);
        _up = new AmtDeconv<T>(width, outChannels);
    }

    public Tensor<T> Forward(Tensor<T> x) => _up.Forward(_res.Forward(_conv.Forward(x)));

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv, _res, _up };
}

/// <summary>
/// RAFT's update block as AMT uses it (SmallUpdateBlock for AMT-S, BasicUpdateBlock with its second correlation
/// convolution for AMT-L): motion features from the correlation and both flows, a two-convolution "GRU", and heads for
/// the residual feature and the residual pair of flows, run at a lower size when a scale factor is given.
/// </summary>
internal sealed class AmtUpdateBlock<T> : LazyWeightModule<T>
{
    private readonly AmtConv2d<T> _convc1, _convf1, _convf2, _conv, _gru1, _gru2, _feat1, _feat2, _flow1, _flow2;
    private readonly AmtConv2d<T>? _convc2;
    private readonly int _scale;

    public AmtUpdateBlock(int cdim, int hidden, int flowDim, int corrDim, int? corrDim2, int fcDim, int corrPlanes, int scale)
    {
        _scale = scale;
        _convc1 = new AmtConv2d<T>(2 * corrPlanes, corrDim, 1, 1, 0);
        int corrOut = corrDim;
        if (corrDim2 is int second)
        {
            _convc2 = new AmtConv2d<T>(corrDim, second, 3);
            corrOut = second;
        }
        _convf1 = new AmtConv2d<T>(4, flowDim * 2, 7);
        _convf2 = new AmtConv2d<T>(flowDim * 2, flowDim, 3);
        _conv = new AmtConv2d<T>(corrOut + flowDim, fcDim, 3);
        _gru1 = new AmtConv2d<T>(fcDim + 4 + cdim, hidden, 3);
        _gru2 = new AmtConv2d<T>(hidden, hidden, 3);
        _feat1 = new AmtConv2d<T>(hidden, hidden, 3);
        _feat2 = new AmtConv2d<T>(hidden, cdim, 3);
        _flow1 = new AmtConv2d<T>(hidden, hidden, 3);
        _flow2 = new AmtConv2d<T>(hidden, 4, 3);
    }

    private Tensor<T> Leaky(Tensor<T> x) => Engine.LeakyReLU(x, NumOps.FromDouble(0.1));

    public (Tensor<T> DeltaFeature, Tensor<T> DeltaFlow) Forward(Tensor<T> net, Tensor<T> flow, Tensor<T> corr)
    {
        if (_scale > 1) net = AmtOps.Resize(Engine, net, net.Shape[2] / _scale, net.Shape[3] / _scale);
        var cor = Leaky(_convc1.Forward(corr));
        if (_convc2 is not null) cor = Leaky(_convc2.Forward(cor));
        var flo = Leaky(_convf2.Forward(Leaky(_convf1.Forward(flow))));
        var inp = Leaky(_conv.Forward(Engine.TensorConcatenate(new[] { cor, flo }, 1)));
        var output = _gru2.Forward(Leaky(_gru1.Forward(Engine.TensorConcatenate(new[] { inp, flow, net }, 1))));
        var deltaNet = _feat2.Forward(Leaky(_feat1.Forward(output)));
        var deltaFlow = _flow2.Forward(Leaky(_flow1.Forward(output)));
        if (_scale > 1)
        {
            deltaNet = AmtOps.Resize(Engine, deltaNet, deltaNet.Shape[2] * _scale, deltaNet.Shape[3] * _scale);
            deltaFlow = Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, deltaFlow, deltaFlow.Shape[2] * _scale, deltaFlow.Shape[3] * _scale),
                NumOps.FromDouble(_scale));
        }
        return (deltaNet, deltaFlow);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>?[] { _convc1, _convc2, _convf1, _convf2, _conv, _gru1, _gru2, _feat1, _feat2, _flow1, _flow2 };
}

/// <summary>The hyperparameters that distinguish AMT-S and AMT-L (networks/AMT-S.py, AMT-L.py).</summary>
internal sealed class AmtArchitecture
{
    public int[] Channels { get; set; } = { 20, 32, 44, 56 };
    public int SkipChannels { get; set; } = 20;
    public int NumFlows { get; set; } = 3;
    public int CorrelationRadius { get; set; } = 3;
    public int CorrelationLevels { get; set; } = 4;
    public int[] FeatureWidths { get; set; } = { 32, 64, 96 };
    public int FeatureDim { get; set; } = 84;
    public bool Large { get; set; }
    public int Hidden { get; set; } = 76;
    public int FlowDim { get; set; } = 20;
    public int CorrDim { get; set; } = 64;
    public int? CorrDim2 { get; set; }
    public int FcDim { get; set; } = 68;
    public int CombineKernel { get; set; } = 3;
}

/// <summary>
/// AMT's network (Li et al. 2023): bidirectional all-pairs correlation at 1/8 size, an IFRNet-style coarse-to-fine
/// decoder whose flows are refined by correlation lookups at three levels, multiple flow fields per pixel at full size,
/// and their combination.
/// </summary>
internal sealed class AmtNetwork<T> : LazyWeightModule<T>
{
    private readonly AmtArchitecture _arch;
    private readonly AmtFeatureEncoder<T> _features;
    private readonly AmtContextEncoder<T> _encoder;
    private readonly AmtDecoderBlock<T> _decoder4, _decoder3, _decoder2, _decoder1;
    private readonly AmtUpdateBlock<T> _update4, _update3, _update2;
    private readonly AmtConv2d<T> _combine1, _combine2;
    private readonly AmtPRelu<T> _combinePRelu;

    public AmtNetwork(AmtArchitecture arch)
    {
        _arch = arch;
        var c = arch.Channels;
        int skip = arch.SkipChannels, n = arch.NumFlows;
        _features = new AmtFeatureEncoder<T>(arch.FeatureWidths, arch.FeatureDim, bottleneck: !arch.Large);
        _encoder = new AmtContextEncoder<T>(c, arch.Large);
        _decoder4 = new AmtDecoderBlock<T>(c[3] * 2 + 1, c[3] * 2, skip, c[2] + 4);
        _decoder3 = new AmtDecoderBlock<T>(c[2] * 3 + 4, c[2] * 3, skip, c[1] + 4);
        _decoder2 = new AmtDecoderBlock<T>(c[1] * 3 + 4, c[1] * 3, skip, c[0] + 4);
        _decoder1 = new AmtDecoderBlock<T>(c[0] * 3 + 4, c[0] * 3, skip, 8 * n);
        int planes = arch.CorrelationLevels * (2 * arch.CorrelationRadius + 1) * (2 * arch.CorrelationRadius + 1);
        AmtUpdateBlock<T> Update(int cdim, int scale)
            => new(cdim, arch.Hidden, arch.FlowDim, arch.CorrDim, arch.CorrDim2, arch.FcDim, planes, scale);
        _update4 = Update(c[2], 1);
        _update3 = Update(c[1], 2);
        _update2 = Update(c[0], 4);
        _combine1 = new AmtConv2d<T>(3 * n, 6 * n, arch.CombineKernel);
        _combinePRelu = new AmtPRelu<T>(6 * n);
        _combine2 = new AmtConv2d<T>(6 * n, 3, arch.CombineKernel);
    }

    private Tensor<T> Up2(Tensor<T> flow)
        => Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flow, flow.Shape[2] * 2, flow.Shape[3] * 2), NumOps.FromDouble(2.0));

    private static Tensor<T> Batch(Tensor<T> x, int start, int count) => Engine.TensorNarrow(x, 0, start, count);

    // _corr_scale_lookup: the t→1 and t→0 flows extended linearly to 0→1 and 1→0 at the correlation's size.
    private Tensor<T> CorrelationLookup(Tensor<T>[] forward, Tensor<T>[] transposed, Tensor<T> coord, Tensor<T> flow0, Tensor<T> flow1,
        double t, int downsample)
    {
        if (downsample != 1)
        {
            var inv = NumOps.FromDouble(1.0 / downsample);
            flow0 = Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flow0, flow0.Shape[2] / downsample, flow0.Shape[3] / downsample), inv);
            flow1 = Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flow1, flow1.Shape[2] / downsample, flow1.Shape[3] / downsample), inv);
        }
        var coords0 = Engine.TensorAdd(coord, Engine.TensorMultiplyScalar(flow1, NumOps.FromDouble(1.0 / t)));
        var coords1 = Engine.TensorAdd(coord, Engine.TensorMultiplyScalar(flow0, NumOps.FromDouble(1.0 / (1.0 - t))));
        int r = _arch.CorrelationRadius;
        return Engine.TensorConcatenate(new[] { AmtOps.Lookup(Engine, forward, coords0, r), AmtOps.Lookup(Engine, transposed, coords1, r) }, 1);
    }

    private (Tensor<T> Flow0, Tensor<T> Flow1, Tensor<T> Feature) Refine(AmtUpdateBlock<T> update, Tensor<T> feature, Tensor<T> flow0, Tensor<T> flow1,
        Tensor<T> corr, int downsample)
    {
        var flow = Engine.TensorConcatenate(new[] { flow0, flow1 }, 1);
        if (downsample != 1)
        {
            var inv = NumOps.FromDouble(1.0 / downsample);
            flow = Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flow, flow.Shape[2] / downsample, flow.Shape[3] / downsample), inv);
        }
        var (deltaFeature, deltaFlow) = update.Forward(feature, flow, corr);
        return (Engine.TensorAdd(flow0, Engine.TensorNarrow(deltaFlow, 1, 0, 2)), Engine.TensorAdd(flow1, Engine.TensorNarrow(deltaFlow, 1, 2, 2)),
            Engine.TensorAdd(feature, deltaFeature));
    }

    private (Tensor<T> Flow0, Tensor<T> Flow1, Tensor<T> Feature) Intermediate(AmtDecoderBlock<T> decoder, Tensor<T> feature, Tensor<T> f0, Tensor<T> f1,
        Tensor<T> flow0, Tensor<T> flow1)
    {
        var input = Engine.TensorConcatenate(new[] { feature, AmtOps.Warp(Engine, f0, flow0), AmtOps.Warp(Engine, f1, flow1), flow0, flow1 }, 1);
        var output = decoder.Forward(input);
        return (Engine.TensorAdd(Engine.TensorNarrow(output, 1, 0, 2), Up2(flow0)), Engine.TensorAdd(Engine.TensorNarrow(output, 1, 2, 2), Up2(flow1)),
            Engine.TensorNarrow(output, 1, 4, output.Shape[1] - 4));
    }

    /// <summary>
    /// The frame at time <paramref name="t"/> from frames [N, 3, H, W] in [0, 1], H and W multiples of 16, clamped to
    /// [0, 1].
    /// </summary>
    public Tensor<T> Forward(Tensor<T> image0, Tensor<T> image1, double t)
    {
        int b = image0.Shape[0], h = image0.Shape[2], w = image0.Shape[3], n = _arch.NumFlows;
        // The release subtracts each sample's mean over both frames and adds it back at the end.
        var mean = Engine.ReduceMean(Engine.TensorConcatenate(new[] { image0, image1 }, 2), new[] { 1, 2, 3 }, true);
        var img0 = Engine.TensorSubtract(image0, Engine.TensorBroadcastTo(mean, image0.Shape.ToArray()));
        var img1 = Engine.TensorSubtract(image1, Engine.TensorBroadcastTo(mean, image1.Shape.ToArray()));
        var both = Engine.TensorConcatenate(new[] { img0, img1 }, 0);

        var fmaps = _features.Forward(both);
        var (forward, transposed) = AmtOps.CorrelationPyramids(Engine, Batch(fmaps, 0, b), Batch(fmaps, b, b), _arch.CorrelationLevels);
        var coord = AmtOps.CoordsGrid<T>(b, h / 8, w / 8);
        var levels = _encoder.Forward(both);
        Tensor<T> F0(int i) => Batch(levels[i], 0, b);
        Tensor<T> F1(int i) => Batch(levels[i], b, b);

        var embt = new Tensor<T>(new[] { b, 1, h / 16, w / 16 });
        embt.Fill(NumOps.FromDouble(t));
        var out4 = _decoder4.Forward(Engine.TensorConcatenate(new[] { F0(3), F1(3), embt }, 1));
        var flow0 = Engine.TensorNarrow(out4, 1, 0, 2);
        var flow1 = Engine.TensorNarrow(out4, 1, 2, 2);
        var feature = Engine.TensorNarrow(out4, 1, 4, out4.Shape[1] - 4);
        (flow0, flow1, feature) = Refine(_update4, feature, flow0, flow1, CorrelationLookup(forward, transposed, coord, flow0, flow1, t, 1), 1);

        (flow0, flow1, feature) = Intermediate(_decoder3, feature, F0(2), F1(2), flow0, flow1);
        (flow0, flow1, feature) = Refine(_update3, feature, flow0, flow1, CorrelationLookup(forward, transposed, coord, flow0, flow1, t, 2), 2);

        (flow0, flow1, feature) = Intermediate(_decoder2, feature, F0(1), F1(1), flow0, flow1);
        (flow0, flow1, feature) = Refine(_update2, feature, flow0, flow1, CorrelationLookup(forward, transposed, coord, flow0, flow1, t, 4), 4);

        // MultiFlowDecoder: n flow pairs, n masks and n residuals at full size.
        var out1 = _decoder1.Forward(Engine.TensorConcatenate(new[] { feature, AmtOps.Warp(Engine, F0(0), flow0), AmtOps.Warp(Engine, F1(0), flow1), flow0, flow1 }, 1));
        Tensor<T> Repeat(Tensor<T> flow) => Engine.TensorConcatenate(Enumerable.Repeat(Up2(flow), n).ToArray(), 1);
        var flows0 = Engine.TensorAdd(Engine.TensorNarrow(out1, 1, 0, 2 * n), Repeat(flow0));
        var flows1 = Engine.TensorAdd(Engine.TensorNarrow(out1, 1, 2 * n, 2 * n), Repeat(flow1));
        var mask = Engine.Sigmoid(Engine.TensorNarrow(out1, 1, 4 * n, n));
        var residual = Engine.TensorNarrow(out1, 1, 5 * n, 3 * n);

        // multi_flow_combine: every field warps both frames; the blends are averaged and corrected by comb_block.
        Tensor<T> PerFlow(Tensor<T> x, int channels) => Engine.Reshape(x, new[] { b * n, channels, h, w });
        Tensor<T> Stack(Tensor<T> image) => Engine.Reshape(Engine.TensorStack(Enumerable.Repeat(image, n).ToArray(), 1), new[] { b * n, 3, h, w });
        var warped0 = AmtOps.Warp(Engine, Stack(img0), PerFlow(flows0, 2));
        var warped1 = AmtOps.Warp(Engine, Stack(img1), PerFlow(flows1, 2));
        var m = Engine.TensorBroadcastTo(PerFlow(mask, 1), new[] { b * n, 3, h, w });
        var oneMinus = Engine.TensorAddScalar(Engine.TensorNegate(m), NumOps.One);
        var meanPerFlow = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorStack(Enumerable.Repeat(mean, n).ToArray(), 1), new[] { b * n, 1, 1, 1 }),
            new[] { b * n, 3, h, w });
        var blends = Engine.TensorAdd(Engine.TensorAdd(Engine.TensorAdd(Engine.TensorMultiply(m, warped0), Engine.TensorMultiply(oneMinus, warped1)),
            meanPerFlow), PerFlow(residual, 3));
        var grouped = Engine.Reshape(blends, new[] { b, n, 3, h, w });
        var correction = _combine2.Forward(_combinePRelu.Forward(_combine1.Forward(Engine.Reshape(grouped, new[] { b, 3 * n, h, w }))));
        var prediction = Engine.TensorAdd(Engine.ReduceMean(grouped, new[] { 1 }, false), correction);
        return Engine.TensorClamp(prediction, NumOps.Zero, NumOps.One);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _features, _encoder, _decoder4, _decoder3, _decoder2, _decoder1, _update4, _update3, _update2, _combine1, _combinePRelu, _combine2 };
}
