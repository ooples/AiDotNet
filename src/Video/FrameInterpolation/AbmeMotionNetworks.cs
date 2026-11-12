using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>A PWC-style dense decoder: each convolution's output is concatenated in front of its input (np.cumsum widths).</summary>
internal sealed class AbmeDenseDecoder<T> : LazyWeightModule<T>
{
    private readonly List<AbmeConv2d<T>> _convs = new();

    public AbmeDenseDecoder(int inChannels, IReadOnlyList<int> widths)
    {
        int channels = inChannels;
        foreach (int width in widths)
        {
            _convs.Add(new AbmeConv2d<T>(channels, width, 3, 1, 1, 1, AbmeInit.KaimingFanIn, leaky: true));
            channels += width;
        }
        OutChannels = channels;
    }

    public int OutChannels { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        foreach (var conv in _convs) x = Engine.TensorConcatenate(new[] { conv.Forward(x), x }, 1);
        return x;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _convs;
}

/// <summary>The dilated context network that refines the finest flow (dc_conv1-7).</summary>
internal sealed class AbmeContextNetwork<T> : LazyWeightModule<T>
{
    private static readonly int[] Dilations = { 1, 2, 4, 8, 16, 1 };
    private readonly List<AbmeConv2d<T>> _convs = new();
    private readonly AbmeConv2d<T> _predict;

    // Widths {d0, d1, d1, d2, d3, d4} of the decoder's (128, 128, 128, 96, 64, 32 at SBME's defaults).
    public AbmeContextNetwork(int inChannels, IReadOnlyList<int> decoder)
    {
        var widths = new[] { decoder[0], decoder[1], decoder[1], decoder[2], decoder[3], decoder[4] };
        int channels = inChannels;
        for (int i = 0; i < widths.Length; i++)
        {
            _convs.Add(new AbmeConv2d<T>(channels, widths[i], 3, 1, Dilations[i], Dilations[i], AbmeInit.KaimingFanIn, leaky: true));
            channels = widths[i];
        }
        _predict = new AbmeConv2d<T>(channels, 2, 3, 1, 1, 1, AbmeInit.KaimingFanIn, leaky: false);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        foreach (var conv in _convs) x = conv.Forward(x);
        return _predict.Forward(x);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _convs.Concat(new[] { _predict });
}

/// <summary>Three convolutions of one encoder level, the first strided by 2 (convNa, convNaa, convNb).</summary>
internal sealed class AbmeEncoderLevel<T> : LazyWeightModule<T>
{
    private readonly AbmeConv2d<T>[] _convs;

    public AbmeEncoderLevel(int inChannels, int outChannels)
        => _convs = new[]
        {
            new AbmeConv2d<T>(inChannels, outChannels, 3, 2, 1, 1, AbmeInit.KaimingFanIn, leaky: true),
            new AbmeConv2d<T>(outChannels, outChannels, 3, 1, 1, 1, AbmeInit.KaimingFanIn, leaky: true),
            new AbmeConv2d<T>(outChannels, outChannels, 3, 1, 1, 1, AbmeInit.KaimingFanIn, leaky: true),
        };

    public Tensor<T> Forward(Tensor<T> x)
    {
        foreach (var conv in _convs) x = conv.Forward(x);
        return x;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _convs;
}

/// <summary>
/// SBMENet (model/SBMNet.py): BMBC's symmetric bilateral motion estimator with a strided first convolution. A six-level
/// encoder shared by both frames; from the coarsest level to the second, a bilateral cost volume around the upsampled
/// motion, the features warped toward the intermediate time, a dense decoder and a flow head; the dilated context
/// network refines the finest flow. Motion is in units of 20 pixels at the input size.
/// </summary>
internal sealed class AbmeSbmeNet<T> : LazyWeightModule<T>
{
    private readonly AbmeEncoderLevel<T>[] _encoder;
    private readonly AbmeDenseDecoder<T>[] _decoders;
    private readonly AbmeConv2d<T>[] _predict;
    private readonly AbmeDeconv<T>[] _upFlow;
    private readonly AbmeDeconv<T>[] _upFeature;
    private readonly AbmeContextNetwork<T> _context;
    private readonly int[] _displacements;

    /// <param name="encoder">The six encoder widths (16, 32, 64, 96, 128, 196).</param>
    /// <param name="decoder">The five dense-decoder widths (128, 128, 96, 64, 32).</param>
    /// <param name="displacements">The bilateral correlation's displacement from level 6 to level 2 (6, 4, 4, 2, 2).</param>
    public AbmeSbmeNet(IReadOnlyList<int> encoder, IReadOnlyList<int> decoder, IReadOnlyList<int> displacements)
    {
        _displacements = displacements.ToArray();
        _encoder = new AbmeEncoderLevel<T>[6];
        for (int l = 0; l < 6; l++) _encoder[l] = new AbmeEncoderLevel<T>(l == 0 ? 3 : encoder[l - 1], encoder[l]);
        _decoders = new AbmeDenseDecoder<T>[5];
        _predict = new AbmeConv2d<T>[5];
        _upFlow = new AbmeDeconv<T>[4];
        _upFeature = new AbmeDeconv<T>[4];
        for (int k = 0; k < 5; k++)
        {
            int md = _displacements[k];
            int od = (2 * md + 1) * (2 * md + 1) + (k == 0 ? 0 : 2 * encoder[5 - k] + 4);
            _decoders[k] = new AbmeDenseDecoder<T>(od, decoder);
            int channels = _decoders[k].OutChannels;
            _predict[k] = new AbmeConv2d<T>(channels, 2, 3, 1, 1, 1, AbmeInit.KaimingFanIn, leaky: false);
            if (k < 4)
            {
                _upFlow[k] = new AbmeDeconv<T>(2, 2);
                _upFeature[k] = new AbmeDeconv<T>(channels, 2);
            }
        }
        _context = new AbmeContextNetwork<T>(_decoders[4].OutChannels, decoder);
    }

    private Tensor<T> Leaky(Tensor<T> x) => Engine.LeakyReLU(x, NumOps.FromDouble(0.1));

    /// <summary>The motion from the intermediate time to the second frame at a quarter of the input size, in units of
    /// 20 pixels; the input is both frames [N, 6, H, W] with H and W multiples of 64.</summary>
    public Tensor<T> Forward(Tensor<T> pair)
    {
        int n = pair.Shape[0];
        // Both frames go through the encoder together; the batch halves are the two frames' features.
        var x = Engine.TensorConcatenate(new[] { Engine.TensorNarrow(pair, 1, 0, 3), Engine.TensorNarrow(pair, 1, 3, 3) }, 0);
        var features = new Tensor<T>[6];
        for (int l = 0; l < 6; l++) features[l] = x = _encoder[l].Forward(x);

        Tensor<T>? upFlow = null, upFeature = null, flow = null, decoded = null;
        for (int k = 0; k < 5; k++)
        {
            var level = features[5 - k];
            var c1 = Engine.TensorNarrow(level, 0, 0, n);
            var c2 = Engine.TensorNarrow(level, 0, n, n);
            Tensor<T> input;
            if (upFlow is null || upFeature is null)
            {
                var zero = new Tensor<T>(new[] { n, 2, level.Shape[2], level.Shape[3] });
                input = Leaky(AbmeOps.BilateralCorrelation(Engine, c1, c2, zero, _displacements[k]));
            }
            else
            {
                // 20 / 2^level: the motion in this level's pixels (0.625 at level 5 to 5 at level 2).
                var motion = Engine.TensorMultiplyScalar(upFlow, NumOps.FromDouble(20.0 / (1 << (6 - k))));
                var warp1 = AbmeOps.Warp(Engine, c1, Engine.TensorNegate(motion));
                var warp2 = AbmeOps.Warp(Engine, c2, motion);
                var correlation = Leaky(AbmeOps.BilateralCorrelation(Engine, c1, c2, motion, _displacements[k]));
                input = Engine.TensorConcatenate(new[] { correlation, warp1, warp2, upFlow, upFeature }, 1);
            }
            decoded = _decoders[k].Forward(input);
            flow = _predict[k].Forward(decoded);
            if (k < 4)
            {
                upFlow = _upFlow[k].Forward(flow);
                upFeature = _upFeature[k].Forward(decoded);
            }
        }
        if (flow is null || decoded is null) throw new InvalidOperationException("SBMENet has no decoder level.");
        return Engine.TensorAdd(flow, _context.Forward(decoded));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => _encoder.Cast<LazyWeightModule<T>>().Concat(_decoders).Concat(_predict).Concat(_upFlow).Concat(_upFeature)
            .Concat(new LazyWeightModule<T>[] { _context });
}

/// <summary>
/// ABMRNet (model/ABMNet.py): the asymmetric bilateral motion refiner. From the anchor frame and one input frame, the
/// symmetric motion V and the reliability map Z at a quarter of the input size, it refines V on two levels with a
/// correlation of the anchor's features (gated by Z) against the frame's warped features, and returns the asymmetric
/// motion at the input size (units of 20 pixels) and the refined reliability.
/// </summary>
internal sealed class AbmeAbmrNet<T> : LazyWeightModule<T>
{
    private readonly AbmeEncoderLevel<T> _level1, _level2;
    private readonly AbmeConv2d<T> _asfm1, _asfm2, _residual;
    private readonly AbmeDenseDecoder<T> _decoder2, _decoder1;
    private readonly AbmeConv2d<T> _flow2, _mask2, _flow1;
    private readonly AbmeDeconv<T> _upFeature2;
    private readonly AbmeContextNetwork<T> _context;
    private readonly int _md;

    /// <param name="encoder">The two encoder widths (16, 32).</param>
    /// <param name="decoder">The five dense-decoder widths (64, 64, 48, 32, 16).</param>
    /// <param name="md">The correlation's displacement (2).</param>
    public AbmeAbmrNet(IReadOnlyList<int> encoder, IReadOnlyList<int> decoder, int md)
    {
        _md = md;
        int nd = (2 * md + 1) * (2 * md + 1), e0 = encoder[0], e1 = encoder[1];
        _level1 = new AbmeEncoderLevel<T>(3, e0);
        _level2 = new AbmeEncoderLevel<T>(e0, e1);
        _asfm1 = new AbmeConv2d<T>(e0, e0, 3, 1, 1, 1, AbmeInit.KaimingFanIn, leaky: true);
        _asfm2 = new AbmeConv2d<T>(e1, e1, 3, 1, 1, 1, AbmeInit.KaimingFanIn, leaky: true);
        _decoder2 = new AbmeDenseDecoder<T>(nd + e1 + 2, decoder);
        _flow2 = new AbmeConv2d<T>(_decoder2.OutChannels, 2, 3, 1, 1, 1, AbmeInit.KaimingFanIn, leaky: false);
        _mask2 = new AbmeConv2d<T>(_decoder2.OutChannels, 1, 3, 1, 1, 1, AbmeInit.KaimingFanIn, leaky: false);
        _upFeature2 = new AbmeDeconv<T>(_decoder2.OutChannels, e0);
        _decoder1 = new AbmeDenseDecoder<T>(nd + e0 + 2 + e0, decoder);
        _flow1 = new AbmeConv2d<T>(_decoder1.OutChannels, 2, 3, 1, 1, 1, AbmeInit.KaimingFanIn, leaky: false);
        _context = new AbmeContextNetwork<T>(_decoder1.OutChannels, decoder);
        _residual = new AbmeConv2d<T>(e0, e0, 3, 1, 1, 1, AbmeInit.KaimingFanIn, leaky: false);
    }

    private Tensor<T> Leaky(Tensor<T> x) => Engine.LeakyReLU(x, NumOps.FromDouble(0.1));

    private static Tensor<T> Broadcast(Tensor<T> map, Tensor<T> like) => Engine.TensorBroadcastTo(map, like.Shape.ToArray());

    /// <param name="pair">The anchor frame and the input frame, [N, 6, H, W].</param>
    /// <param name="motion">The symmetric motion toward the input frame at H/4, W/4.</param>
    /// <param name="reliability">The initial reliability map [N, 1, H/4, W/4].</param>
    public (Tensor<T> Motion, Tensor<T> Reliability) Forward(Tensor<T> pair, Tensor<T> motion, Tensor<T> reliability)
    {
        int n = pair.Shape[0];
        var x = Engine.TensorConcatenate(new[] { Engine.TensorNarrow(pair, 1, 0, 3), Engine.TensorNarrow(pair, 1, 3, 3) }, 0);
        var f1 = _level1.Forward(x);
        var f2 = _level2.Forward(f1);
        var c11 = Engine.TensorNarrow(f1, 0, 0, n);
        var c21 = Engine.TensorNarrow(f1, 0, n, n);
        var c12 = Engine.TensorNarrow(f2, 0, 0, n);
        var c22 = Engine.TensorNarrow(f2, 0, n, n);

        var warp2 = AbmeOps.Warp(Engine, _asfm2.Forward(c22), Engine.TensorMultiplyScalar(motion, NumOps.FromDouble(5.0)));
        var corr2 = Leaky(AbmeOps.Correlation(Engine, Engine.TensorMultiply(c12, Broadcast(reliability, c12)), warp2, _md));
        var d2 = _decoder2.Forward(Engine.TensorConcatenate(new[] { corr2, c12, motion }, 1));
        var v2 = Engine.TensorAdd(motion, _flow2.Forward(d2));
        var z2 = _mask2.Forward(d2);
        var feature1 = Leaky(_upFeature2.Forward(d2));

        var v1 = AbmeOps.Upsample2(Engine, v2);
        var z1 = AbmeOps.Upsample2(Engine, z2);
        var warp1 = AbmeOps.Warp(Engine, _asfm1.Forward(c21), Engine.TensorMultiplyScalar(v1, NumOps.FromDouble(10.0)));
        var gated = Leaky(Engine.TensorAdd(Engine.TensorMultiply(c11, Broadcast(Engine.Sigmoid(z1), c11)), _residual.Forward(feature1)));
        var corr1 = Leaky(AbmeOps.Correlation(Engine, gated, warp1, _md));
        var d1 = _decoder1.Forward(Engine.TensorConcatenate(new[] { corr1, gated, v1, feature1 }, 1));
        v1 = Engine.TensorAdd(v1, _flow1.Forward(d1));
        var flow = Engine.TensorAdd(v1, _context.Forward(d1));
        return (AbmeOps.Resize(Engine, flow, 2 * flow.Shape[2], 2 * flow.Shape[3]), Engine.Sigmoid(z1));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _level1, _level2, _asfm1, _asfm2, _decoder2, _flow2, _mask2, _upFeature2, _decoder1, _flow1, _context, _residual };
}
