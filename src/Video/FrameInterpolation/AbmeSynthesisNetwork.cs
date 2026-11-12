using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>The four block shapes of the synthesis GridNets.</summary>
internal enum AbmeGridBlockKind
{
    /// <summary>conv, PReLU, conv: a row's entry block.</summary>
    First,

    /// <summary>PReLU, conv, PReLU, conv: a lateral block and the last block.</summary>
    Lateral,

    /// <summary>PReLU, stride-2 conv, PReLU, conv.</summary>
    Down,

    /// <summary>Bilinear 2x upsampling (align_corners=True), PReLU, conv, PReLU, conv.</summary>
    Up,
}

/// <summary>One GridNet block of model/SynthesisNet.py (torch-default initialization, biased convolutions).</summary>
internal sealed class AbmeGridBlock<T> : LazyWeightModule<T>
{
    private readonly AbmeGridBlockKind _kind;
    private readonly AbmeConv2d<T> _conv1, _conv2;
    private readonly AbmePRelu<T>? _prelu0;
    private readonly AbmePRelu<T> _prelu1 = new();

    public AbmeGridBlock(AbmeGridBlockKind kind, int inChannels, int outChannels)
    {
        _kind = kind;
        if (kind != AbmeGridBlockKind.First) _prelu0 = new AbmePRelu<T>();
        _conv1 = new AbmeConv2d<T>(inChannels, outChannels, 3, kind == AbmeGridBlockKind.Down ? 2 : 1, 1, 1, AbmeInit.TorchDefault, leaky: false);
        _conv2 = new AbmeConv2d<T>(outChannels, outChannels, 3, 1, 1, 1, AbmeInit.TorchDefault, leaky: false);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        if (_kind == AbmeGridBlockKind.Up)
            x = Engine.Interpolate(x, new[] { 2 * x.Shape[2], 2 * x.Shape[3] }, InterpolateMode.Bilinear, true);
        if (_prelu0 is not null) x = _prelu0.Forward(x);
        return _conv2.Forward(_prelu1.Forward(_conv1.Forward(x)));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>?[] { _prelu0, _conv1, _prelu1, _conv2 };
}

/// <summary>
/// GridNet_Filter and GridNet_Refine: three rows at full, half and quarter size, six columns, with two downsampling
/// columns, two lateral-only columns at the coarsest row and three upsampling columns.
/// </summary>
internal sealed class AbmeGridNet<T> : LazyWeightModule<T>
{
    private readonly AbmeGridBlock<T> _first, _last, _row2Entry, _row3Entry;
    private readonly AbmeGridBlock<T>[] _row1, _row2, _row3;
    private readonly AbmeGridBlock<T>[] _down1, _down2, _up1, _up2;

    /// <param name="row1In">The full-size input's channels.</param>
    /// <param name="row2In">The half-size input's channels.</param>
    /// <param name="row3In">The quarter-size input's channels.</param>
    /// <param name="widths">The three rows' widths (32, 64, 96).</param>
    /// <param name="outChannels">The output channels.</param>
    public AbmeGridNet(int row1In, int row2In, int row3In, IReadOnlyList<int> widths, int outChannels)
    {
        int w1 = widths[0], w2 = widths[1], w3 = widths[2];
        AbmeGridBlock<T> Lateral(int c) => new(AbmeGridBlockKind.Lateral, c, c);
        _first = new AbmeGridBlock<T>(AbmeGridBlockKind.First, row1In, w1);
        _row2Entry = new AbmeGridBlock<T>(AbmeGridBlockKind.First, row2In, w2);
        _row3Entry = new AbmeGridBlock<T>(AbmeGridBlockKind.First, row3In, w3);
        _row1 = Enumerable.Range(0, 5).Select(_ => Lateral(w1)).ToArray();
        _row2 = Enumerable.Range(0, 5).Select(_ => Lateral(w2)).ToArray();
        _row3 = Enumerable.Range(0, 5).Select(_ => Lateral(w3)).ToArray();
        _down1 = Enumerable.Range(0, 3).Select(_ => new AbmeGridBlock<T>(AbmeGridBlockKind.Down, w1, w2)).ToArray();
        _down2 = Enumerable.Range(0, 3).Select(_ => new AbmeGridBlock<T>(AbmeGridBlockKind.Down, w2, w3)).ToArray();
        _up1 = Enumerable.Range(0, 3).Select(_ => new AbmeGridBlock<T>(AbmeGridBlockKind.Up, w2, w1)).ToArray();
        _up2 = Enumerable.Range(0, 3).Select(_ => new AbmeGridBlock<T>(AbmeGridBlockKind.Up, w3, w2)).ToArray();
        _last = new AbmeGridBlock<T>(AbmeGridBlockKind.Lateral, w1, outChannels);
    }

    private static Tensor<T> Sum(params Tensor<T>[] terms) => terms.Skip(1).Aggregate(terms[0], (a, b) => Engine.TensorAdd(a, b));

    public Tensor<T> Forward(Tensor<T> row1Input, Tensor<T> row2Input, Tensor<T> row3Input)
    {
        var v11 = _first.Forward(row1Input);
        var v12 = Sum(_row1[0].Forward(v11), v11);
        var v13 = Sum(_row1[1].Forward(v12), v12);
        var v20 = _row2Entry.Forward(row2Input);
        var v21 = Sum(_down1[0].Forward(v11), v20);
        var v22 = Sum(_down1[1].Forward(v12), _row2[0].Forward(v21), v21);
        var v23 = Sum(_down1[2].Forward(v13), _row2[1].Forward(v22), v22);
        var v30 = _row3Entry.Forward(row3Input);
        var v31 = Sum(_down2[0].Forward(v21), v30);
        var v32 = Sum(_down2[1].Forward(v22), _row3[0].Forward(v31), v31);
        var v33 = Sum(_down2[2].Forward(v23), _row3[1].Forward(v32), v32);
        var v34 = Sum(_row3[2].Forward(v33), v33);
        var v35 = Sum(_row3[3].Forward(v34), v34);
        var v36 = Sum(_row3[4].Forward(v35), v35);
        var v24 = Sum(_up2[0].Forward(v34), _row2[2].Forward(v23), v23);
        var v25 = Sum(_up2[1].Forward(v35), _row2[3].Forward(v24), v24);
        var v26 = Sum(_up2[2].Forward(v36), _row2[4].Forward(v25), v25);
        var v14 = Sum(_up1[0].Forward(v24), _row1[2].Forward(v13), v13);
        var v15 = Sum(_up1[1].Forward(v25), _row1[3].Forward(v14), v14);
        var v16 = Sum(_up1[2].Forward(v26), _row1[4].Forward(v15), v15);
        return _last.Forward(v16);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new[] { _first, _row2Entry, _row3Entry }.Concat(_row1).Concat(_row2).Concat(_row3)
            .Concat(_down1).Concat(_down2).Concat(_up1).Concat(_up2).Concat(new[] { _last });
}

/// <summary>Feature_Pyramid: two bias-free convolutions with PReLU per level, the second and third levels strided.</summary>
internal sealed class AbmeFeaturePyramid<T> : LazyWeightModule<T>
{
    private readonly List<(AbmeConv2d<T> Conv, AbmePRelu<T> Activation)> _layers = new();

    public AbmeFeaturePyramid(IReadOnlyList<int> widths)
    {
        int channels = 3;
        for (int level = 0; level < 3; level++)
        {
            _layers.Add((new AbmeConv2d<T>(channels, widths[level], 3, level == 0 ? 1 : 2, 1, 1, AbmeInit.TorchDefault, leaky: false, bias: false), new AbmePRelu<T>()));
            _layers.Add((new AbmeConv2d<T>(widths[level], widths[level], 3, 1, 1, 1, AbmeInit.TorchDefault, leaky: false, bias: false), new AbmePRelu<T>()));
            channels = widths[level];
        }
    }

    public Tensor<T>[] Forward(Tensor<T> image)
    {
        var levels = new Tensor<T>[3];
        var x = image;
        for (int level = 0; level < 3; level++)
        {
            for (int j = 0; j < 2; j++)
            {
                var (conv, activation) = _layers[2 * level + j];
                x = activation.Forward(conv.Forward(x));
            }
            levels[level] = x;
        }
        return levels;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => _layers.SelectMany(layer => new LazyWeightModule<T>[] { layer.Conv, layer.Activation });
}

/// <summary>
/// SynthesisNet (model/SynthesisNet.py). Both frames and their feature pyramids are warped along the symmetric and both
/// asymmetric motions at three scales; GridNet_Filter predicts 3x3 dynamic filters over the four warped candidates
/// (softmax over the 36 taps), which blend every channel of the full-size candidates; GridNet_Refine adds a residual to
/// the blended frame.
/// </summary>
internal sealed class AbmeSynthesisNet<T> : LazyWeightModule<T>
{
    private const int Taps = 9;
    private readonly AbmeFeaturePyramid<T> _context;
    private readonly AbmeGridNet<T> _filter, _refine;

    /// <param name="widths">The feature pyramid's and the GridNets' widths (32, 64, 96).</param>
    public AbmeSynthesisNet(IReadOnlyList<int> widths)
    {
        _context = new AbmeFeaturePyramid<T>(widths);
        _filter = new AbmeGridNet<T>(4 * (3 + widths[0]), 4 * widths[1], 4 * widths[2], widths, 4 * Taps);
        _refine = new AbmeGridNet<T>(3 + widths[0] + 4 * widths[0], 4 * widths[1], 4 * widths[2], widths, 3);
    }

    // Flow_pyramid and Img_pyramid: bilinear (align_corners=False) halving and quartering, flows scaled with the size.
    private static Tensor<T>[] Pyramid(Tensor<T> x, bool flow)
    {
        var levels = new Tensor<T>[3];
        levels[0] = x;
        for (int i = 1; i < 3; i++)
        {
            int factor = 1 << i;
            var resized = AbmeOps.Resize(Engine, x, x.Shape[2] / factor, x.Shape[3] / factor);
            levels[i] = flow ? Engine.TensorMultiplyScalar(resized, NumOps.FromDouble(1.0 / factor)) : resized;
        }
        return levels;
    }

    // DynFilter applied to every channel: the four candidates' 3x3 neighbourhoods weighted by the 36 filter taps.
    private static Tensor<T> Filter(Tensor<T>[] candidates, Tensor<T> filter)
    {
        int n = candidates[0].Shape[0], c = candidates[0].Shape[1], h = candidates[0].Shape[2], w = candidates[0].Shape[3];
        var identity = new Tensor<T>(new[] { Taps, 1, 3, 3 });
        for (int k = 0; k < Taps; k++) identity[k * Taps + k] = NumOps.One;
        // [N, C, 4, H, W]: one tap block per candidate, in the order DynFilter concatenates them.
        var stacked = Engine.TensorStack(candidates, 2);
        var columns = Engine.Conv2D(Engine.Reshape(stacked, new[] { n * c * 4, 1, h, w }), identity, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
        var taps = Engine.Reshape(columns, new[] { n, c, 4 * Taps, h, w });
        var weights = Engine.TensorBroadcastTo(Engine.Reshape(filter, new[] { n, 1, 4 * Taps, h, w }), taps.Shape.ToArray());
        return Engine.ReduceSum(Engine.TensorMultiply(taps, weights), new[] { 2 }, false);
    }

    private static Tensor<T> Features(Tensor<T> warped) => Engine.TensorNarrow(warped, 1, 3, warped.Shape[1] - 3);

    /// <summary>
    /// The intermediate frame from [N, 12, H, W]: the two frames, the symmetric motion toward the second frame and the
    /// asymmetric motions toward the second and the first frame, in pixels; H and W multiples of 4.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> input)
    {
        var i0 = Engine.TensorNarrow(input, 1, 0, 3);
        var i1 = Engine.TensorNarrow(input, 1, 3, 3);
        var sbm = Pyramid(Engine.TensorNarrow(input, 1, 6, 2), flow: true);
        var abmToSecond = Pyramid(Engine.TensorNarrow(input, 1, 8, 2), flow: true);
        var abmToFirst = Pyramid(Engine.TensorNarrow(input, 1, 10, 2), flow: true);
        var features0 = _context.Forward(i0);
        var features1 = _context.Forward(i1);
        var images0 = Pyramid(i0, flow: false);
        var images1 = Pyramid(i1, flow: false);

        // Per level: frame 0 along −SBM and its ABM, frame 1 along SBM and its ABM.
        var warped = new Tensor<T>[3][];
        for (int i = 0; i < 3; i++)
        {
            var source0 = Engine.TensorConcatenate(new[] { images0[i], features0[i] }, 1);
            var source1 = Engine.TensorConcatenate(new[] { images1[i], features1[i] }, 1);
            warped[i] = new[]
            {
                AbmeOps.Warp(Engine, source0, Engine.TensorNegate(sbm[i])),
                AbmeOps.Warp(Engine, source0, abmToFirst[i]),
                AbmeOps.Warp(Engine, source1, sbm[i]),
                AbmeOps.Warp(Engine, source1, abmToSecond[i]),
            };
        }

        var filter = Engine.Softmax(_filter.Forward(
            Engine.TensorConcatenate(warped[0], 1),
            Engine.TensorConcatenate(warped[1].Select(Features).ToArray(), 1),
            Engine.TensorConcatenate(warped[2].Select(Features).ToArray(), 1)), 1);
        var filtered = Filter(warped[0], filter);
        var residual = _refine.Forward(
            Engine.TensorConcatenate(new[] { filtered }.Concat(warped[0].Select(Features)).ToArray(), 1),
            Engine.TensorConcatenate(warped[1].Select(Features).ToArray(), 1),
            Engine.TensorConcatenate(warped[2].Select(Features).ToArray(), 1));
        return Engine.TensorAdd(Engine.TensorNarrow(filtered, 1, 0, 3), residual);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _context, _filter, _refine };
}
