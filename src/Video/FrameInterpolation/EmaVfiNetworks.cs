using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// InterFrameAttention: queries from one frame's window attend to the other frame's; the attention-weighted
/// coordinate embedding minus the window's own gives the motion feature.
/// </summary>
internal sealed class EmaInterFrameAttention<T> : LazyWeightModule<T>
{
    private readonly EmaLinear<T> _q, _kv, _corEmbed, _proj, _motionProj;
    private readonly int _heads, _motionDim;

    public EmaInterFrameAttention(int dim, int motionDim, int heads)
    {
        _heads = heads;
        _motionDim = motionDim;
        _q = new EmaLinear<T>(dim, dim);
        _kv = new EmaLinear<T>(dim, dim * 2);
        _corEmbed = new EmaLinear<T>(2, motionDim);
        _proj = new EmaLinear<T>(dim, dim);
        _motionProj = new EmaLinear<T>(motionDim, motionDim);
    }

    // [B, N, heads·d] to [B·heads, N, d].
    private Tensor<T> Heads(Tensor<T> x, int d)
    {
        int b = x.Shape[0], n = x.Shape[1];
        return Engine.Reshape(Engine.TensorPermute(Engine.Reshape(x, new[] { b, n, _heads, d }), new[] { 0, 2, 1, 3 }), new[] { b * _heads, n, d });
    }

    private Tensor<T> Merge(Tensor<T> x, int b)
    {
        int n = x.Shape[1], d = x.Shape[2];
        return Engine.Reshape(Engine.TensorPermute(Engine.Reshape(x, new[] { b, _heads, n, d }), new[] { 0, 2, 1, 3 }), new[] { b, n, _heads * d });
    }

    /// <summary>The appearance update and motion feature for windows [B·nW, N, C] (mask [nW, N, N] or null).</summary>
    public (Tensor<T> Appearance, Tensor<T> Motion) Forward(Tensor<T> x1, Tensor<T> x2, Tensor<T> cor, Tensor<T>? mask)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = x1.Shape[0], n = x1.Shape[1], c = x1.Shape[2], hd = c / _heads;
        var q = Heads(_q.Forward(x1), hd);
        var kv = _kv.Forward(x2);
        var k = Heads(Engine.TensorNarrow(kv, 2, 0, c), hd);
        var v = Heads(Engine.TensorNarrow(kv, 2, c, c), hd);
        var corEmbedFlat = _corEmbed.Forward(cor);
        var corEmbed = Heads(corEmbedFlat, _motionDim / _heads);
        var attn = Engine.TensorMultiplyScalar(Engine.BatchMatMul(q, Engine.TensorPermute(k, new[] { 0, 2, 1 })), ops.FromDouble(1.0 / Math.Sqrt(hd)));
        if (mask is not null)
        {
            int nW = mask.Shape[0];
            var shaped = Engine.Reshape(attn, new[] { b / nW, nW, _heads, n, n });
            shaped = Engine.TensorAdd(shaped, Engine.TensorBroadcastTo(Engine.Reshape(mask, new[] { 1, nW, 1, n, n }), shaped.Shape.ToArray()));
            attn = Engine.Reshape(shaped, new[] { b * _heads, n, n });
        }
        attn = Engine.Softmax(attn, axis: -1);
        var x = _proj.Forward(Merge(Engine.BatchMatMul(attn, v), b));
        var reverse = Merge(Engine.BatchMatMul(attn, corEmbed), b);
        return (x, _motionProj.Forward(Engine.TensorSubtract(reverse, corEmbedFlat)));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _q, _kv, _corEmbed, _proj, _motionProj };
}

/// <summary>MotionFormerBlock: (shifted) window inter-frame attention over both frames, then an MLP with a depthwise
/// convolution.</summary>
internal sealed class EmaMotionFormerBlock<T> : LazyWeightModule<T>
{
    private readonly EmaLayerNorm<T> _norm1, _norm2;
    private readonly EmaInterFrameAttention<T> _attn;
    private readonly EmaLinear<T> _fc1, _fc2;
    private readonly EmaDepthwiseConv<T> _dwconv;
    private readonly int _window, _shift;
    private readonly Dictionary<(int, int), Tensor<T>?> _masks = new();

    public EmaMotionFormerBlock(int dim, int motionDim, int heads, int window, int shift, double mlpRatio)
    {
        _window = window;
        _shift = shift;
        _norm1 = new EmaLayerNorm<T>(dim, 1e-6);
        _attn = new EmaInterFrameAttention<T>(dim, motionDim, heads);
        _norm2 = new EmaLayerNorm<T>(dim, 1e-6);
        int hidden = (int)(dim * mlpRatio);
        _fc1 = new EmaLinear<T>(dim, hidden);
        _dwconv = new EmaDepthwiseConv<T>(hidden);
        _fc2 = new EmaLinear<T>(hidden, dim);
    }

    private Tensor<T>? Mask(int h, int w)
    {
        if (_masks.TryGetValue((h, w), out var cached)) return cached;
        var values = EmaOps.AttentionMask(h, w, _window, _shift);
        Tensor<T>? mask = null;
        if (values is not null)
        {
            var (left, right, top, bottom) = EmaOps.WindowPadding(h, w, _window);
            int nW = ((h + top + bottom) / _window) * ((w + left + right) / _window), n = _window * _window;
            var ops = MathHelper.GetNumericOperations<T>();
            mask = new Tensor<T>(new[] { nW, n, n });
            for (int i = 0; i < values.Length; i++) mask[i] = ops.FromDouble(values[i]);
        }
        _masks[(h, w)] = mask;
        return mask;
    }

    // Channels-last [B, H, W, C] → centre-padded, rolled by −shift, windows [B·nW, N, C].
    private Tensor<T> ToWindows(Tensor<T> x, int h, int w)
    {
        var (left, right, top, bottom) = EmaOps.WindowPadding(h, w, _window);
        var nchw = Engine.TensorPermute(x, new[] { 0, 3, 1, 2 });
        if (left + right + top + bottom > 0)
            nchw = Engine.PadNd(nchw, new[] { left, right, top, bottom }, PadMode.Constant, MathHelper.GetNumericOperations<T>().Zero);
        if (_shift > 0) nchw = Engine.TensorRoll(nchw, new[] { -_shift, -_shift }, new[] { 2, 3 });
        return EmaOps.Partition(Engine, Engine.TensorPermute(nchw, new[] { 0, 2, 3, 1 }), _window);
    }

    // Windows back to tokens [B, H·W, C]: reversed, rolled back, centre padding removed.
    private Tensor<T> FromWindows(Tensor<T> windows, int h, int w)
    {
        var (left, right, top, bottom) = EmaOps.WindowPadding(h, w, _window);
        int hp = h + top + bottom, wp = w + left + right;
        var nchw = Engine.TensorPermute(EmaOps.Reverse(Engine, windows, _window, hp, wp), new[] { 0, 3, 1, 2 });
        if (_shift > 0) nchw = Engine.TensorRoll(nchw, new[] { _shift, _shift }, new[] { 2, 3 });
        if (left + right + top + bottom > 0)
            nchw = Engine.TensorSlice(nchw, new[] { 0, 0, top, left }, new[] { nchw.Shape[0], nchw.Shape[1], h, w });
        return Engine.Reshape(Engine.TensorPermute(nchw, new[] { 0, 2, 3, 1 }), new[] { nchw.Shape[0], h * w, nchw.Shape[1] });
    }

    /// <summary>Tokens [2B, H·W, C] of both frames (frame 0 first) and the coordinate grid [2B, H, W, 2] to the updated
    /// tokens and motion features [2B, H·W, motion].</summary>
    public (Tensor<T> Tokens, Tensor<T> Motion) Forward(Tensor<T> tokens, Tensor<T> cor, int h, int w)
    {
        int b2 = tokens.Shape[0], c = tokens.Shape[2];
        var windows = ToWindows(Engine.Reshape(tokens, new[] { b2, h, w, c }), h, w);
        var corWindows = ToWindows(cor, h, w);
        int nwB = windows.Shape[0];
        var normed = _norm1.Forward(windows);
        var reverse = Engine.TensorConcatenate(new[] { Engine.TensorNarrow(normed, 0, nwB / 2, nwB / 2), Engine.TensorNarrow(normed, 0, 0, nwB / 2) }, 0);
        var (appearance, motion) = _attn.Forward(normed, reverse, corWindows, Mask(h, w));
        // The release adds the attention to the normalized windows, not to the block's input.
        var x = FromWindows(Engine.TensorAdd(normed, appearance), h, w);
        var motionTokens = FromWindows(motion, h, w);
        var hidden = _fc1.Forward(_norm2.Forward(x));
        hidden = Engine.GELU(_dwconv.Forward(hidden, h, w));
        return (Engine.TensorAdd(x, _fc2.Forward(hidden)), motionTokens);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _norm1, _attn, _norm2, _fc1, _dwconv, _fc2 };
}

/// <summary>
/// MotionFormer: three convolution stages (full, 1/2, 1/4 size), a cross-scale patch embedding into the first
/// attention stage at 1/8 and an overlapping one into the second at 1/16, each attention stage yielding motion
/// features.
/// </summary>
internal sealed class EmaMotionFormer<T> : LazyWeightModule<T>
{
    private readonly List<LazyWeightModule<T>> _modules = new();
    private readonly EmaConvPRelu<T>[][] _convStages = new EmaConvPRelu<T>[3][];
    private readonly EmaConvPRelu<T>?[] _downs = new EmaConvPRelu<T>?[3];
    private readonly EmaConv2d<T>[] _crossScale;
    private readonly int[] _crossSource;
    private readonly EmaConv2d<T> _crossProj, _overlap;
    private readonly EmaLayerNorm<T> _crossNorm, _overlapNorm;
    private readonly EmaMotionFormerBlock<T>[][] _blocks = new EmaMotionFormerBlock<T>[2][];
    private readonly EmaLayerNorm<T>[] _stageNorms = new EmaLayerNorm<T>[2];

    public EmaMotionFormer(int f, int window, int[] depths, double mlpRatio)
    {
        var dims = new[] { f, 2 * f, 4 * f, 8 * f, 16 * f };
        for (int i = 0; i < 3; i++)
        {
            if (i > 0)
            {
                var down = new EmaConvPRelu<T>(dims[i - 1], dims[i], 3, 2);
                _downs[i] = down;
                _modules.Add(down);
            }
            _convStages[i] = Enumerable.Range(0, depths[i]).Select(j => new EmaConvPRelu<T>(j == 0 ? (i == 0 ? 3 : dims[i]) : dims[i], dims[i])).ToArray();
            _modules.AddRange(_convStages[i]);
        }

        // CrossScalePatchEmbed(in_dims = [F, 2F, 4F]): 2^i dilated stride-2^(i+1) convolutions from the i-th coarsest map.
        var cross = new List<EmaConv2d<T>>();
        var source = new List<int>();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < (1 << i); j++)
            {
                cross.Add(new EmaConv2d<T>(dims[2 - i], dims[0], 3, 1 << (i + 1), 1 + j, 1 + j));
                source.Add(2 - i);
            }
        _crossScale = cross.ToArray();
        _crossSource = source.ToArray();
        _crossProj = new EmaConv2d<T>(dims[0] * cross.Count, dims[3], 1);
        _crossNorm = new EmaLayerNorm<T>(dims[3], 1e-5);
        _overlap = new EmaConv2d<T>(dims[3], dims[4], 3, 2, 1);
        _overlapNorm = new EmaLayerNorm<T>(dims[4], 1e-5);
        _modules.AddRange(_crossScale);
        _modules.Add(_crossProj);
        _modules.Add(_crossNorm);
        _modules.Add(_overlap);
        _modules.Add(_overlapNorm);
        for (int s = 0; s < 2; s++)
        {
            // config.py: 8F / 32 and 16F / 32 heads, each block contributing dim / depth motion channels.
            int i = s + 3, heads = dims[i] / 32;
            int motionDim = dims[i] / depths[i];
            _blocks[s] = Enumerable.Range(0, depths[i])
                .Select(j => new EmaMotionFormerBlock<T>(dims[i], motionDim, heads, window, j % 2 == 0 ? 0 : window / 2, mlpRatio)).ToArray();
            _stageNorms[s] = new EmaLayerNorm<T>(dims[i], 1e-6);
            _modules.AddRange(_blocks[s]);
            _modules.Add(_stageNorms[s]);
        }
    }

    // [B, C, H, W] to tokens [B, H·W, C].
    private Tensor<T> Tokens(Tensor<T> x)
    {
        int b = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        return Engine.Reshape(Engine.TensorPermute(x, new[] { 0, 2, 3, 1 }), new[] { b, h * w, c });
    }

    private Tensor<T> Map(Tensor<T> tokens, int h, int w)
        => Engine.TensorPermute(Engine.Reshape(tokens, new[] { tokens.Shape[0], h, w, tokens.Shape[2] }), new[] { 0, 3, 1, 2 });

    /// <summary>Both frames' appearance features (five maps) and the two attention stages' motion features, frame 0's
    /// samples first in each batch.</summary>
    public (Tensor<T>[] Appearance, Tensor<T>[] Motion) Forward(Tensor<T> image0, Tensor<T> image1)
    {
        var x = Engine.TensorConcatenate(new[] { image0, image1 }, 0);
        var appearance = new Tensor<T>[5];
        var motion = new Tensor<T>[2];
        for (int i = 0; i < 3; i++)
        {
            if (_downs[i] is { } down) x = down.Forward(x);
            foreach (var conv in _convStages[i]) x = conv.Forward(x);
            appearance[i] = x;
        }
        for (int s = 0; s < 2; s++)
        {
            Tensor<T> embedded;
            if (s == 0)
            {
                var parts = _crossScale.Select((conv, k) => conv.Forward(appearance[_crossSource[k]])).ToArray();
                embedded = _crossProj.Forward(Engine.TensorConcatenate(parts, 1));
            }
            else
            {
                embedded = _overlap.Forward(appearance[3]);
            }
            int h = embedded.Shape[2], w = embedded.Shape[3];
            var tokens = (s == 0 ? _crossNorm : _overlapNorm).Forward(Tokens(embedded));
            var cor = EmaOps.CoordinateGrid<T>(tokens.Shape[0], h, w);
            var motions = new List<Tensor<T>>();
            foreach (var block in _blocks[s])
            {
                var (updated, blockMotion) = block.Forward(tokens, cor, h, w);
                tokens = updated;
                motions.Add(Map(blockMotion, h, w));
            }
            appearance[s + 3] = Map(_stageNorms[s].Forward(tokens), h, w);
            motion[s] = Engine.TensorConcatenate(motions.ToArray(), 1);
        }
        return (appearance, motion);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _modules;
}

/// <summary>flow_estimation.Head: motion features shuffled up by 4 and the images at that size predict a flow pair and
/// a mask, upsampled to quarter size.</summary>
internal sealed class EmaHead<T> : LazyWeightModule<T>
{
    private readonly EmaConvPRelu<T> _conv1, _conv2, _conv3;
    private readonly int _scale;

    public EmaHead(int inPlanes, int scale, int c, int inElse)
    {
        _scale = scale;
        _conv1 = new EmaConvPRelu<T>(inPlanes * 2 / 16 + inElse, c);
        _conv2 = new EmaConvPRelu<T>(c, c);
        _conv3 = new EmaConvPRelu<T>(c, 5);
    }

    public (Tensor<T> Flow, Tensor<T> Mask) Forward(Tensor<T> motionFeature, Tensor<T> x, Tensor<T>? flow)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var feature = Engine.PixelShuffle(Engine.PixelShuffle(motionFeature, 2), 2);
        int h = feature.Shape[2], w = feature.Shape[3];
        if (_scale != 4) x = EmaOps.Resize(Engine, x, h, w);
        if (flow is not null)
        {
            if (_scale != 4) flow = Engine.TensorMultiplyScalar(EmaOps.Resize(Engine, flow, h, w), ops.FromDouble(4.0 / _scale));
            x = Engine.TensorConcatenate(new[] { x, flow }, 1);
        }
        var y = _conv3.Forward(_conv2.Forward(_conv1.Forward(Engine.TensorConcatenate(new[] { feature, x }, 1))));
        Tensor<T> outFlow;
        if (_scale != 4)
        {
            int up = _scale / 4;
            y = EmaOps.Resize(Engine, y, h * up, w * up);
            outFlow = Engine.TensorMultiplyScalar(Engine.TensorNarrow(y, 1, 0, 4), ops.FromDouble(up));
        }
        else
        {
            outFlow = Engine.TensorNarrow(y, 1, 0, 4);
        }
        return (outFlow, Engine.TensorNarrow(y, 1, 4, 1));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv1, _conv2, _conv3 };
}

/// <summary>refine.Unet: frames, warped frames, mask, flow and the warped appearance pyramid to a residual in (−1, 1).</summary>
internal sealed class EmaUnet<T> : LazyWeightModule<T>
{
    private readonly EmaConvPRelu<T>[] _down;
    private readonly AmtDeconv<T>[] _up;
    private readonly AmtPRelu<T>[] _upAct;
    private readonly EmaConv2d<T> _out;

    public EmaUnet(int c)
    {
        EmaConvPRelu<T> Conv(int i, int o, int s) => new(i, o, 3, s, EmaConvInit.FanOutNormal);
        _down = new[]
        {
            Conv(17 + c, 2 * c, 2), Conv(2 * c, 2 * c, 1),
            Conv(4 * c, 4 * c, 2), Conv(4 * c, 4 * c, 1),
            Conv(8 * c, 8 * c, 2), Conv(8 * c, 8 * c, 1),
            Conv(16 * c, 16 * c, 2), Conv(16 * c, 16 * c, 1),
        };
        _up = new[] { new AmtDeconv<T>(32 * c, 8 * c), new AmtDeconv<T>(16 * c, 4 * c), new AmtDeconv<T>(8 * c, 2 * c), new AmtDeconv<T>(4 * c, c) };
        _upAct = new[] { new AmtPRelu<T>(8 * c), new AmtPRelu<T>(4 * c), new AmtPRelu<T>(2 * c), new AmtPRelu<T>(c) };
        _out = new EmaConv2d<T>(c, 3, 3, init: EmaConvInit.FanOutNormal);
    }

    private Tensor<T> Down(int stage, Tensor<T> x) => _down[stage * 2 + 1].Forward(_down[stage * 2].Forward(x));

    private Tensor<T> Up(int stage, Tensor<T> x) => _upAct[stage].Forward(_up[stage].Forward(x));

    public Tensor<T> Forward(Tensor<T> img0, Tensor<T> img1, Tensor<T> warped0, Tensor<T> warped1, Tensor<T> mask, Tensor<T> flow,
        Tensor<T>[] c0, Tensor<T>[] c1)
    {
        var s0 = Down(0, Engine.TensorConcatenate(new[] { img0, img1, warped0, warped1, mask, flow, c0[0], c1[0] }, 1));
        var s1 = Down(1, Engine.TensorConcatenate(new[] { s0, c0[1], c1[1] }, 1));
        var s2 = Down(2, Engine.TensorConcatenate(new[] { s1, c0[2], c1[2] }, 1));
        var s3 = Down(3, Engine.TensorConcatenate(new[] { s2, c0[3], c1[3] }, 1));
        var x = Up(0, Engine.TensorConcatenate(new[] { s3, c0[4], c1[4] }, 1));
        x = Up(1, Engine.TensorConcatenate(new[] { x, s2 }, 1));
        x = Up(2, Engine.TensorConcatenate(new[] { x, s1 }, 1));
        x = Up(3, Engine.TensorConcatenate(new[] { x, s0 }, 1));
        return Engine.Sigmoid(_out.Forward(x));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => _down.Cast<LazyWeightModule<T>>().Concat(_up).Concat(_upAct).Concat(new LazyWeightModule<T>[] { _out });
}

/// <summary>One forward pass's per-stage flows, sigmoid masks and warped blends, and the refined frame.</summary>
internal sealed class EmaVfiOutput<T>
{
    public List<Tensor<T>> Flows { get; } = new();
    public List<Tensor<T>> Masks { get; } = new();
    public List<Tensor<T>> Merged { get; } = new();
    public Tensor<T> Prediction { get; set; } = new Tensor<T>(new[] { 0 });
}

/// <summary>flow_estimation.MultiScaleFlow: MotionFormer features, two flow heads at 1/16 and 1/8 and the U-Net
/// refinement.</summary>
internal sealed class EmaVfiNetwork<T> : LazyWeightModule<T>
{
    private readonly EmaMotionFormer<T> _backbone;
    private readonly EmaHead<T>[] _heads;
    private readonly EmaUnet<T> _unet;

    public EmaVfiNetwork(int f, int window, int[] depths, double mlpRatio)
    {
        _backbone = new EmaMotionFormer<T>(f, window, depths, mlpRatio);
        var dims = new[] { f, 2 * f, 4 * f, 8 * f, 16 * f };
        // Head i reads stage 4 − i: motion (dim / depth per block, depth blocks) plus appearance.
        _heads = new[]
        {
            new EmaHead<T>(dims[4] / depths[4] * depths[4] + dims[4], 16, 4 * f, 6),
            new EmaHead<T>(dims[3] / depths[3] * depths[3] + dims[3], 8, 4 * f, 17),
        };
        _unet = new EmaUnet<T>(2 * f);
    }

    private Tensor<T> Half(Tensor<T> x, bool first)
    {
        int b = x.Shape[0] / 2;
        return Engine.TensorNarrow(x, 0, first ? 0 : b, b);
    }

    /// <summary>MultiScaleFlow.forward on frames [B, 3, H, W] in [0, 1] at time t.</summary>
    public EmaVfiOutput<T> Forward(Tensor<T> img0, Tensor<T> img1, double t)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var (appearance, motion) = _backbone.Forward(img0, img1);
        var output = new EmaVfiOutput<T>();
        Tensor<T>? flow = null, mask = null;
        Tensor<T> warped0 = img0, warped1 = img1;
        for (int i = 0; i < _heads.Length; i++)
        {
            var mf = motion[1 - i];
            var af = appearance[4 - i];
            var feature = Engine.TensorConcatenate(new[]
            {
                Engine.TensorMultiplyScalar(Half(mf, true), ops.FromDouble(t)), Engine.TensorMultiplyScalar(Half(mf, false), ops.FromDouble(1 - t)),
                Half(af, true), Half(af, false),
            }, 1);
            if (flow is null || mask is null)
            {
                (flow, mask) = _heads[i].Forward(feature, Engine.TensorConcatenate(new[] { img0, img1 }, 1), null);
            }
            else
            {
                var (flowD, maskD) = _heads[i].Forward(feature, Engine.TensorConcatenate(new[] { img0, img1, warped0, warped1, mask }, 1), flow);
                flow = Engine.TensorAdd(flow, flowD);
                mask = Engine.TensorAdd(mask, maskD);
            }
            var sigmoid = Engine.Sigmoid(mask);
            output.Masks.Add(sigmoid);
            output.Flows.Add(flow);
            warped0 = EmaOps.Warp(Engine, img0, Engine.TensorNarrow(flow, 1, 0, 2));
            warped1 = EmaOps.Warp(Engine, img1, Engine.TensorNarrow(flow, 1, 2, 2));
            var shape = warped0.Shape.ToArray();
            output.Merged.Add(Engine.TensorAdd(Engine.TensorMultiply(warped0, Engine.TensorBroadcastTo(sigmoid, shape)),
                Engine.TensorMultiply(warped1, Engine.TensorBroadcastTo(Engine.ScalarMinusTensor(ops.One, sigmoid), shape))));
        }

        var finalFlow = flow ?? throw new InvalidOperationException("EMA-VFI has two flow heads.");
        var finalMask = mask ?? throw new InvalidOperationException("EMA-VFI has two flow heads.");
        // warp_features: each appearance map warped by the flow halved per level.
        var c0 = new Tensor<T>[5];
        var c1 = new Tensor<T>[5];
        var levelFlow = finalFlow;
        for (int k = 0; k < 5; k++)
        {
            c0[k] = EmaOps.Warp(Engine, Half(appearance[k], true), Engine.TensorNarrow(levelFlow, 1, 0, 2));
            c1[k] = EmaOps.Warp(Engine, Half(appearance[k], false), Engine.TensorNarrow(levelFlow, 1, 2, 2));
            levelFlow = Engine.TensorMultiplyScalar(EmaOps.Resize(Engine, levelFlow, levelFlow.Shape[2] / 2, levelFlow.Shape[3] / 2), ops.FromDouble(0.5));
        }
        var tmp = _unet.Forward(img0, img1, warped0, warped1, finalMask, finalFlow, c0, c1);
        var residual = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(Engine.TensorNarrow(tmp, 1, 0, 3), ops.FromDouble(2.0)), ops.FromDouble(-1.0));
        output.Prediction = Engine.TensorClamp(Engine.TensorAdd(output.Merged[output.Merged.Count - 1], residual), ops.Zero, ops.One);
        return output;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _backbone, _heads[0], _heads[1], _unet };
}
