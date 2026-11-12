using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>The VFIT release's host-side operations: AdaCoF and the shifted-window mask.</summary>
internal static class VfitOps
{
    /// <summary>
    /// cupy_module/adacof.py's FunctionAdaCoF: out(c, i, j) = Σ_{k,l} W_kl(i, j)·P(c, i + k·d + α_kl, j + l·d + β_kl), the
    /// frame P [B, C, H + (K−1)d, W + (K−1)d] sampled bilinearly between the integers A = (int)α and A + 1 (truncated
    /// toward zero, so a negative fraction extrapolates), each index clamped to the frame. The weights and offsets are
    /// [B, K², H, W]; the gradient reaches them as the kernel's backward does, and not the frame.
    /// </summary>
    internal static Tensor<T> AdaCof<T>(IEngine engine, Tensor<T> frame, Tensor<T> weight, Tensor<T> alpha, Tensor<T> beta, int dilation)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = frame.Shape[0], c = frame.Shape[1], hin = frame.Shape[2], win = frame.Shape[3];
        int taps = weight.Shape[1], h = weight.Shape[2], w = weight.Shape[3], size = (int)Math.Round(Math.Sqrt(taps));
        if (size * size != taps || hin != h + (size - 1) * dilation || win != w + (size - 1) * dilation)
            throw new ArgumentException("AdaCoF needs K² weights and a frame padded by (K − 1)·dilation.", nameof(frame));
        var alphaValues = alpha.ToArray();
        var betaValues = beta.ToArray();
        int count = taps * b * h * w;
        var index = new int[4 * count];
        var floorA = new T[count];
        var floorB = new T[count];
        int Clamp(int v, int n) => v < 0 ? 0 : v > n - 1 ? n - 1 : v;
        // Corner order (A, B), (A + 1, B), (A, B + 1), (A + 1, B + 1); each [K², B, H, W] into the rows [B·Hin·Win].
        for (int t = 0; t < taps; t++)
            for (int n = 0; n < b; n++)
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < w; j++)
                    {
                        int source = ((n * taps + t) * h + i) * w + j, target = ((t * b + n) * h + i) * w + j;
                        int a = (int)ops.ToDouble(alphaValues[source]), bb = (int)ops.ToDouble(betaValues[source]);
                        floorA[target] = ops.FromDouble(a);
                        floorB[target] = ops.FromDouble(bb);
                        int y0 = Clamp(i + t / size * dilation + a, hin), y1 = Clamp(i + t / size * dilation + a + 1, hin);
                        int x0 = Clamp(j + t % size * dilation + bb, win), x1 = Clamp(j + t % size * dilation + bb + 1, win);
                        index[target] = (n * hin + y0) * win + x0;
                        index[count + target] = (n * hin + y1) * win + x0;
                        index[2 * count + target] = (n * hin + y0) * win + x1;
                        index[3 * count + target] = (n * hin + y1) * win + x1;
                    }
        var table = engine.Reshape(engine.TensorPermute(frame, new[] { 0, 2, 3, 1 }).Contiguous(), new[] { b * hin * win, c });
        var gathered = engine.TensorGather(table, new Tensor<int>(index, new[] { 4 * count }), axis: 0);
        Tensor<T> Corner(int k) => engine.TensorPermute(engine.Reshape(engine.TensorNarrow(gathered, 0, k * count, count), new[] { taps, b, h, w, c }),
            new[] { 0, 1, 4, 2, 3 }).Contiguous();
        Tensor<T> Tap(Tensor<T> x) => engine.Reshape(engine.TensorPermute(x, new[] { 1, 0, 2, 3 }).Contiguous(), new[] { taps, b, 1, h, w });
        var shape = new[] { taps, b, c, h, w };
        var fa = engine.TensorBroadcastTo(engine.TensorSubtract(Tap(alpha), new Tensor<T>(floorA, new[] { taps, b, 1, h, w })), shape);
        var fb = engine.TensorBroadcastTo(engine.TensorSubtract(Tap(beta), new Tensor<T>(floorB, new[] { taps, b, 1, h, w })), shape);
        var ga = engine.TensorAddScalar(engine.TensorNegate(fa), ops.One);
        var gb = engine.TensorAddScalar(engine.TensorNegate(fb), ops.One);
        var sample = engine.TensorAdd(
            engine.TensorAdd(engine.TensorMultiply(Corner(0), engine.TensorMultiply(ga, gb)), engine.TensorMultiply(Corner(1), engine.TensorMultiply(fa, gb))),
            engine.TensorAdd(engine.TensorMultiply(Corner(2), engine.TensorMultiply(ga, fb)), engine.TensorMultiply(Corner(3), engine.TensorMultiply(fa, fb))));
        var weighted = engine.TensorMultiply(engine.TensorBroadcastTo(Tap(weight), shape), sample);
        return engine.ReduceSum(weighted, new[] { 0 }, false);
    }

    // compute_mask's label of a padded position along one axis: a zero shift labels the whole axis alike.
    private static int Region(int p, int n, int window, int shift)
        => shift == 0 ? 0 : p < n - window ? 0 : p < n - shift ? 1 : 2;

    /// <summary>
    /// compute_mask for a (1, wh, ww) window over a padded (D, Hp, Wp) clip: [D·nH·nW, N, N], −100 between two positions of
    /// a window from different regions.
    /// </summary>
    internal static double[] ShiftMask(int depth, int h, int w, int windowH, int windowW, int shiftH, int shiftW)
    {
        int nh = h / windowH, nw = w / windowW, n = windowH * windowW, windows = nh * nw;
        var plane = new double[windows * n * n];
        for (int win = 0; win < windows; win++)
        {
            int wy = win / nw, wx = win % nw;
            for (int i = 0; i < n; i++)
            {
                int a = Region(wy * windowH + i / windowW, h, windowH, shiftH) * 3 + Region(wx * windowW + i % windowW, w, windowW, shiftW);
                for (int j = 0; j < n; j++)
                {
                    int bb = Region(wy * windowH + j / windowW, h, windowH, shiftH) * 3 + Region(wx * windowW + j % windowW, w, windowW, shiftW);
                    if (a != bb) plane[(win * n + i) * n + j] = -100.0;
                }
            }
        }
        var mask = new double[depth * plane.Length];
        for (int d = 0; d < depth; d++) Array.Copy(plane, 0, mask, d * plane.Length, plane.Length);
        return mask;
    }

    /// <summary>
    /// WindowAttention3D's relative position index for a (wd, wh, ww) window, [N·N]; the attention reads its top-left
    /// [n, n] block when a window shrinks to n positions, as the release slices it.
    /// </summary>
    internal static int[] RelativeIndex(int wd, int wh, int ww)
    {
        int n = wd * wh * ww;
        var index = new int[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                int di = i / (wh * ww), hi = i / ww % wh, wi = i % ww, dj = j / (wh * ww), hj = j / ww % wh, wj = j % ww;
                index[i * n + j] = (di - dj + wd - 1) * (2 * wh - 1) * (2 * ww - 1) + (hi - hj + wh - 1) * (2 * ww - 1) + (wi - wj + ww - 1);
            }
        return index;
    }
}

/// <summary>WindowAttention3D: multi-head self-attention in a window with a learned relative position bias.</summary>
internal sealed class VfitAttention<T> : LazyWeightModule<T>
{
    private readonly int _heads, _full;
    private readonly int[] _index;
    private readonly LazyWeight<T> _table;
    private readonly EmaLinear<T> _qkv, _proj;

    public VfitAttention(int dim, int wd, int wh, int ww, int heads)
    {
        _heads = heads;
        _full = wd * wh * ww;
        _index = VfitOps.RelativeIndex(wd, wh, ww);
        _table = EmaInit.Normal<T>(new[] { (2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), heads }, 0.02);
        _qkv = new EmaLinear<T>(dim, 3 * dim, truncatedNormal: false);
        _proj = new EmaLinear<T>(dim, dim, truncatedNormal: false);
    }

    private Tensor<T> Heads(Tensor<T> x, int d)
    {
        int b = x.Shape[0], n = x.Shape[1];
        return Engine.Reshape(Engine.TensorPermute(Engine.Reshape(x, new[] { b, n, _heads, d }), new[] { 0, 2, 1, 3 }).Contiguous(), new[] { b * _heads, n, d });
    }

    /// <summary>Windows [B·nW, N, C] with an optional mask [nW, N, N].</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? mask)
    {
        int b = x.Shape[0], n = x.Shape[1], c = x.Shape[2], d = c / _heads;
        var qkv = _qkv.Forward(x);
        var q = Engine.TensorMultiplyScalar(Heads(Engine.TensorNarrow(qkv, 2, 0, c).Contiguous(), d), NumOps.FromDouble(Math.Pow(d, -0.5)));
        var k = Heads(Engine.TensorNarrow(qkv, 2, c, c).Contiguous(), d);
        var v = Heads(Engine.TensorNarrow(qkv, 2, 2 * c, c).Contiguous(), d);
        var attn = Engine.Reshape(Engine.BatchMatMul(q, Engine.TensorPermute(k, new[] { 0, 2, 1 }).Contiguous()), new[] { b, _heads, n, n });
        var rows = new int[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                rows[i * n + j] = _index[i * _full + j];
        var bias = Engine.TensorPermute(Engine.Reshape(Engine.TensorGather(_table.Value, new Tensor<int>(rows, new[] { n * n }), axis: 0), new[] { n, n, _heads }), new[] { 2, 0, 1 }).Contiguous();
        attn = Engine.TensorAdd(attn, Engine.TensorBroadcastTo(Engine.Reshape(bias, new[] { 1, _heads, n, n }), attn.Shape.ToArray()));
        if (mask is not null)
        {
            int nW = mask.Shape[0];
            var windows = Engine.Reshape(attn, new[] { b / nW, nW, _heads, n, n });
            windows = Engine.TensorAdd(windows, Engine.TensorBroadcastTo(Engine.Reshape(mask, new[] { 1, nW, 1, n, n }), windows.Shape.ToArray()));
            attn = Engine.Reshape(windows, new[] { b, _heads, n, n });
        }
        var weights = Engine.Softmax(Engine.Reshape(attn, new[] { b * _heads, n, n }), axis: -1);
        var y = Engine.Reshape(Engine.TensorPermute(Engine.Reshape(Engine.BatchMatMul(weights, v), new[] { b, _heads, n, d }), new[] { 0, 2, 1, 3 }).Contiguous(), new[] { b, n, c });
        return _proj.Forward(y);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _table;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _qkv, _proj };
}

/// <summary>
/// SepSTSBlock: spatial (shifted) window attention over (1, w, w) windows, then temporal attention over each pixel's
/// frames, a residual, and a GELU MLP with its own residual.
/// </summary>
internal sealed class VfitBlock<T> : LazyWeightModule<T>
{
    private readonly int _window, _shift, _frames;
    private readonly EmaLayerNorm<T> _norm1, _norm2;
    private readonly VfitAttention<T> _depthAttention, _pointAttention;
    private readonly EmaLinear<T> _fc1, _fc2;

    public VfitBlock(int dim, int heads, int window, bool shifted, int frames, double mlpRatio)
    {
        _window = window;
        _shift = shifted ? window / 2 : 0;
        _frames = frames;
        _norm1 = new EmaLayerNorm<T>(dim, 1e-5);
        _depthAttention = new VfitAttention<T>(dim, 1, window, window, heads);
        _pointAttention = new VfitAttention<T>(dim, frames, 1, 1, heads);
        _norm2 = new EmaLayerNorm<T>(dim, 1e-5);
        int hidden = (int)(dim * mlpRatio);
        _fc1 = new EmaLinear<T>(dim, hidden, truncatedNormal: false);
        _fc2 = new EmaLinear<T>(hidden, dim, truncatedNormal: false);
    }

    /// <summary>The window and shift get_window_size uses on an axis of length n.</summary>
    internal static (int Window, int Shift) Effective(int n, int window, int shift) => n <= window ? (n, 0) : (window, shift);

    /// <summary>A clip [B, D, H, W, C] and the layer's mask for the shifted blocks.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? mask)
    {
        int b = x.Shape[0], d = x.Shape[1], h = x.Shape[2], w = x.Shape[3], c = x.Shape[4];
        var (wh, sh) = Effective(h, _window, _shift);
        var (ww, sw) = Effective(w, _window, _shift);
        int padB = (wh - h % wh) % wh, padR = (ww - w % ww) % ww, hp = h + padB, wp = w + padR;
        var y = _norm1.Forward(x);
        if (padB + padR > 0) y = Engine.PadNd(y, new[] { 0, 0, 0, padR, 0, padB }, PadMode.Constant, NumOps.Zero);
        bool shifted = sh > 0 || sw > 0;
        if (shifted) y = Engine.TensorRoll(y, new[] { -sh, -sw }, new[] { 2, 3 });
        int nh = hp / wh, nw = wp / ww;
        var windows = Engine.Reshape(Engine.TensorPermute(Engine.Reshape(y, new[] { b, d, nh, wh, nw, ww, c }), new[] { 0, 1, 2, 4, 3, 5, 6 }).Contiguous(),
            new[] { b * d * nh * nw, wh * ww, c });
        var attended = _depthAttention.Forward(windows, shifted ? mask : null);
        y = Engine.Reshape(Engine.TensorPermute(Engine.Reshape(attended, new[] { b, d, nh, nw, wh, ww, c }), new[] { 0, 1, 2, 4, 3, 5, 6 }).Contiguous(),
            new[] { b, d, hp, wp, c });
        if (shifted) y = Engine.TensorRoll(y, new[] { sh, sw }, new[] { 2, 3 });
        if (padB + padR > 0) y = Engine.TensorSlice(y, new[] { 0, 0, 0, 0, 0 }, new[] { b, d, h, w, c }).Contiguous();
        int wd = d <= _frames ? d : _frames, groups = d / wd;
        var points = Engine.Reshape(Engine.TensorPermute(Engine.Reshape(y, new[] { b, groups, wd, h, w, c }), new[] { 0, 1, 3, 4, 2, 5 }).Contiguous(),
            new[] { b * groups * h * w, wd, c });
        var temporal = _pointAttention.Forward(points, null);
        y = Engine.Reshape(Engine.TensorPermute(Engine.Reshape(temporal, new[] { b, groups, h, w, wd, c }), new[] { 0, 1, 4, 2, 3, 5 }).Contiguous(),
            new[] { b, d, h, w, c });
        x = Engine.TensorAdd(x, y);
        return Engine.TensorAdd(x, _fc2.Forward(Engine.GELU(_fc1.Forward(_norm2.Forward(x)))));
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights() => Array.Empty<LazyWeight<T>>();

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _norm1, _depthAttention, _pointAttention, _norm2, _fc1, _fc2 };
}

/// <summary>SepSTSBasicLayer: the blocks, every second one shifted, over a clip [B, C, D, H, W].</summary>
internal sealed class VfitLayer<T> : LazyWeightModule<T>
{
    private readonly int _window;
    private readonly VfitBlock<T>[] _blocks;

    public VfitLayer(int dim, int depth, int heads, int window, int frames, double mlpRatio)
    {
        _window = window;
        _blocks = Enumerable.Range(0, depth).Select(i => new VfitBlock<T>(dim, heads, window, i % 2 == 1, frames, mlpRatio)).ToArray();
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        int d = x.Shape[2], h = x.Shape[3], w = x.Shape[4];
        var (wh, sh) = VfitBlock<T>.Effective(h, _window, _window / 2);
        var (ww, sw) = VfitBlock<T>.Effective(w, _window, _window / 2);
        int hp = (h + wh - 1) / wh * wh, wp = (w + ww - 1) / ww * ww;
        var mask = new Tensor<T>(VfitOps.ShiftMask(d, hp, wp, wh, ww, sh, sw).Select(v => NumOps.FromDouble(v)).ToArray(),
            new[] { d * (hp / wh) * (wp / ww), wh * ww, wh * ww });
        var y = Engine.TensorPermute(x, new[] { 0, 2, 3, 4, 1 }).Contiguous();
        foreach (var block in _blocks) y = block.Forward(y, mask);
        return Engine.TensorPermute(y, new[] { 0, 4, 1, 2, 3 }).Contiguous();
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights() => Array.Empty<LazyWeight<T>>();

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _blocks;
}

/// <summary>ResBlock: relu(conv2(relu(conv1(x))) + x) with biased 3x3x3 convolutions.</summary>
internal sealed class VfitResBlock<T> : LazyWeightModule<T>
{
    private readonly FlavrConv3d<T> _conv1, _conv2;

    public VfitResBlock(int channels)
    {
        _conv1 = new FlavrConv3d<T>(channels, channels, new[] { 3, 3, 3 }, new[] { 1, 1, 1 }, new[] { 1, 1, 1 }, true, FlavrInit.TorchDefault);
        _conv2 = new FlavrConv3d<T>(channels, channels, new[] { 3, 3, 3 }, new[] { 1, 1, 1 }, new[] { 1, 1, 1 }, true, FlavrInit.TorchDefault);
    }

    public Tensor<T> Forward(Tensor<T> x) => Engine.ReLU(Engine.TensorAdd(_conv2.Forward(Engine.ReLU(_conv1.Forward(x))), x));

    protected override IEnumerable<LazyWeight<T>> OwnWeights() => Array.Empty<LazyWeight<T>>();

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv1, _conv2 };
}

/// <summary>upSplit: ConvTranspose3d (3, 3, 3), stride (1, 2, 2), padding 1, sized to the skip it joins.</summary>
internal sealed class VfitUpSplit<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight, _bias;

    public VfitUpSplit(int inChannels, int outChannels)
    {
        // ConvTranspose3d's fan_in is weight.size(1) · kernel volume: the output channels.
        _weight = LazyWeight<T>.Uniform(new[] { inChannels, outChannels, 3, 3, 3 }, outChannels * 27);
        _bias = LazyWeight<T>.Uniform(new[] { outChannels }, outChannels * 27);
    }

    public Tensor<T> Forward(Tensor<T> x, int height, int width)
    {
        int padH = height - (2 * x.Shape[3] - 1), padW = width - (2 * x.Shape[4] - 1);
        return MgldOps.ChannelBias(Engine, Engine.ConvTranspose3D(x, _weight.Value, new[] { 1, 2, 2 }, new[] { 1, 1, 1 }, new[] { 0, padH, padW }), _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>ConvTranspose2d(3, stride 2, padding 1) sized to the output it is asked for.</summary>
internal sealed class VfitUpConv2d<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _weight, _bias;

    public VfitUpConv2d(int inChannels, int outChannels)
    {
        _weight = LazyWeight<T>.Uniform(new[] { inChannels, outChannels, 3, 3 }, outChannels * 9);
        _bias = LazyWeight<T>.Uniform(new[] { outChannels }, outChannels * 9);
    }

    public Tensor<T> Forward(Tensor<T> x, int height, int width)
    {
        int padH = height - (2 * x.Shape[2] - 1), padW = width - (2 * x.Shape[3] - 1);
        return MgldOps.ChannelBias(Engine, Engine.ConvTranspose2D(x, _weight.Value, new[] { 2, 2 }, new[] { 1, 1 }, new[] { padH, padW }), _bias.Value);
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _weight;
        yield return _bias;
    }
}

/// <summary>SynBlock's subnets: conv, LeakyReLU, conv, LeakyReLU, the sized transposed convolution and a conv.</summary>
internal sealed class VfitSubnet<T> : LazyWeightModule<T>
{
    private readonly AmtConv2d<T> _conv1, _conv2, _conv3;
    private readonly VfitUpConv2d<T> _up;

    public VfitSubnet(int inChannels, int middle, int outChannels)
    {
        _conv1 = new AmtConv2d<T>(inChannels, inChannels, 3);
        _conv2 = new AmtConv2d<T>(inChannels, middle, 3);
        _up = new VfitUpConv2d<T>(middle, middle);
        _conv3 = new AmtConv2d<T>(middle, outChannels, 3);
    }

    public Tensor<T> Forward(Tensor<T> x, int height, int width)
        => _conv3.Forward(_up.Forward(VfiformerOps.LeakyRelu(Engine, _conv2.Forward(VfiformerOps.LeakyRelu(Engine, _conv1.Forward(x)))), height, width));

    protected override IEnumerable<LazyWeight<T>> OwnWeights() => Array.Empty<LazyWeight<T>>();

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _conv1, _conv2, _up, _conv3 };
}

/// <summary>
/// SynBlock: per input frame, AdaCoF kernels (weights, softmax-normalized at the coarsest scale, and vertical and
/// horizontal offsets) from that frame's features, gated by occlusion maps softmaxed over the frames from the fused
/// features; the frames, resized to the output, are warped and summed.
/// </summary>
internal sealed class VfitSynthesis<T> : LazyWeightModule<T>
{
    private readonly int _inputs, _kernel, _dilation;
    private readonly bool _normalizeWeights;
    private readonly VfitSubnet<T> _weight, _alpha, _beta, _occlusion;
    private readonly AmtConv2d<T> _fuse;

    public VfitSynthesis(int inputs, int channels, int kernel, int dilation, bool normalizeWeights)
    {
        _inputs = inputs;
        _kernel = kernel;
        _dilation = dilation;
        _normalizeWeights = normalizeWeights;
        int taps = kernel * kernel;
        _weight = new VfitSubnet<T>(channels, taps, taps);
        _alpha = new VfitSubnet<T>(channels, taps, taps);
        _beta = new VfitSubnet<T>(channels, taps, taps);
        _occlusion = new VfitSubnet<T>(channels, channels, inputs);
        _fuse = new AmtConv2d<T>(channels * inputs, channels, 1, 1, 0);
    }

    /// <summary>Features [B, C, T, h, w] and the frames [B, 3, H0, W0] to the frame at (height, width).</summary>
    public Tensor<T> Forward(Tensor<T> features, IReadOnlyList<Tensor<T>> frames, int height, int width)
    {
        int b = features.Shape[0], c = features.Shape[1], t = features.Shape[2], fh = features.Shape[3], fw = features.Shape[4];
        var fused = VfiformerOps.LeakyRelu(Engine, _fuse.Forward(Engine.Reshape(features, new[] { b, c * t, fh, fw })));
        var occlusion = Engine.Softmax(_occlusion.Forward(fused, height, width), axis: 1);
        var perFrame = Engine.Reshape(Engine.TensorPermute(features, new[] { 0, 2, 1, 3, 4 }).Contiguous(), new[] { b * t, c, fh, fw });
        var weights = _weight.Forward(perFrame, height, width);
        if (_normalizeWeights) weights = Engine.Softmax(weights, axis: 1);
        var alphas = _alpha.Forward(perFrame, height, width);
        var betas = _beta.Forward(perFrame, height, width);
        int taps = _kernel * _kernel, pad = (_kernel - 1) * _dilation / 2;
        Tensor<T> Frame(Tensor<T> x, int i)
            => Engine.Reshape(Engine.TensorNarrow(Engine.Reshape(x, new[] { b, t, taps, height, width }), 1, i, 1).Contiguous(), new[] { b, taps, height, width });
        Tensor<T>? sum = null;
        for (int i = 0; i < _inputs; i++)
        {
            var frame = AmtOps.Resize(Engine, frames[i], height, width);
            var padded = Engine.PadNd(frame, new[] { pad, pad, pad, pad }, PadMode.Replicate, NumOps.Zero);
            var warped = VfitOps.AdaCof(Engine, padded, Frame(weights, i), Frame(alphas, i), Frame(betas, i), _dilation);
            var gated = Engine.TensorMultiply(Engine.TensorBroadcastTo(Engine.TensorNarrow(occlusion, 1, i, 1).Contiguous(), warped.Shape.ToArray()), warped);
            sum = sum is null ? gated : Engine.TensorAdd(sum, gated);
        }
        return sum ?? throw new InvalidOperationException("VFIT needs at least one input frame.");
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights() => Array.Empty<LazyWeight<T>>();

    protected override IEnumerable<LazyWeightModule<T>?> Children() => new LazyWeightModule<T>[] { _weight, _alpha, _beta, _occlusion, _fuse };
}

/// <summary>
/// UNet_3D_3D: the Sep-STS encoder (a 3x3x3 stem and residual block at full size, then four stages each halving the size
/// and running Sep-STS blocks), a transposed-convolution decoder joined to the encoder by concatenation, and AdaCoF
/// synthesis at 1/4, 1/2 and full size, each finer frame added to the coarser one upsampled.
/// </summary>
internal sealed class VfitNetwork<T> : LazyWeightModule<T>
{
    private readonly FlavrConv3d<T> _stem;
    private readonly VfitResBlock<T> _stemBlock;
    private readonly FlavrConv3d<T>[] _down;
    private readonly VfitLayer<T>[] _stages;
    private readonly VfitUpSplit<T>[] _up;
    private readonly FlavrConv3d<T>[] _smoothConv;
    private readonly VfitResBlock<T>[] _smoothBlock;
    private readonly VfitSynthesis<T>[] _predict;

    /// <param name="widths">nf: the stage widths, deepest first (512, 256, 128, 64 for VFIT-B).</param>
    public VfitNetwork(IReadOnlyList<int> widths, int frames, IReadOnlyList<int> depths, IReadOnlyList<int> heads, int window, double mlpRatio,
        int synthesisChannels, int kernel, int dilation)
    {
        int stem = widths[3] / 2;
        _stem = new FlavrConv3d<T>(3, stem, new[] { 3, 3, 3 }, new[] { 1, 1, 1 }, new[] { 1, 1, 1 }, true, FlavrInit.TorchDefault);
        _stemBlock = new VfitResBlock<T>(stem);
        var inputs = new[] { stem, widths[3], widths[2], widths[1] };
        _down = Enumerable.Range(0, 4).Select(i => new FlavrConv3d<T>(inputs[i], widths[3 - i], new[] { 3, 3, 3 }, new[] { 1, 2, 2 }, new[] { 1, 1, 1 }, true, FlavrInit.TorchDefault)).ToArray();
        _stages = Enumerable.Range(0, 4).Select(i => new VfitLayer<T>(widths[3 - i], depths[i], heads[i], window, frames, mlpRatio)).ToArray();
        _up = new[] { new VfitUpSplit<T>(widths[0], widths[1]), new VfitUpSplit<T>(2 * widths[1], widths[2]), new VfitUpSplit<T>(2 * widths[2], widths[3]) };
        _smoothConv = Enumerable.Range(0, 3).Select(i => new FlavrConv3d<T>(2 * widths[i + 1], synthesisChannels, new[] { 3, 3, 3 }, new[] { 1, 1, 1 }, new[] { 1, 1, 1 }, true, FlavrInit.TorchDefault)).ToArray();
        _smoothBlock = Enumerable.Range(0, 3).Select(_ => new VfitResBlock<T>(synthesisChannels)).ToArray();
        _predict = Enumerable.Range(0, 3).Select(i => new VfitSynthesis<T>(frames, synthesisChannels, kernel, dilation, i == 0)).ToArray();
    }

    /// <summary>The frames [B, 3, H, W] in [0, 1], earliest first, to the middle frame.</summary>
    public Tensor<T> Forward(IReadOnlyList<Tensor<T>> frames, IDictionary<string, Tensor<T>>? activations = null)
    {
        int b = frames[0].Shape[0], h = frames[0].Shape[2], w = frames[0].Shape[3];
        var clip = Engine.TensorConcatenate(frames.Select(f => Engine.Reshape(f, new[] { b, 3, 1, h, w })).ToArray(), 2);
        // CAIN's batch mean normalization: each sample's per-channel mean over the clip.
        var mean = Engine.ReduceMean(clip, new[] { 2, 3, 4 }, true);
        clip = Engine.TensorSubtract(clip, Engine.TensorBroadcastTo(mean, clip.Shape.ToArray()));
        var x0 = _stemBlock.Forward(VfiformerOps.LeakyRelu(Engine, _stem.Forward(clip)));
        var skips = new Tensor<T>[5];
        skips[0] = x0;
        for (int i = 0; i < 4; i++) skips[i + 1] = _stages[i].Forward(_down[i].Forward(skips[i]));
        var decoded = new Tensor<T>[3];
        var current = skips[4];
        for (int i = 0; i < 3; i++)
        {
            var skip = skips[3 - i];
            var up = VfiformerOps.LeakyRelu(Engine, _up[i].Forward(current, skip.Shape[3], skip.Shape[4]));
            current = Engine.TensorConcatenate(new[] { up, skip }, 1);
            decoded[i] = current;
        }
        Tensor<T>? output = null;
        for (int i = 0; i < 3; i++)
        {
            var size = skips[2 - i];
            var features = _smoothBlock[i].Forward(_smoothConv[i].Forward(decoded[i]));
            var frame = _predict[i].Forward(features, frames, size.Shape[3], size.Shape[4]);
            output = output is null ? frame : Engine.TensorAdd(AmtOps.Resize(Engine, output, frame.Shape[2], frame.Shape[3]), frame);
            if (activations is not null) activations[$"Synthesis{i}"] = output;
        }
        if (activations is not null)
            for (int i = 0; i < 5; i++) activations[$"Encoder{i}"] = skips[i];
        return output ?? throw new InvalidOperationException("VFIT's synthesis produced no frame.");
    }

    protected override IEnumerable<LazyWeight<T>> OwnWeights() => Array.Empty<LazyWeight<T>>();

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _stem, _stemBlock }.Concat(_down).Concat(_stages).Concat(_up).Concat(_smoothConv).Concat(_smoothBlock).Concat(_predict);
}
