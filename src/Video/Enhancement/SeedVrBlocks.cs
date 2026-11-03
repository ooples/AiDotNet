using AiDotNet.ActivationFunctions;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// SeedVR's attention windows (<c>models/dit/window.py</c>, the <c>720pwin_by_size_bysize</c> and
/// <c>720pswin_by_size_bysize</c> methods): the token grid is cut into a window count fixed at 720p, (4, 3, 3) by default, so a
/// window covers the same share of the frame at any resolution; the shifted variant offsets every axis that has more than one
/// window by half a window, leaving partial windows at the edges. The windows are the product of one interval list per axis.
/// </summary>
internal sealed class SeedVrWindows
{
    private SeedVrWindows(IReadOnlyList<(int Start, int Length)> time, IReadOnlyList<(int Start, int Length)> height,
        IReadOnlyList<(int Start, int Length)> width)
    {
        Time = time;
        Height = height;
        Width = width;
    }

    /// <summary>The time intervals.</summary>
    internal IReadOnlyList<(int Start, int Length)> Time { get; }

    /// <summary>The height intervals.</summary>
    internal IReadOnlyList<(int Start, int Length)> Height { get; }

    /// <summary>The width intervals.</summary>
    internal IReadOnlyList<(int Start, int Length)> Width { get; }

    /// <summary>The number of windows.</summary>
    internal int Count => Time.Count * Height.Count * Width.Count;

    /// <param name="t">The token grid's frames.</param>
    /// <param name="h">Its height.</param>
    /// <param name="w">Its width.</param>
    /// <param name="counts">The window counts (time, height, width) at 720p.</param>
    /// <param name="shifted">Whether the windows are shifted by half.</param>
    internal static SeedVrWindows Build(int t, int h, int w, (int T, int H, int W) counts, bool shifted)
    {
        // The 720p latent grid is 45 x 80 tokens; Python's round() rounds halves to even, as Math.Round does.
        double scale = Math.Sqrt(45.0 * 80.0 / (h * w));
        int resizedH = (int)Math.Round(h * scale), resizedW = (int)Math.Round(w * scale);
        int wh = (resizedH + counts.H - 1) / counts.H, ww = (resizedW + counts.W - 1) / counts.W;
        int wt = (Math.Min(t, 30) + counts.T - 1) / counts.T;
        return new SeedVrWindows(Axis(t, wt, shifted), Axis(h, wh, shifted), Axis(w, ww, shifted));
    }

    private static IReadOnlyList<(int, int)> Axis(int size, int window, bool shifted)
    {
        var intervals = new List<(int, int)>();
        if (!shifted)
        {
            int n = (size + window - 1) / window;
            for (int i = 0; i < n; i++)
            {
                int start = i * window, end = Math.Min((i + 1) * window, size);
                if (end > start) intervals.Add((start, end - start));
            }
            return intervals;
        }
        double shift = window < size ? 0.5 : 0.0;
        int count = shift > 0 ? (int)Math.Ceiling((size - shift) / window) + 1 : 1;
        for (int i = 0; i < count; i++)
        {
            // int() truncates toward zero, as the casts do.
            int start = Math.Max((int)((i - shift) * window), 0), end = Math.Min((int)((i - shift + 1) * window), size);
            if (end > start) intervals.Add((start, end - start));
        }
        return intervals;
    }
}

/// <summary>
/// SeedVR's 3D rotary embedding (<c>RotaryEmbedding3d</c> on <c>rotary_embedding_torch</c> with <c>freqs_for="pixel"</c>,
/// <c>max_freq=256</c>): each axis of a window gets positions spread over [−1, 1] and ⌊⌊headDim/2⌋/3⌋/2 frequencies from π to
/// 128π, each rotating an adjacent pair; the time, height and width pairs fill the first features and the rest pass through.
/// </summary>
internal static class SeedVrRope
{
    /// <summary>The cosines and sines [1, t·h·w, headDim] of a (t, h, w) window, in that token order.</summary>
    internal static (Tensor<T> Cos, Tensor<T> Sin) Build<T>(int headDim, int t, int h, int w)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int axisDim = headDim / 2 / 3, frequencies = axisDim / 2;
        var freqs = new double[frequencies];
        for (int j = 0; j < frequencies; j++)
            freqs[j] = (frequencies == 1 ? 1.0 : 1.0 + j * (128.0 - 1.0) / (frequencies - 1)) * Math.PI;
        static double Position(int i, int n) => n == 1 ? -1.0 : -1.0 + 2.0 * i / (n - 1);
        int tokens = t * h * w;
        var cos = new Tensor<T>(new[] { 1, tokens, headDim });
        var sin = new Tensor<T>(new[] { 1, tokens, headDim });
        for (int it = 0; it < t; it++)
            for (int ih = 0; ih < h; ih++)
                for (int iw = 0; iw < w; iw++)
                {
                    int row = ((it * h + ih) * w + iw) * headDim;
                    var positions = new[] { Position(it, t), Position(ih, h), Position(iw, w) };
                    for (int d = 0; d < headDim; d++) cos[row + d] = ops.One;
                    for (int axis = 0; axis < 3; axis++)
                        for (int j = 0; j < frequencies; j++)
                        {
                            double angle = positions[axis] * freqs[j];
                            int at = row + 2 * (axis * frequencies + j);
                            for (int r = 0; r < 2; r++)
                            {
                                cos[at + r] = ops.FromDouble(Math.Cos(angle));
                                sin[at + r] = ops.FromDouble(Math.Sin(angle));
                            }
                        }
                }
        return (cos, sin);
    }
}

/// <summary>
/// The adaptive modulation of one branch of a block (<c>AdaSingle</c>): for the attention and the MLP, a learned shift, scale and
/// gate added to the slices of the shared time embedding laid out as <c>b (d l g)</c>.
/// </summary>
internal sealed class SeedVrAda<T>
{
    private readonly int _dim;
    private readonly int[] _shift = new int[2], _scale = new int[2], _gate = new int[2];

    internal SeedVrAda(List<ILayer<T>> layers, int dim)
    {
        _dim = dim;
        double std = 1.0 / Math.Sqrt(dim);
        for (int l = 0; l < 2; l++)
        {
            _shift[l] = layers.Count;
            layers.Add(Parameter(dim, 0.0, std));
            _scale[l] = layers.Count;
            layers.Add(Parameter(dim, 1.0, std));
            _gate[l] = layers.Count;
            layers.Add(Parameter(dim, 0.0, std));
        }
    }

    private static DenseLayer<T> Parameter(int dim, double mean, double std)
        => new(dim, (IActivationFunction<T>?)null, new NormalInitializationStrategy<T>(mean, std), AiDotNet.Enums.BiasMode.Never);

    // Slice g of layer l of emb [B, 6D] laid out (d, l, g), as [B, 1, D].
    private Tensor<T> Emb(IEngine engine, Tensor<T> emb, int l, int g)
    {
        int b = emb.Shape[0];
        var grid = engine.Reshape(emb, new[] { b, _dim, 2, 3 });
        var slice = engine.TensorNarrow(engine.TensorNarrow(grid, 2, l, 1), 3, g, 1);
        return engine.Reshape(slice, new[] { b, 1, _dim });
    }

    private Tensor<T> Sum(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> emb, int l, int g, int index, int[] shape)
    {
        var learned = engine.Reshape(RealViformerBlocks.Read(layers, index), new[] { 1, 1, _dim });
        var a = Emb(engine, emb, l, g);
        return engine.TensorBroadcastTo(engine.TensorAdd(a, engine.TensorBroadcastTo(learned, a.Shape.ToArray())), shape);
    }

    /// <summary>mode="in": x·(scaleA + scaleB) + (shiftA + shiftB).</summary>
    internal Tensor<T> In(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, Tensor<T> emb, int layer)
    {
        var shape = x.Shape.ToArray();
        return engine.TensorAdd(engine.TensorMultiply(x, Sum(engine, layers, emb, layer, 1, _scale[layer], shape)),
            Sum(engine, layers, emb, layer, 0, _shift[layer], shape));
    }

    /// <summary>mode="out": x·(gateA + gateB).</summary>
    internal Tensor<T> Out(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, Tensor<T> emb, int layer)
        => engine.TensorMultiply(x, Sum(engine, layers, emb, layer, 2, _gate[layer], x.Shape.ToArray()));
}

/// <summary>One branch's (video's or text's) weights in a block.</summary>
internal sealed class SeedVrBranch<T>
{
    internal SeedVrBranch(List<ILayer<T>> layers, int dim, int heads, int headDim, int expandRatio, bool qkBias)
    {
        int inner = heads * headDim;
        Qkv = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(3 * inner, bias: qkBias));
        NormQ = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(headDim, 1.0));
        NormK = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(headDim, 1.0));
        Out = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        MlpIn = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim * expandRatio));
        MlpOut = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        Ada = new SeedVrAda<T>(layers, dim);
    }

    internal int Qkv { get; }

    internal int NormQ { get; }

    internal int NormK { get; }

    internal int Out { get; }

    internal int MlpIn { get; }

    internal int MlpOut { get; }

    internal SeedVrAda<T> Ada { get; }
}

/// <summary>
/// One SeedVR block (<c>NaMMSRTransformerBlock</c>): an MM-DiT block whose video and text branches have their own weights. Both
/// are RMS-normed and modulated, then attend jointly inside each window: every window's video tokens attend to themselves and
/// the whole prompt (the 3D rotary embedding on the video queries and keys at window-local positions), the text tokens attend
/// within each window too and their outputs are averaged over the windows. A GELU(tanh) MLP follows, each half gated.
/// </summary>
internal sealed class SeedVrBlock<T>
{
    private readonly int _heads, _headDim;
    private readonly double _epsilon;
    private readonly bool _shifted;
    private readonly SeedVrBranch<T> _vid, _txt;
    private readonly GELUActivation<T> _gelu = new();

    internal SeedVrBlock(List<ILayer<T>> layers, int dim, int heads, int headDim, int expandRatio, double epsilon, bool qkBias, bool shifted)
    {
        FirstLayer = layers.Count;
        _heads = heads;
        _headDim = headDim;
        _epsilon = epsilon;
        _shifted = shifted;
        _vid = new SeedVrBranch<T>(layers, dim, heads, headDim, expandRatio, qkBias);
        _txt = new SeedVrBranch<T>(layers, dim, heads, headDim, expandRatio, qkBias);
        EndLayer = layers.Count;
    }

    /// <summary>The index of the block's first layer.</summary>
    internal int FirstLayer { get; }

    /// <summary>One past the index of the block's last layer.</summary>
    internal int EndLayer { get; }

    /// <summary>Whether the block's windows are the shifted ones.</summary>
    internal bool Shifted => _shifted;

    // RMS norm over the last axis without a learned scale.
    private Tensor<T> Rms(IEngine engine, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = x.Shape.ToArray();
        var meanSquare = engine.ReduceMean(engine.TensorSquare(x), new[] { shape.Length - 1 }, true);
        return engine.TensorDivide(x, engine.TensorBroadcastTo(engine.TensorSqrt(engine.TensorAddScalar(meanSquare, ops.FromDouble(_epsilon))), shape));
    }

    // [B, L, 3·H·hd] to q, k, v [B, L, H, hd] with q and k RMS-normed per head.
    private (Tensor<T> Q, Tensor<T> K, Tensor<T> V) Qkv(IEngine engine, IReadOnlyList<ILayer<T>> layers, SeedVrBranch<T> branch, Tensor<T> x)
    {
        int b = x.Shape[0], l = x.Shape[1];
        var qkv = engine.Reshape(layers[branch.Qkv].Forward(x), new[] { b, l, 3, _heads, _headDim });
        Tensor<T> Part(int o) => engine.Reshape(engine.TensorNarrow(qkv, 2, o, 1), new[] { b, l, _heads, _headDim });
        return (WanOps.RmsNorm(engine, layers, branch.NormQ, Part(0), _epsilon), WanOps.RmsNorm(engine, layers, branch.NormK, Part(1), _epsilon),
            Part(2));
    }

    // [B, L, H, hd] to [B·H, L, hd].
    private static Tensor<T> Heads(IEngine engine, Tensor<T> x)
    {
        int b = x.Shape[0], l = x.Shape[1], h = x.Shape[2], d = x.Shape[3];
        return engine.Reshape(engine.TensorPermute(x, new[] { 0, 2, 1, 3 }), new[] { b * h, l, d });
    }

    private Tensor<T> Attention(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> vid, Tensor<T> txt, (int T, int H, int W) grid,
        SeedVrWindowPlan<T> plan, out Tensor<T> txtOut)
    {
        int b = vid.Shape[0], lt = txt.Shape[1], inner = _heads * _headDim;
        var (vq, vk, vv) = Qkv(engine, layers, _vid, vid);
        var (tq, tk, tv) = Qkv(engine, layers, _txt, txt);
        var windows = _shifted ? plan.Shifted : plan.Regular;
        var shape = new[] { b, grid.T, grid.H, grid.W, _heads, _headDim };
        vq = engine.Reshape(vq, shape);
        vk = engine.Reshape(vk, shape);
        vv = engine.Reshape(vv, shape);
        var (tqh, tkh, tvh) = (Heads(engine, tq), Heads(engine, tk), Heads(engine, tv));

        Tensor<T>? txtSum = null;
        var timeSlabs = new List<Tensor<T>>();
        foreach (var (t0, tn) in windows.Time)
        {
            var rows = new List<Tensor<T>>();
            foreach (var (h0, hn) in windows.Height)
            {
                var cells = new List<Tensor<T>>();
                foreach (var (w0, wn) in windows.Width)
                {
                    int n = tn * hn * wn;
                    Tensor<T> Window(Tensor<T> x)
                    {
                        var box = engine.TensorNarrow(engine.TensorNarrow(engine.TensorNarrow(x, 1, t0, tn), 2, h0, hn), 3, w0, wn);
                        return Heads(engine, engine.Reshape(box, new[] { b, n, _heads, _headDim }));
                    }
                    var (cos, sin) = plan.Rope(tn, hn, wn);
                    var q = CogVideoXRope.Apply(engine, Window(vq), cos, sin);
                    var k = CogVideoXRope.Apply(engine, Window(vk), cos, sin);
                    var attended = WanOps.Attend(engine,
                        engine.TensorConcatenate(new[] { q, tqh }, 1),
                        engine.TensorConcatenate(new[] { k, tkh }, 1),
                        engine.TensorConcatenate(new[] { Window(vv), tvh }, 1), null);
                    var windowTxt = engine.TensorNarrow(attended, 1, n, lt);
                    txtSum = txtSum is null ? windowTxt : engine.TensorAdd(txtSum, windowTxt);
                    // [B·H, n, hd] back to [B, tn, hn, wn, H·hd].
                    var cell = engine.TensorPermute(engine.Reshape(engine.TensorNarrow(attended, 1, 0, n), new[] { b, _heads, tn, hn, wn, _headDim }),
                        new[] { 0, 2, 3, 4, 1, 5 });
                    cells.Add(engine.Reshape(cell, new[] { b, tn, hn, wn, inner }));
                }
                rows.Add(cells.Count == 1 ? cells[0] : engine.TensorConcatenate(cells.ToArray(), 3));
            }
            timeSlabs.Add(rows.Count == 1 ? rows[0] : engine.TensorConcatenate(rows.ToArray(), 2));
        }
        var vidOut = timeSlabs.Count == 1 ? timeSlabs[0] : engine.TensorConcatenate(timeSlabs.ToArray(), 1);
        vidOut = engine.Reshape(vidOut, new[] { b, grid.T * grid.H * grid.W, inner });
        var ops = MathHelper.GetNumericOperations<T>();
        var txtMean = engine.TensorMultiplyScalar(txtSum ?? throw new InvalidOperationException("SeedVR's token grid has no windows."),
            ops.FromDouble(1.0 / windows.Count));
        txtOut = layers[_txt.Out].Forward(engine.Reshape(engine.TensorPermute(engine.Reshape(txtMean, new[] { b, _heads, lt, _headDim }),
            new[] { 0, 2, 1, 3 }), new[] { b, lt, inner }));
        return layers[_vid.Out].Forward(vidOut);
    }

    /// <param name="engine">The engine.</param>
    /// <param name="layers">The model's layers.</param>
    /// <param name="vid">Video tokens [B, T·H·W, D] in (t, h, w) order.</param>
    /// <param name="txt">Text tokens [B, Lt, D].</param>
    /// <param name="emb">The time embedding [B, 6D].</param>
    /// <param name="grid">The token grid.</param>
    /// <param name="plan">The windows and rotary tables for the grid.</param>
    internal (Tensor<T> Vid, Tensor<T> Txt) Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> vid, Tensor<T> txt, Tensor<T> emb,
        (int T, int H, int W) grid, SeedVrWindowPlan<T> plan)
    {
        var vidAttn = _vid.Ada.In(engine, layers, Rms(engine, vid), emb, 0);
        var txtAttn = _txt.Ada.In(engine, layers, Rms(engine, txt), emb, 0);
        var vidUpdate = Attention(engine, layers, vidAttn, txtAttn, grid, plan, out var txtUpdate);
        vid = engine.TensorAdd(_vid.Ada.Out(engine, layers, vidUpdate, emb, 0), vid);
        txt = engine.TensorAdd(_txt.Ada.Out(engine, layers, txtUpdate, emb, 0), txt);
        return (Mlp(engine, layers, _vid, vid, emb), Mlp(engine, layers, _txt, txt, emb));
    }

    private Tensor<T> Mlp(IEngine engine, IReadOnlyList<ILayer<T>> layers, SeedVrBranch<T> branch, Tensor<T> x, Tensor<T> emb)
    {
        var h = branch.Ada.In(engine, layers, Rms(engine, x), emb, 1);
        h = layers[branch.MlpOut].Forward(_gelu.Activate(layers[branch.MlpIn].Forward(h)));
        return engine.TensorAdd(branch.Ada.Out(engine, layers, h, emb, 1), x);
    }
}

/// <summary>The regular and shifted windows of one token grid with the rotary tables of each window shape.</summary>
internal sealed class SeedVrWindowPlan<T>
{
    private readonly int _headDim;
    private readonly Dictionary<(int, int, int), (Tensor<T>, Tensor<T>)> _rope = new();

    internal SeedVrWindowPlan(int t, int h, int w, (int T, int H, int W) counts, int headDim)
    {
        _headDim = headDim;
        Regular = SeedVrWindows.Build(t, h, w, counts, false);
        Shifted = SeedVrWindows.Build(t, h, w, counts, true);
    }

    internal SeedVrWindows Regular { get; }

    internal SeedVrWindows Shifted { get; }

    internal (Tensor<T> Cos, Tensor<T> Sin) Rope(int t, int h, int w)
    {
        if (!_rope.TryGetValue((t, h, w), out var tables))
        {
            tables = SeedVrRope.Build<T>(_headDim, t, h, w);
            _rope[(t, h, w)] = tables;
        }
        return tables;
    }
}

/// <summary>
/// SeedVR's <c>NaDiT</c> (<c>models/dit/nadit.py</c>): a (1, 2, 2) patch embedding of the noisy latent concatenated with the
/// condition latent and its mask, the prompt projected to the model width, a time embedding (the 256-wide sin-cos sinusoid
/// through Linear-SiLU-Linear-SiLU-Linear to six widths) shared by every block, the blocks alternating regular and shifted
/// windows, and a linear map back to the patches.
/// </summary>
internal sealed class SeedVrDiT<T>
{
    private const int SinusoidalDim = 256;
    private readonly int _patchIn, _txtIn, _embIn, _embHid, _embOut, _patchOut;
    private readonly int _dim, _heads, _headDim, _outChannels;
    private readonly (int T, int H, int W) _windows;
    private readonly SeedVrBlock<T>[] _blocks;

    internal SeedVrDiT(List<ILayer<T>> layers, int outChannels, int dim, int heads, int headDim, int expandRatio, int depth,
        double epsilon, bool qkBias, (int T, int H, int W) windows)
    {
        _dim = dim;
        _heads = heads;
        _headDim = headDim;
        _outChannels = outChannels;
        _windows = windows;
        _patchIn = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        _txtIn = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        _embIn = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        _embHid = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(dim));
        _embOut = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(6 * dim));
        _blocks = new SeedVrBlock<T>[depth];
        for (int i = 0; i < depth; i++) _blocks[i] = new SeedVrBlock<T>(layers, dim, heads, headDim, expandRatio, epsilon, qkBias, i % 2 == 1);
        _patchOut = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(outChannels * 4));
    }

    /// <summary>The blocks.</summary>
    internal IReadOnlyList<SeedVrBlock<T>> Blocks => _blocks;

    /// <summary>diffusers <c>get_timestep_embedding(flip_sin_to_cos=False, downscale_freq_shift=0)</c>: [sin, cos].</summary>
    internal static Tensor<T> Sinusoid(IReadOnlyList<double> timesteps, int dim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int half = dim / 2;
        var emb = new Tensor<T>(new[] { timesteps.Count, dim });
        for (int b = 0; b < timesteps.Count; b++)
            for (int i = 0; i < half; i++)
            {
                double angle = timesteps[b] * Math.Exp(-Math.Log(10000.0) * i / half);
                emb[b * dim + i] = ops.FromDouble(Math.Sin(angle));
                emb[b * dim + half + i] = ops.FromDouble(Math.Cos(angle));
            }
        return emb;
    }

    /// <param name="engine">The engine.</param>
    /// <param name="layers">The model's layers.</param>
    /// <param name="input">[B, C, F, H, W] (noise, condition and mask channels) with H and W even.</param>
    /// <param name="prompt">[B, Lt, textDim] prompt embeddings.</param>
    /// <param name="timesteps">One timestep in [0, 1000] per sample.</param>
    /// <returns>The prediction [B, outChannels, F, H, W].</returns>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> input, Tensor<T> prompt, IReadOnlyList<double> timesteps)
    {
        int b = input.Shape[0], c = input.Shape[1], f = input.Shape[2], h = input.Shape[3] / 2, w = input.Shape[4] / 2;
        // '(T t) (H h) (W w) c -> T H W (t h w c)' with patch (1, 2, 2).
        var patches = engine.Reshape(input, new[] { b, c, f, h, 2, w, 2 });
        patches = engine.Reshape(engine.TensorPermute(patches, new[] { 0, 2, 3, 5, 4, 6, 1 }), new[] { b, f * h * w, 4 * c });
        var vid = layers[_patchIn].Forward(patches);
        var txt = layers[_txtIn].Forward(prompt);
        var emb = engine.Swish(layers[_embIn].Forward(Sinusoid(timesteps, SinusoidalDim)));
        emb = layers[_embOut].Forward(engine.Swish(layers[_embHid].Forward(emb)));
        var plan = new SeedVrWindowPlan<T>(f, h, w, _windows, _headDim);
        foreach (var block in _blocks) (vid, txt) = block.Forward(engine, layers, vid, txt, emb, (f, h, w), plan);
        // 'T H W (t h w c) -> (T t) (H h) (W w) c'.
        var output = engine.Reshape(layers[_patchOut].Forward(vid), new[] { b, f, h, w, 2, 2, _outChannels });
        output = engine.TensorPermute(output, new[] { 0, 6, 1, 2, 4, 3, 5 });
        return engine.Reshape(output, new[] { b, _outChannels, f, 2 * h, 2 * w });
    }
}
